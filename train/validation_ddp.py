import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
sys.path.append('..')
import argparse

import cv2
import torch
import numpy as np
from PIL import Image

from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_video
from controlnet_aux import HEDdetector, CannyDetector, MidasDetector

import gc
import wandb
from typing import Any, Dict, Union, List, Optional
from accelerate import Accelerator
from diffusers.utils.torch_utils import is_compiled_module

from pathlib import Path
import transformers
import diffusers

from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs, ProjectConfiguration, set_seed
from datetime import timedelta

# keep your repo paths
sys.path.append('..')
from wan_transformer import CustomWanTransformer3DModel
from wan_controlnet import WanControlnet
from safetensors.torch import load
from Wan_controlnet_img2vid_pipeline import WanImageToVideoControlnetPipeline

from tqdm.auto import tqdm


def reset_memory(device: Union[str, torch.device]) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.reset_accumulated_memory_stats(device)


def print_memory(device: Union[str, torch.device]) -> None:
    memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
    max_memory_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
    max_memory_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
    print(f"{memory_allocated=:.3f} GB")
    print(f"{max_memory_allocated=:.3f} GB")
    print(f"{max_memory_reserved=:.3f} GB")
    print(torch.cuda.memory_summary(device=device, abbreviated=True))
    print()


def unwrap_model(accelerator: Accelerator, model):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def apply_gaussian_blur(image, ksize=5, sigmaX=1.0):
    image_np = np.array(image)
    if ksize % 2 == 0:
        ksize += 1
    blurred_image = cv2.GaussianBlur(image_np, (ksize, ksize), sigmaX=sigmaX)
    return Image.fromarray(blurred_image)

class TilePreprocessor:
    def __call__(self, image, target_h, target_w, ksize=5, downscale_coef=4):
        img = image.resize((target_w // downscale_coef, target_h // downscale_coef))
        img = apply_gaussian_blur(img, ksize=ksize, sigmaX=ksize // 2)
        return img.resize((target_w, target_h))

def init_controlnet_processor(controlnet_type):
    if controlnet_type in ['canny', 'tile']:
        return controlnet_mapping[controlnet_type]()
    return controlnet_mapping[controlnet_type].from_pretrained('lllyasviel/Annotators').to(device='cuda')


controlnet_mapping = {
    'canny': CannyDetector,
    'hed': HEDdetector,
    'depth': MidasDetector,
    'tile': TilePreprocessor
}

def _rank_info(accelerator: Accelerator):
    return accelerator.local_process_index, accelerator.num_processes

def _sanitize_for_path(s: str, limit: int = 50):
    s = (s or "")[:limit]
    for ch in " '\"/:*?<>|":
        s = s.replace(ch, "_")
    return s

def log_validation(
    accelerator: Accelerator,
    pipe,
    args: Dict[str, Any],
    all_pipeline_args: List[Dict[str, Any]],
    is_final_validation: bool = False,
    prompts_to_log: List[str] = [],
    validate: bool = False,
    name_suffixes: Optional[List[str]] = None,
):
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
    phase_name = "test" if is_final_validation else "validation"

    for i, pipeline_kwargs in enumerate(all_pipeline_args):
        for j in range(args.num_validation_videos):
            prompt_to_log = prompts_to_log[i]
            prompt = _sanitize_for_path(prompt_to_log, limit=25)
            prefix = f"{name_suffixes[i]}_" if name_suffixes and i < len(name_suffixes) else ""
            cfg = pipeline_kwargs['guidance_scale']
            cntw = pipeline_kwargs['controlnet_weight']
            filename = os.path.join(args.output_path, f"{phase_name}_{prefix}video_{i}_{j}_{cfg=}_{cntw=}_{prompt}.mp4")
            if os.path.exists(filename):
                continue
            
            video = pipe(**pipeline_kwargs, generator=generator, output_type="np").frames[0]
            export_to_video(video, filename, fps=8)
            del video


def load_latent(path: str, device) -> torch.Tensor:
    return torch.load(path, map_location=device)

def check_pipeline_devices(pipe):
    for name, component in pipe.components.items():
        if hasattr(component, 'device'):
            print(f"{name}: {component.device}")
        else:
            print(f"{name}: No device attribute (likely CPU-based)")


@torch.no_grad()
def validate_videos(args, accelerator, transformer, controlnet, scheduler, weight_dtype, validate=False):
    # Parallel banner
    rank, world = _rank_info(accelerator)
    print(f"[rank {rank}] validate={validate} WORLD_SIZE={world} "
          f"CUDA available={torch.cuda.is_available()} device_count={torch.cuda.device_count()} "
          f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES')} "
          f"ROCR_VISIBLE_DEVICES={os.environ.get('ROCR_VISIBLE_DEVICES')}", flush=True)

    accelerator.print("===== Memory before validation =====")
    print_memory(accelerator.device)
    torch.cuda.synchronize(accelerator.device)
    torch.cuda.empty_cache()

    vae = AutoencoderKLWan.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", torch_dtype=torch.float32)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    vae = vae.to(accelerator.device).eval()

    pipe = WanImageToVideoControlnetPipeline(
        transformer=unwrap_model(accelerator, transformer).eval(),
        vae=vae,
        controlnet=unwrap_model(accelerator, controlnet).eval(),
        scheduler=scheduler,
        expand_timesteps=True,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=5.0)

    if ';' in args.validation_images:
        validation_images = args.validation_images.split(';')
        validation_prompts = args.validation_prompt.split(';')
        validation_prompts_log = args.validation_prompt_log.split(';') if ';' in args.validation_prompt_log else open(args.validation_prompt_log).readlines()
        validation_videos = args.video_path.split(';')
    else:
        validation_images = [os.path.join(args.validation_images, pth) for pth in os.listdir(args.validation_images)]
        validation_prompts = [os.path.join(args.validation_prompt, pth) for pth in os.listdir(args.validation_prompt)]
        validation_videos = [os.path.join(args.video_path, pth) for pth in os.listdir(args.video_path)]
        
        validation_prompts_log = open(args.validation_prompt_log).readlines()
        assert len(validation_images) == len(validation_prompts) == len(validation_videos) == len(validation_prompts_log), 'wrong length'

        # validation_images = sorted(validation_images, key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[1]))
        # validation_prompts = sorted(validation_prompts, key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[1]))
        # validation_videos = sorted(validation_videos, key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[1]))

        validation_images = sorted(validation_images, key=lambda x: int(os.path.basename(x).split('.pt')[0].split('_')[-1]))
        validation_prompts = sorted(validation_prompts, key=lambda x: int(os.path.basename(x).split('.pt')[0].split('_')[-1]))
        validation_videos = sorted(validation_videos, key=lambda x: int(os.path.basename(x).split('.mp4')[0].split('_')[-1]))

    # Shard across ranks when validate=True
    total_items = len(validation_images)
    shard_idx = [i for i in range(total_items)] if (not validate or world == 1) else [i for i in range(total_items) if (i % world) == rank]

    shard_validation_images = [validation_images[i] for i in shard_idx]
    shard_validation_prompts = [validation_prompts[i] for i in shard_idx]
    shard_validation_videos = [validation_videos[i] for i in shard_idx]
    shard_validation_prompts_log = [validation_prompts_log[i].strip() for i in shard_idx]
    name_suffixes = [f"gi{i}_r{rank}" for i in shard_idx]  # unique filenames per-rank

    print(f"[rank {rank}] Assigned {len(shard_idx)}/{total_items} items. Sample idx: {shard_idx[:6]}", flush=True)

    # Per-rank progress bar
    progress_bar = tqdm(range(0, len(shard_validation_images)),
                        desc=f"Num val vids (rank {rank})",
                        position=rank, disable=False)

    negative_prompt_latent = load_latent(path=args.negative_prompt_latent, device=accelerator.device)

    base_pipeline_args = {
        "negative_prompt_embeds": negative_prompt_latent,
        "height": args.video_height,
        "width": args.video_width,
        "num_frames": args.num_frames,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "controlnet_guidance_start": args.controlnet_guidance_start,
        "controlnet_guidance_end": args.controlnet_guidance_end,
        "controlnet_weight": args.controlnet_weights,
        "controlnet_stride": args.controlnet_stride,
        "teacache_treshold": args.teacache_treshold,
    }

    for gi, (validation_video, validation_prompt, validation_image_path, validation_prompt_log) in enumerate(
        zip(shard_validation_videos, shard_validation_prompts, shard_validation_images, shard_validation_prompts_log)
    ):
        controlnet_frames = load_video(validation_video)
        validation_image_latent = load_latent(path=validation_image_path, device=accelerator.device)
        validation_prompt_latent = load_latent(path=validation_prompt, device=accelerator.device)

        current_gen_kwargs = {
            **base_pipeline_args,
            "image_latents": validation_image_latent,
            "prompt_embeds": validation_prompt_latent,
            "controlnet_frames": controlnet_frames,
            "stage_2": args.stage_2_training,
        }

        log_validation(
            accelerator=accelerator,
            pipe=pipe,
            args=args,
            all_pipeline_args=[current_gen_kwargs],
            prompts_to_log=[validation_prompt_log],
            validate=validate,
            name_suffixes=[name_suffixes[gi]],
        )
        progress_bar.update(1)

        del controlnet_frames, validation_image_latent, validation_prompt_latent

    transformer.train()
    controlnet.train()

    accelerator.print("===== Memory after validation =====")
    print_memory(accelerator.device)
    reset_memory(accelerator.device)
    torch.cuda.synchronize(accelerator.device)


def get_args():
    parser = argparse.ArgumentParser(description="Validation / generation for Wan2.2 ControlNet (multi-GPU capable).")
    parser.add_argument("--num_validation_videos", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_path", type=str, default="./outputs")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None, required=True)

    parser.add_argument("--enable_slicing", action="store_true", default=False)
    parser.add_argument("--enable_tiling", action="store_true", default=False)
    parser.add_argument("--enable_model_cpu_offload", action="store_true", default=False)

    parser.add_argument("--validation_images", type=str, required=True)
    parser.add_argument("--validation_prompt", type=str, required=True)
    parser.add_argument("--validation_prompt_log", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)

    parser.add_argument("--video_height", type=int, default=480)
    parser.add_argument("--video_width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--negative_prompt", type=str, default="bad quality, worst quality")
    parser.add_argument("--negative_prompt_latent", type=str, default="negative_prompt_latent.pt")

    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--controlnet_guidance_end", type=float, default=0.8)
    parser.add_argument("--controlnet_weights", type=float, default=1.0)
    parser.add_argument("--controlnet_stride", type=int, default=1)
    parser.add_argument("--teacache_treshold", type=float, default=0.0)
    parser.add_argument("--output_dir", type=str, default="wan2.2-controlnet")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--report_to", type=str, default=None)
    parser.add_argument("--controlnet_input_channels", type=int, default=3)
    parser.add_argument("--controlnet_transformer_num_layers", type=int, default=2)
    parser.add_argument("--downscale_coef", type=int, default=16)
    parser.add_argument("--vae_channels", type=int, default=16)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--tracker_name", type=str, default=None)
    parser.add_argument("--init_from_transformer", action="store_true")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--wandb_key", type=str, default=None)
    parser.add_argument("--controlnet_guidance_start", type=float, default=0.0)
    parser.add_argument("--stage_2_training", action="store_true", default=False, help=("Indicate the second stage of the training."))
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    if len(os.listdir(args.output_dir)) == 9:
        print(f'exit from dit {args.output_dir=}')
        exit()

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    ipg = InitProcessGroupKwargs(timeout=timedelta(minutes=30))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ipg, kwargs],
    )

    rank = accelerator.local_process_index
    world = accelerator.num_processes
    print(f"[main rank {rank}] WORLD_SIZE={world}", flush=True)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.output_path, exist_ok=True)

    load_dtype = torch.bfloat16
    transformer = CustomWanTransformer3DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=load_dtype,
    )

    parameters = {
        "added_kv_proj_dim": None,
        "attention_head_dim": 128,
        "cross_attn_norm": True,
        "eps": 1e-06,
        "ffn_dim": 14336,
        "freq_dim": 256,
        "image_dim": None,
        "num_attention_heads": 24,
        "patch_size": [1, 2, 2],
        "qk_norm": "rms_norm_across_heads",
        "rope_max_seq_len": 1024,
        "text_dim": 4096,
        "in_channels": args.controlnet_input_channels,
        "num_layers": args.controlnet_transformer_num_layers,
        "downscale_coef": args.downscale_coef,
        "out_proj_dim": transformer.config.num_attention_heads * transformer.config.attention_head_dim,
        "vae_channels": args.vae_channels,
    }
    controlnet = WanControlnet(**parameters).to(dtype=torch.bfloat16)
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )

    if args.init_from_transformer:
        controlnet_state_dict = {}
        for name, params in transformer.state_dict().items():
            if 'patch_embedding.weight' in name:
                patch_params = torch.cat([params] * 5, dim=1)
                controlnet_state_dict[name] = patch_params
                del patch_params
                continue
            controlnet_state_dict[name] = params

        m, u = controlnet.load_state_dict(controlnet_state_dict, strict=False)
        print(f'[ Weights from transformer were loaded into controlnet ] [M: {len(m)} | U: {len(u)}]')

    if args.resume_from_checkpoint:
        controlnet = WanControlnet.from_pretrained(args.resume_from_checkpoint, torch_dtype=torch.bfloat16, use_safetensors=True)
        print(f'[ Weights from pretrained controlnet were loaded into controlnet ]')

    transformer.requires_grad_(False)
    controlnet.requires_grad_(False) # only for inference

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.state.deepspeed_plugin:
        # DeepSpeed is handling precision, use what's in the DeepSpeed config
        if (
            "fp16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["fp16"]["enabled"]
        ):
            weight_dtype = torch.float16
        if (
            "bf16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["bf16"]["enabled"]
        ):
            weight_dtype = torch.bfloat16
    else:
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )
    
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    transformer.to(accelerator.device, dtype=weight_dtype)
    controlnet.to(accelerator.device, dtype=weight_dtype)

    if accelerator.is_main_process:
        tracker_name = args.tracker_name or "wan2.2-controlnet"
        if args.wandb_key:
            wandb.login(key=args.wandb_key)
        accelerator.init_trackers(tracker_name, config=vars(args))

    validate_videos(args, accelerator, transformer, controlnet, noise_scheduler, weight_dtype, validate=True)
