"""
Running the Script:
To run the script, use the following command with appropriate arguments:

```bash
python -m inference.cli_demo \
    --args.video_path "./resources/bubble.mp4" \
    --prompt "Close-up shot with soft lighting, focusing sharply on the lower half of a young woman's face. Her lips are slightly parted as she blows an enormous bubblegum bubble. The bubble is semi-transparent, shimmering gently under the light, and surprisingly contains a miniature aquarium inside, where two orange-and-white goldfish slowly swim, their fins delicately fluttering as if in an aquatic universe. The background is a pure light blue color." \
    --controlnet_type "depth" \
    --args.pretrained_model_name_or_path Wan-AI/Wan2.2-TI2V-5B-Diffusers \
    --controlnet_model_path TheDenk/wan2.2-ti2v-5b-controlnet-depth-v1
```

Additional options are available to specify the guidance scale, number of inference steps, video generation type, and output paths.
"""


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
sys.path.append('..')
import argparse

import cv2
import torch
import numpy as np
from PIL import Image

from diffusers import (
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
    UniPCMultistepScheduler
)
from diffusers.utils import export_to_video, load_video
from controlnet_aux import HEDdetector, CannyDetector, MidasDetector


import gc
import wandb
from typing import Any, Dict, Union, List
from PIL import Image
from accelerate import Accelerator
from diffusers.utils.torch_utils import is_compiled_module

from pathlib import Path
import transformers
import diffusers

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs, ProjectConfiguration, set_seed
from datetime import timedelta

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

def log_validation(
    accelerator,
    pipe,
    args: Dict[str, Any],
    all_pipeline_args: Dict[str, Any],
    is_final_validation: bool = False,
    prompts_to_log: list = [],
    validate=False
):

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
    phase_name = "test" if is_final_validation else "validation"

    video_filenames = []
    for i, pipeline_kwargs in enumerate(all_pipeline_args):
        for j in range(args.num_validation_videos):
            video = pipe(**pipeline_kwargs, generator=generator, output_type="np").frames[0]

            prompt_to_log = prompts_to_log[i]
            prompt = (
                prompt_to_log[:25]
                .replace(" ", "_").replace("'", "_").replace('"', "_")
                .replace("/", "_").replace(":", "_").replace("*", "_")
                .replace("?", "_").replace("<", "_").replace(">", "_").replace("|", "_")
                )
            cfg = pipeline_kwargs['guidance_scale']
            cntw = pipeline_kwargs['controlnet_weight']
            filename = os.path.join(args.output_path, f"{phase_name}_video_{i}_{j}_{cfg=}_{cntw=}_{prompt}.mp4")
            export_to_video(video, filename, fps=8)

            del video

            video_filenames.append((filename, j, prompt_to_log))

    if not validate:
        for tracker in accelerator.trackers:
            if tracker.name == "wandb":
                tracker.log({
                    phase_name: [wandb.Video(fn, caption=f"{i}_{j}: {p}") for i, (fn, j, p) in enumerate(video_filenames)]
                })

    del video_filenames
    

def load_latent(path: str, device) -> torch.Tensor:
    return torch.load(path, map_location=device)

def check_pipeline_devices(pipe):
    for name, component in pipe.components.items():
        if hasattr(component, 'device'):
            print(f"{name}: {component.device}")
        else:
            print(f"{name}: No device attribute (likely CPU-based)")


@torch.no_grad()
def validate_videos(
        args,
        accelerator,
        transformer,
        controlnet,
        scheduler,
        weight_dtype,
        validate=False,
    ):
    """
    Generates a video based on the given prompt and saves it to the specified path.
    """
    accelerator.print("===== Memory before validation =====")
    print_memory(accelerator.device)
    torch.cuda.synchronize(accelerator.device)
    torch.cuda.empty_cache()

    # 1.  Load the pre-trained Wan2.2 models with the specified precision (bfloat16).
    # tokenizer = T5TokenizerFast.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    # text_encoder = UMT5EncoderModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    vae = AutoencoderKLWan.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", torch_dtype=torch.float32)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    vae = vae.to(accelerator.device).eval()

    pipe = WanImageToVideoControlnetPipeline(
        # tokenizer=tokenizer, 
        # text_encoder=text_encoder,
        transformer=unwrap_model(accelerator, transformer).eval(),
        vae=vae, 
        controlnet=unwrap_model(accelerator, controlnet).eval(),
        scheduler=scheduler,
        expand_timesteps=True,  # for Wan2.2 
    )
    # check_pipeline_devices(pipe)

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=5.0)
    
    # if args.enable_slicing:
    #     pipe.vae.enable_slicing()
    # if args.enable_tiling:
    #     pipe.vae.enable_tiling()
    # if args.enable_model_cpu_offload:
        # pipe.enable_model_cpu_offload()
        
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


        progress_bar = tqdm(
            range(0, len(validation_images)),
            desc="Num val vids",
            # Only show the progress bar once on each machine.
            disable=not accelerator.is_local_main_process,
        )
    
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
    all_pipeline_args = []
    prompts_to_log = []

    for i, (validation_video, validation_prompt, validation_image_path, validation_prompt_log) in enumerate(zip(validation_videos, validation_prompts, validation_images, validation_prompts_log)):
        # if i <= len(os.listdir(args.output_path)) - 1:
        #     progress_bar.update(1)
        #     continue

        controlnet_frames: List[Image.Image] = load_video(validation_video)
        
        validation_image_latent = load_latent(path=validation_image_path, device=accelerator.device)  # normalized by VAEs stats inside __call__ func (prepare_latents) 
        validation_prompt_latent = load_latent(path=validation_prompt, device=accelerator.device)

        #important: `image` and `image_embeds` are not the same, former used in condition w/ noise and the latter as `encoder_hidden_states_image` args
        gen_kwargs = {
                "image_latents": validation_image_latent,
                "prompt_embeds": validation_prompt_latent,
                "controlnet_frames": controlnet_frames,
                "stage_2": args.stage_2_training,
        }
        current_gen_kwargs = {**base_pipeline_args, **gen_kwargs}

        if validate:
            log_validation(
                pipe=pipe,
                args=args,
                accelerator=accelerator,
                all_pipeline_args=[current_gen_kwargs],
                prompts_to_log=[validation_prompt_log],  # cuz we're using only precalculated latents of prompts 
                validate=validate
            )
            progress_bar.update(1)

        else:
            all_pipeline_args.append(current_gen_kwargs)
            prompts_to_log.append(validation_prompt_log)

        del controlnet_frames, validation_image_latent, validation_prompt_latent
        del gen_kwargs

    if not validate:
        log_validation(
            pipe=pipe,
            args=args,
            accelerator=accelerator,
            all_pipeline_args=all_pipeline_args,
            prompts_to_log=prompts_to_log  # cuz we're using only precalculated latents of prompts 
        )

    transformer.train()
    controlnet.train()

    del all_pipeline_args, prompts_to_log
    del pipe, vae, scheduler

    accelerator.print("===== Memory after validation =====")
    print_memory(accelerator.device)
    reset_memory(accelerator.device)
    torch.cuda.synchronize(accelerator.device)


def get_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script for Wan2.2.")
    parser.add_argument(
        "--num_validation_videos",
        type=int,
        default=1,
        help="Number of videos that should be generated during validation per `validation_prompt`.",
    )    
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--output_path", type=str, default="./output.mp4", help="The path where the generated video will be saved"
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--enable_slicing",
        action="store_true",
        default=False,
        help="Whether or not to use VAE slicing for saving memory.",
    )

    parser.add_argument(
        "--enable_tiling",
        action="store_true",
        default=False,
        help="Whether or not to use VAE tiling for saving memory.",
    )

    parser.add_argument(
        "--enable_model_cpu_offload",
        action="store_true",
        default=False,
        help="Whether or not to enable model-wise CPU offloading when performing validation/testing to save memory.",
    )
    parser.add_argument("--validation_images", type=str, required=True, help="The path of the image for I2V generation in validation phase.")
    parser.add_argument("--validation_prompt", type=str, required=True, help="Latents of the description of the video to be generated")
    parser.add_argument("--validation_prompt_log", type=str, required=True, help="The description of the video to be generated")
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="The path of the video for controlnet processing.",
    )
    parser.add_argument("--video_height", type=int, default=480, help="Output video height")
    parser.add_argument("--video_width", type=int, default=832, help="Output video width")
    parser.add_argument("--num_frames", type=int, default=81, help="Output frames count")
    parser.add_argument("--negative_prompt", type=str, default="bad quality, worst quality", help="Negative prompt")
    parser.add_argument("--negative_prompt_latent", type=str, default="negative_prompt_latent.pt", help="Path to negative prompt latent")

    parser.add_argument("--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance")
    parser.add_argument(
        "--num_inference_steps", type=int, default=50, help="Number of steps for the inference process"
    )
    parser.add_argument("--controlnet_guidance_end", type=float, default=0.8, help="The stage when the controlnet end to be applied")
    parser.add_argument(
        "--controlnet_weights",
        type=float,
        default=1.0,
        required=False,
        help=("Controlnet blocks weight."),
    )
    parser.add_argument(
        "--controlnet_stride",
        type=int,
        default=1,
        required=False,
        help=("Controlnet block stride. Controlnet block is applied each N step."),
    )
    parser.add_argument("--teacache_treshold", type=float, default=0.0, help="TeaCache value. Best from [0.3, 0.5, 0.7, 0.9]")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="wan2.2-controlnet",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="Directory where logs are stored.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default=None,
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--controlnet_input_channels",
        type=int,
        default=3,
        required=False,
        help=("Controlnet encoder input channels."),
    )
    parser.add_argument(
        "--controlnet_transformer_num_layers",
        type=int,
        default=2,
        required=False,
        help=("Count of controlnet blocks."),
    )
    parser.add_argument(
        "--downscale_coef",
        type=int,
        default=16,
        required=False,
        help=("Downscale coef as encoder decreases resolutio before apply transformer."),
    )
    parser.add_argument(
        "--vae_channels",
        type=int,
        default=16,
        required=False,
        help=("Vae output channels."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        required=False,
        help=("Path to controlnet .pt checkpoint."),
    )
    parser.add_argument("--tracker_name", type=str, default=None, help="Project tracker name")
    parser.add_argument(
        "--init_from_transformer",
        action="store_true",
        help="Whether or not load start controlnet parameters from transformer model.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--wandb_key",
        type=str,
        default=None,
        help=(
            'Wandb API key.'
        ),
    )
    parser.add_argument("--controlnet_guidance_start", type=float, default=0.0, help="The stage when the controlnet starts to be applied")
    parser.add_argument("--stage_2_training", action="store_true", default=False, help=("Indicate the second stage of the training."))

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)  #TODO: was True but always get: [W1024 14:03:25.792762686 reducer.cpp:1431] Warning: find_unused_parameters=True was specified in DDP constructor, but did not find any unused parameters in the forward pass. This flag results in an extra traversal of the autograd graph every iteration,  which can adversely affect performance. If your model indeed never has any unused parameters in the forward pass, consider turning this flag off. Note that this warning may be a false positive if your model has flow control causing later iterations to have unused parameters. (function operator())
    ipg = InitProcessGroupKwargs(timeout=timedelta(minutes=30))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ipg, kwargs],
    )
    
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    load_dtype = torch.bfloat16 
    transformer = CustomWanTransformer3DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=load_dtype,
    )

    # https://github.com/Wan-Video/Wan2.2/blob/a64d5b25af052a24e8e1bc23aa7af3ee130b1d84/wan/configs/wan_ti2v_5B.py#L15
    parameters = {
        "added_kv_proj_dim": None,
        "attention_head_dim": 128,
        "cross_attn_norm": True,
        "eps": 1e-06,
        "ffn_dim": 14336, # 8960,
        "freq_dim": 256,
        "image_dim": None,
        "num_attention_heads": 24, # 12,
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
                patch_params = torch.cat([params] * 5, dim=1)  # 4 for Wan2.1 w/ downsc_f=8 and factor 5(2) for Wan2.2 w/ downscale_factor=16(8)

                controlnet_state_dict[name] = patch_params
                del patch_params
                continue
            controlnet_state_dict[name] = params

        m, u = controlnet.load_state_dict(controlnet_state_dict, strict=False)
        print(f'[ Weights from transformer were loaded into controlnet ] [M: {len(m)} | U: {len(u)}]')

    if args.resume_from_checkpoint:
        controlnet = WanControlnet.from_pretrained(args.resume_from_checkpoint, torch_dtype=torch.bfloat16, use_safetensors=True)
        print(f'[ Weights from pretrained controlnet were loaded into controlnet ]')

    # We only train the additional adapter controlnet layers
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
        
        wandb.login(key=args.wandb_key)
        accelerator.init_trackers(tracker_name, config=vars(args))


    validate_videos(args, accelerator, transformer, controlnet, noise_scheduler, weight_dtype, validate=True)
