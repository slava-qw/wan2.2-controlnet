import sys
sys.path.append('..')
import argparse
import logging
import math
import os
from pathlib import Path

import torch
import transformers
import numpy as np
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs, ProjectConfiguration, set_seed
from datetime import timedelta
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import diffusers
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
)
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module

from dataset import ControlnetDataset
from scheduler import FlowMatchScheduler

from wan_transformer import CustomWanTransformer3DModel
from wan_controlnet import WanControlnet

import wandb
from validation import validate_videos
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from diffusers.utils.torch_utils import randn_tensor

from default_weighing_scheme import default_weighing_scheme
import glob
import torch.distributed as dist
import shutil
import gc


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0.dev0")

logger = get_logger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script for Wan2.2.")

    # Model information
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--latents_dir",
        type=str,
        default=None,
        required=True,
        help=("A folder containing the training data."),
    )
    parser.add_argument(
        "--text_embeds_dir",
        type=str,
        default=None,
        required=True,
        help=("A folder containing the training data."),
    )
    parser.add_argument(
        "--controlnet_video_dir",
        type=str,
        default=None,
        required=True,
        help=("A folder containing the training data."),
    )
    parser.add_argument(
        "--control_video_dir",
        type=str,
        default=None,
        required=True,
        help=("A folder containing the training data."),
    )
    parser.add_argument(
        "--stage_2_training",
        action="store_true",
        default=False,
        help=("Indicate the second stage of the training."),
    )
    parser.add_argument(
        "--sample_type",
        default='non_uniform',
        type=str,
        help=("Define the sampling strategy when `stage_2_training` is True."),
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
        "--controlnet_input_channels",
        type=int,
        default=3,
        required=False,
        help=("Controlnet encoder input channels."),
    )
    parser.add_argument("--dtype", type=str, default="fp32", help="fp32 or fp16")
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
    parser.add_argument(
        "--save_checkpoint_postfix",
        type=str,
        default="",
        required=False,
        help=("Postfix for model .pt checkpoint."),
    )
    parser.add_argument(
        "--init_from_transformer",
        action="store_true",
        help="Whether or not load start controlnet parameters from transformer model.",
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["weighted", "sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    # Training information
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
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
        "--output_dir",
        type=str,
        default="wan2.2-controlnet",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides `--num_train_epochs`.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--keep_n_checkpoints",
        type=int,
        default=3,
        help=(
            "The number of checkpoint that should be keeped during training."
            "If exceeds the older one will be deleted from the `--output_dir`."
        ),
    )

    parser.add_argument(
        "--validation_steps",
        type=int,
        default=1000,
        help=(
            "After how many steps run validation during training to get visual results"
        ),
    )

    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    # Optimizer
    parser.add_argument(
        "--optimizer",
        type=lambda s: s.lower(),
        default="adam",
        choices=["adam", "adamw", "prodigy"],
        help=("The optimizer type to use."),
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.95, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="Coefficients for computing the Prodigy optimizer's stepsize using running averages. If set to None, uses the value of square root of beta2.",
    )
    parser.add_argument("--prodigy_decouple", action="store_true", help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--prodigy_use_bias_correction", action="store_true", help="Turn on Adam's bias correction.")
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        action="store_true",
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage.",
    )

    # Other information
    parser.add_argument("--tracker_name", type=str, default=None, help="Project tracker name")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="Directory where logs are stored.",
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
        "--report_to",
        type=str,
        default=None,
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
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
    parser.add_argument("--wandb_run_id", type=str, default=None,
                        help="Existing W&B run ID to resume logging into.")
    parser.add_argument("--wandb_project_name", type=str, default=None,
                        help="(Optional) Explicit project name when resuming/starting.")


    # Validation information
    parser.add_argument("--validation_prompt", type=str, required=True, help="Latents of the description of the video to be generated")
    parser.add_argument("--validation_prompt_log", type=str, required=True, help="The description of the video to be generated")
    
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="The path of the video for controlnet processing.",
    )
    parser.add_argument("--validation_images", type=str, required=True, help="The path of the image for I2V generation in validation phase.")
    

    parser.add_argument("--controlnet_guidance_start", type=float, default=0.0, help="The stage when the controlnet starts to be applied")
    parser.add_argument("--controlnet_guidance_end", type=float, default=0.8, help="The stage when the controlnet end to be applied")
    
    parser.add_argument(
        "--output_path", type=str, default="./output.mp4", help="The path where the generated video will be saved"
    )
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance")
    parser.add_argument(
        "--num_inference_steps", type=int, default=50, help="Number of steps for the inference process"
    )
    parser.add_argument("--video_height", type=int, default=480, help="Output video height")
    parser.add_argument("--video_width", type=int, default=832, help="Output video width")
    parser.add_argument("--num_frames", type=int, default=49, help="Output frames count")
    parser.add_argument("--negative_prompt", type=str, default="bad quality, worst quality", help="Negative prompt")
    parser.add_argument("--negative_prompt_latent", type=str, default="negative_prompt_latent.pt", help="Path to negative prompt latent")
    parser.add_argument("--out_fps", type=int, default=16, help="FPS of output video")
    parser.add_argument(
        "--num_validation_videos",
        type=int,
        default=1,
        help="Number of videos that should be generated during validation per `validation_prompt`.",
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

    return parser.parse_args()


def reset_memory(device: Union[str, torch.device]) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.reset_accumulated_memory_stats(device)


def get_optimizer(args, params_to_optimize, use_deepspeed: bool = False):
    # Use DeepSpeed optimzer
    if use_deepspeed:
        from accelerate.utils import DummyOptim

        return DummyOptim(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )

    # Optimizer creation
    supported_optimizers = ["adam", "adamw", "prodigy"]
    if args.optimizer not in supported_optimizers:
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}. Supported optimizers include {supported_optimizers}. Defaulting to AdamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and args.optimizer.lower() not in ["adam", "adamw"]:
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'Adam' or 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

    if args.optimizer.lower() == "adamw":
        optimizer_class = bnb.optim.AdamW8bit if args.use_8bit_adam else torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )
    elif args.optimizer.lower() == "adam":
        optimizer_class = bnb.optim.Adam8bit if args.use_8bit_adam else torch.optim.Adam

        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )
    elif args.optimizer.lower() == "prodigy":
        try:
            from prodigyopt import Prodigy
        except Exception as e:
            raise ImportError(
                "To use the Prodigy optimizer, please install prodigyopt: `pip install prodigyopt`."
            ) from e

        optimizer = Prodigy(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    return optimizer


def main(args):
    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

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

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
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

    # vae_dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    # vae = AutoencoderKLWan.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", torch_dtype=vae_dtype).to(device=accelerator.device)
    
    from types import SimpleNamespace

    vae_config = SimpleNamespace(
        base_dim=160,
        decoder_base_dim=256,
        z_dim=48,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0,
        latents_mean=[-0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557, -0.1382, 0.0542, 0.2813, 0.0891, 0.157, -0.0098, 0.0375, -0.1825, -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502, -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.123, -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.052, 0.3748, 0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667],
        latents_std=[0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.499, 0.4818, 0.5013, 0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978, 0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659, 0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093, 0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887, 0.3971, 1.06, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744],
        is_residual=True,
        in_channels=12,
        out_channels=12,
        patch_size=2,
        scale_factor_temporal=4,
        scale_factor_spatial=16,
        _class_name='AutoencoderKLWan',
        _diffusers_version='0.35.0.dev0',
        clip_output=False,
        _name_or_path='Wan-AI/Wan2.2-TI2V-5B-Diffusers'
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
    controlnet.requires_grad_(True)

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

    transformer.to(accelerator.device, dtype=weight_dtype)
    controlnet.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        controlnet.enable_gradient_checkpointing()

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        # only upcast trainable parameters into fp32
        cast_training_params([controlnet], dtype=torch.float32)

    trainable_parameters = list(filter(lambda p: p.requires_grad, controlnet.parameters()))

    # Optimization parameters
    trainable_parameters_with_lr = {"params": trainable_parameters, "lr": args.learning_rate}
    params_to_optimize = [trainable_parameters_with_lr]

    use_deepspeed_optimizer = (
        accelerator.state.deepspeed_plugin is not None
        and "optimizer" in accelerator.state.deepspeed_plugin.deepspeed_config
    )
    use_deepspeed_scheduler = (
        accelerator.state.deepspeed_plugin is not None
        and "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    )

    optimizer = get_optimizer(args, params_to_optimize, use_deepspeed=use_deepspeed_optimizer)

    # Dataset and DataLoader
    train_dataset = ControlnetDataset(
        latents_dir=args.latents_dir,
        text_embeds_dir=args.text_embeds_dir,
        controlnet_video_dir=args.controlnet_video_dir,
        control_video_dir=args.control_video_dir,
        downscale_coef=args.downscale_coef,
        stage_2_training=args.stage_2_training,
        sample_type=args.sample_type,
        seed=args.seed
    )
        
    def collate_fn(examples):
        latents = [example["latents"] for example in examples]
        text_embeds = [example["text_embeds"] for example in examples]
        controlnet_video = [example["controlnet_video"] for example in examples]
        control_image = [example["latent_condition"] for example in examples]

        latents = torch.stack(latents)
        latents = latents.to(memory_format=torch.contiguous_format).float()

        text_embeds = torch.stack(text_embeds)
        text_embeds = text_embeds.to(memory_format=torch.contiguous_format).float()

        controlnet_video = torch.cat(controlnet_video, dim=0)  # cuz each already has shape [1, 3, 81/49, 480, 832]
        controlnet_video = controlnet_video.to(memory_format=torch.contiguous_format).float()

        control_image = torch.cat(control_image, dim=0)  # cuz each already has shape [1, 48, 1, 30, 52]
        control_image = control_image.to(memory_format=torch.contiguous_format).float()

        return {
            "latents": latents,
            "text_embeds": text_embeds,
            "controlnet_video": controlnet_video,
            "latent_condition": control_image,
        }

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes


    if use_deepspeed_scheduler:
        from accelerate.utils import DummyScheduler

        lr_scheduler = DummyScheduler(
            name=args.lr_scheduler,
            optimizer=optimizer,
            total_num_steps=num_training_steps_for_scheduler,
            num_warmup_steps=num_warmup_steps_for_scheduler,
        )
    else:
        # do not account for grad. accum. steps; accelerator.prepare will make it for us:
        # https://github.com/huggingface/accelerate/issues/628
        # also check this for reference: https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps_for_scheduler,
            num_training_steps=num_training_steps_for_scheduler,
            num_cycles=args.lr_num_cycles,
            power=args.lr_power,
        )

    # Prepare everything with our `accelerator`.
    controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = args.tracker_name or "wan2.2-controlnet"
        if args.report_to == "wandb":
            wandb.login(key=args.wandb_key)

            if args.wandb_run_id is not None:
                init_kwargs = {"wandb": {"id": args.wandb_run_id,
                                        "resume": "must",
                                        "name": args.wandb_project_name,
                                        "entity": "slava_"}
                            }
            else: init_kwargs = {}
        accelerator.init_trackers(tracker_name, config=vars(args), init_kwargs=init_kwargs)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_trainable_parameters = sum(param.numel() for model in params_to_optimize for param in model["params"])

    global_step = 0
    first_epoch = 0
    logger.info("***** Running training *****")
    logger.info(f"  Num trainable parameters = {num_trainable_parameters}")
    logger.info(f"  Num update steps per epoch = {num_update_steps_per_epoch}")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    if args.wandb_run_id is None:
        initial_global_step = 0
    elif args.resume_from_checkpoint:
        path = args.resume_from_checkpoint
        global_step = int(os.path.basename(path).split("-")[-1].split("_")[0])
        initial_global_step = global_step
        first_epoch = global_step // num_update_steps_per_epoch
        
        ckpt_state_path = os.path.join(os.path.dirname(path), os.path.basename(path).replace('checkpoint', 'checkpoint_state'))
        accelerator.load_state(ckpt_state_path)
    
        # If for some reason the scheduler state wasn't saved, at least align the step:
        if hasattr(lr_scheduler, "last_epoch") and lr_scheduler.last_epoch < global_step:
            lr_scheduler.last_epoch = global_step
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def compute_density_for_timestep_sampling(
        weighting_scheme: str,
        batch_size: int,
        logit_mean: float = None,
        logit_std: float = None,
        mode_scale: float = None,
        device = "cpu",
        generator = None,
    ):
        if weighting_scheme == "logit_normal":
            u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device=device, generator=generator)
            u = torch.nn.functional.sigmoid(u)
        elif weighting_scheme == "mode":
            u = torch.rand(size=(batch_size,), device=device, generator=generator)
            u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
        else:
            u = torch.rand(size=(batch_size,), device=device, generator=generator)
        return u

    def compute_loss_weighting(noise_scheduler, weighting_scheme: str, sigmas=None, timesteps: torch.Tensor = None):
        if weighting_scheme == "sigma_sqrt":
            weighting = (sigmas**-2.0).float()
        elif weighting_scheme == "cosmap":
            bot = 1 - 2 * sigmas + 2 * sigmas**2
            weighting = 2 / (math.pi * bot)
        elif weighting_scheme == "weighted":
            assert timesteps is not None, 'need timesteps to calculate weightnings'
            step_indices = [(noise_scheduler.timesteps == t).nonzero().item() for t in timesteps.to('cpu')]

            weighting = torch.tensor(
                [default_weighing_scheme[i] for i in step_indices],
                device=timesteps.device,
                dtype=timesteps.dtype
            )
        else:
            weighting = torch.ones_like(sigmas)
        return weighting

    def get_sigmas(noise_scheduler, timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma


    def prepare_latents(
        latent_condition,
        batch_size: int,
        num_channels_latents: int = 48,
        height: int = 480,
        width: int = 832,
        num_frames: int = 49,  # use this one to coincide with CogVideoX
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        last_image: Optional[torch.Tensor] = None,
        
        expand_timesteps: bool = True,

    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_latent_frames = (num_frames - 1) // vae_config.scale_factor_temporal + 1
        latent_height = height // vae_config.scale_factor_spatial
        latent_width = width // vae_config.scale_factor_spatial

        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        latents_mean = (
            torch.tensor(vae_config.latents_mean)
            .view(1, vae_config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(vae_config.latents_std).view(1, vae_config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        # laatents were saved just after vae.encode
        latent_condition = latent_condition.to(dtype)
        latent_condition = (latent_condition - latents_mean) * latents_std

        latents = (latents - latents_mean) * latents_std

        if expand_timesteps:  # for Wan2.2
            mask_shape = (1, 1, num_latent_frames, latent_height, latent_width)
             
            first_frame_mask = torch.ones(
                mask_shape, dtype=dtype, device=device
            )
            # shape: [1, 1, 21, 30, 52] => not ..., :latent_condition.shape[2]] = cond (which has shape: [5, 48, 1, 30, 52])
            # because either way conditioning only on the first frame
            first_frame_mask[:, :, 0] = 0
            return latents, latent_condition, first_frame_mask
    
    expand_timesteps = True
    num_channels_latents = vae_config.z_dim
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
    transformer_dtype = transformer.dtype
        

    for epoch in range(first_epoch, args.num_train_epochs):
        controlnet.train()

        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [controlnet]

            with accelerator.accumulate(models_to_accumulate):
                model_input = batch["latents"].to(dtype=weight_dtype) # [B, C, F, H, W]  torch.Size([16, 21, 60, 104])
                prompt_embeds = batch["text_embeds"].to(dtype=weight_dtype) # [B, SEQ, EMB]
                controlnet_video = batch["controlnet_video"].to(dtype=weight_dtype) # [B, C, F, H, W]  torch.Size([B, 3, 81, 480, 832])
                latent_condition = batch["latent_condition"].to(dtype=weight_dtype)
                last_image = None  #TODO: add this feature later

                batch_size = prompt_embeds.shape[0]
                
                latents_outputs = prepare_latents(
                    latent_condition,
                    batch_size * 1,
                    num_channels_latents,
                    # height,
                    # width,
                    num_frames=args.num_frames,
                    dtype=torch.float32,
                    device=accelerator.device,
                    generator=generator,
                    latents=model_input,
                    last_image=last_image,
                    expand_timesteps=expand_timesteps,
                )
                if expand_timesteps:
                    # wan 2.2 5b i2v use firt_frame_mask to mask timesteps
                    model_input, condition, first_frame_mask = latents_outputs
                else:
                    model_input, condition = latents_outputs

                ### Sample using scheduler------------------------------------
                bsz = model_input.shape[0]
                noise = torch.randn_like(model_input, device=accelerator.device, dtype=weight_dtype)
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=0.0,
                    logit_std=1.0,
                    mode_scale=1.29,
                )
                indices = (u * noise_scheduler.config.num_train_timesteps).long()
                timesteps_orig = noise_scheduler.timesteps[indices].to(device=model_input.device)

                # Add noise according to flow matching.
                sigmas = get_sigmas(noise_scheduler, timesteps_orig, n_dim=model_input.ndim, dtype=model_input.dtype)

                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
                target = noise - model_input

                noisy_model_input = noisy_model_input.to(dtype=weight_dtype)
                
                if expand_timesteps:
                    latent_model_input = (1 - first_frame_mask) * condition + first_frame_mask * noisy_model_input
                    latent_model_input = latent_model_input.to(transformer_dtype)

                    #TODO: write this better
                    # from: https://github.com/ostris/ai-toolkit/blob/c6edd71a5bb36f3dffcc8b56ee07cacaee14ab56/extensions_built_in/diffusion_models/wan22/wan22_5b_model.py#L273
                    t_chunks = torch.chunk(timesteps_orig, timesteps_orig.shape[0])
                    out_t_chunks = []
                    for t in t_chunks:
                        # seq_len: num_latent_frames * latent_height//2 * latent_width//2
                        temp_ts = (first_frame_mask[0][0][:, ::2, ::2] * t).flatten()
                        # batch_size, seq_len
                        temp_ts = temp_ts.unsqueeze(0)
                        out_t_chunks.append(temp_ts)
                    timesteps = torch.cat(out_t_chunks, dim=0)

                else:
                    latent_model_input = torch.cat([noisy_model_input, condition], dim=1).to(transformer_dtype)
                    timesteps = timesteps_orig.expand(model_input.shape[0])  # no need?? already have that shape
                    

                target = target.to(dtype=weight_dtype)

                controlnet_states = controlnet(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    controlnet_states=controlnet_video,
                    timestep=timesteps,
                    return_dict=False,
                )[0]

                if isinstance(controlnet_states, (tuple, list)):
                    controlnet_states = [x.to(dtype=weight_dtype) for x in controlnet_states]
                else:
                    controlnet_states = controlnet_states.to(dtype=weight_dtype)

                model_output = transformer(
                    hidden_states=latent_model_input,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    controlnet_states=controlnet_states,
                    controlnet_weight=args.controlnet_weights,
                    controlnet_stride=args.controlnet_stride,
                    return_dict=False,
                )[0]

                weighting = compute_loss_weighting(noise_scheduler, weighting_scheme=args.weighting_scheme, sigmas=sigmas, timesteps=timesteps_orig)
                
                # Ensure weighting has the same shape as model_output and target for proper broadcasting
                if weighting.shape != model_output.shape:
                    # Reshape weighting to match the output tensor shape
                    weighting = weighting.view(model_output.shape[0], *([1] * (len(model_output.shape) - 1)))
                
                loss_mask = first_frame_mask.to(model_output.dtype)
                loss_mask_c = loss_mask.expand(model_output.shape[0], model_output.shape[1], *loss_mask.shape[2:])  # (B,C,F,H,W)

                mse = (model_output.float() - target.float()) ** 2
                weighted_mse = mse * weighting.float() * loss_mask_c  # first frame contributes 0

                # per-sample normalization by the number of unmasked elements
                denom = loss_mask_c.reshape(model_output.shape[0], -1).sum(dim=1).clamp_min(1.0)
                loss = weighted_mse.reshape(model_output.shape[0], -1).sum(dim=1) / denom
                
                loss = loss.mean() 
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    params_to_clip = controlnet.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                if accelerator.state.deepspeed_plugin is None:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}{args.save_checkpoint_postfix}")
                        unwrap_model(controlnet).save_pretrained(save_path)
                        
                        save_path_state = os.path.join(args.output_dir, f"checkpoint_state-{global_step}{args.save_checkpoint_postfix}")
                        accelerator.save_state(save_path_state)

                        # Keep only the n most recent checkpoints
                        # Find all checkpoint directories matching pattern
                        checkpoints = sorted(
                            glob.glob(os.path.join(args.output_dir, "checkpoint-*")),
                            key=lambda p: int(p.split("-")[-1].split("_")[0]),
                            reverse=True
                        )
                        checkpoints = [ckpt for ckpt in checkpoints if args.save_checkpoint_postfix in ckpt]

                        # Remove older ones beyond the n most recent
                        if len(checkpoints) > args.keep_n_checkpoints:
                            old_checkpoints = checkpoints[args.keep_n_checkpoints:]
                            for ckpt in old_checkpoints:
                                try:
                                    shutil.rmtree(ckpt)
                                    pre, post = ckpt.split('-')
                                    shutil.rmtree(f"{pre}_state-{post}")

                                    logger.info(f"Deleted old checkpoint and its state: {ckpt}")
                                except Exception as e:
                                    logger.info(f"Could not delete checkpoint {ckpt}: {e}")
                    
                    should_run_validation = args.validation_prompt is not None and (
                        args.validation_steps is not None and global_step % args.validation_steps == 0
                    )
                    if should_run_validation:
                        validate_videos(args, accelerator, transformer, controlnet, noise_scheduler, weight_dtype)
                    
                # Synchronize after validation to ensure all processes wait
                accelerator.wait_for_everyone()
            
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            # to_drop = [
            #     "loss","model_output","target","weighting",
            #     "latent_model_input","noisy_model_input",
            #     "sigmas","timesteps_orig","timesteps","noise","u","indices",
            #     "condition","first_frame_mask","model_input","prompt_embeds","controlnet_video","latent_condition"
            # ]
            # for name in to_drop:
            #     if name in locals():
            #         del locals()[name]
            # reset_memory(accelerator.device)
            del model_input, prompt_embeds, controlnet_states, model_output
            del target, noisy_model_input, sigmas, condition, controlnet_video, latent_condition
            gc.collect()
            torch.cuda.empty_cache()
                    
            if global_step >= args.max_train_steps:
                break
    
    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}{args.save_checkpoint_postfix}")
        unwrap_model(controlnet).save_pretrained(save_path)
        
        save_path_state = os.path.join(args.output_dir, f"checkpoint_state-{global_step}{args.save_checkpoint_postfix}")
        accelerator.save_state(save_path_state)

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    args = get_args()
    main(args)
    