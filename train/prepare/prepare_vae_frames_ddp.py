import os
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from decord import VideoReader
from diffusers import AutoencoderKLWan
from diffusers.video_processor import VideoProcessor
import pandas as pd


def log(rank, *msg):
    print(f"[rank {rank}]", *msg, flush=True)


def resolve_rank_and_device(args):
    # TorchRun/Slurm envs
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if args.device == "cuda":
        if not torch.cuda.is_available():
            log(local_rank, "CUDA requested but not available; falling back to CPU.")
            device = torch.device("cpu")
        else:
            # Bind this rank to its device index (ROCm: same idiom as CUDA)
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    return local_rank, world_size, device


def process_partition(rank: int, world_size: int, device, args):
    vae_dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    df = pd.read_csv(args.csv_path)
    if "video" not in df.columns or "ids" not in df.columns:
        raise RuntimeError("CSV must have 'video' and 'ids' columns")

    input_video_paths = [os.path.join(args.input_video_dir, vp) for vp in df["video"].tolist()]
    total_items = len(input_video_paths)
    os.makedirs(args.out_latents_dir, exist_ok=True)

    # Strided sharding
    local_indices = list(range(rank, total_items, world_size))

    # Diagnostics
    log(rank, f"torch.cuda.is_available={torch.cuda.is_available()} "
              f"device_count={torch.cuda.device_count()} "
              f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES')} "
              f"ROCR_VISIBLE_DEVICES={os.environ.get('ROCR_VISIBLE_DEVICES')}")
    log(rank, f"Device={device}, dtype={vae_dtype}. Assigned {len(local_indices)}/{total_items}. "
              f"Sample idx: {local_indices[:6]}")

    log(rank, "Loading VAE...")
    vae = AutoencoderKLWan.from_pretrained(
        args.base_model_path, subfolder="vae", torch_dtype=vae_dtype
    ).to(device=device)
    video_processor = VideoProcessor(vae_scale_factor=vae.config.scale_factor_spatial)

    # Per-rank RNG
    generator = torch.Generator(device=device).manual_seed(args.seed + rank)

    # Per-rank progress bar
    for i in tqdm(local_indices, desc=f"worker-{rank}", total=len(local_indices), position=rank, leave=False):
        input_video_path = input_video_paths[i]
        vid_id = df.iloc[i]["ids"]

        basename = os.path.basename(input_video_path)
        # save_basename = f"{basename.split('_')[0]}_{vid_id}.pt"
        save_basename = f"{basename.split('.png')[0]}_{vid_id}.pt"
        out_latents_path = os.path.join(args.out_latents_dir, save_basename)

        if (not args.overwrite) and os.path.exists(out_latents_path):
            continue

        # Only first frame for this file variant
        if input_video_path.endswith('.mp4'):
            video_reader = VideoReader(input_video_path)
            np_video = video_reader.get_batch([0]).asnumpy()
            del video_reader
            img_frames = [Image.fromarray(x) for x in np_video]
        else:
            img_frames = [Image.open(input_video_path).convert("RGB")]

        preprocessed_video = (
            video_processor.preprocess(
                img_frames,
                height=args.height,
                width=args.width,
            )
            .permute(1, 0, 2, 3)  # (C, T, H, W)
            .unsqueeze(0)          # (B=1, C, T, H, W)
            .to(device, dtype=vae_dtype)
        )

        with torch.no_grad():
            latents = vae.encode(preprocessed_video).latent_dist.sample(generator)

        torch.save(latents.cpu(), out_latents_path)

        if device.type == "cuda":
            try:
                del preprocessed_video
            except Exception:
                pass
            if "latents" in locals():
                del latents
            torch.cuda.empty_cache()


def main(args):
    rank, world_size, device = resolve_rank_and_device(args)
    log(rank, f"LOCAL_RANK={rank} WORLD_SIZE={world_size} "
              f"args.device={args.device} args.dtype={args.dtype}")

    os.makedirs(args.out_latents_dir, exist_ok=True)
    process_partition(rank, world_size, device, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate first-frame VAE latents with torchrun (ROCm/Slurm friendly).")
    parser.add_argument("--base_model_path", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                        help="Pretrained model repo/path; must contain VAE in subfolder 'vae'.")
    parser.add_argument("--input_video_dir", type=str, required=True, help="Directory with videos")
    parser.add_argument("--csv_path", type=str, required=True,
                        help="CSV with at least 'video' and 'ids' columns")
    parser.add_argument("--out_latents_dir", type=str, required=True, help="Directory for .pt latents")
    parser.add_argument("--width", type=int, default=832, help="Preprocessor width")
    parser.add_argument("--height", type=int, default=480, help="Preprocessor height")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (per-rank offset added)")
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16"], help="Compute dtype")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Compute device")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute and overwrite existing .pt files instead of skipping")
    args = parser.parse_args()
    main(args)
