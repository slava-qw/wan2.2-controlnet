import os
import argparse
import torch
import numpy as np
from PIL import Image
from decord import VideoReader
from diffusers import AutoencoderKLWan
from diffusers.video_processor import VideoProcessor
import pandas as pd
from tqdm import tqdm


def ddp_log(rank, *msg):
    print(f"[rank {rank}]", *msg, flush=True)


def device_and_rank(args):
    # TorchRun / SLURM provide env vars that are reliable on ROCm
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if args.device == "cuda":
        if not torch.cuda.is_available():
            ddp_log(local_rank, "CUDA requested but not available; falling back to CPU.")
            device = torch.device("cpu")
        else:
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

    input_video_paths = [os.path.join(args.input_video_dir, vid_path) for vid_path in df["video"].tolist()]
    total_items = len(input_video_paths)
    os.makedirs(args.out_latents_dir, exist_ok=True)

    # Partition (strided)
    local_indices = list(range(rank, total_items, world_size))

    ddp_log(rank, f"torch.cuda.is_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()} "
                  f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES')} "
                  f"ROCR_VISIBLE_DEVICES={os.environ.get('ROCR_VISIBLE_DEVICES')}")
    ddp_log(rank, f"Device={device}, dtype={vae_dtype}. Assigned {len(local_indices)}/{total_items}. "
                  f"Sample idx: {local_indices[:6]}")

    ddp_log(rank, "Loading VAE...")
    vae = AutoencoderKLWan.from_pretrained(args.base_model_path, subfolder="vae",
                                           torch_dtype=vae_dtype).to(device=device)
    video_processor = VideoProcessor(vae_scale_factor=vae.config.scale_factor_spatial)
    generator = torch.Generator(device=device).manual_seed(args.seed + rank)

    for i in tqdm(local_indices, desc=f"worker-{rank}", total=len(local_indices), position=rank, leave=False):
        input_video_path = input_video_paths[i]
        vid_id = df.iloc[i]["ids"]

        basename = os.path.basename(input_video_path)
        save_basename = f"{basename.split('_')[0]}_{vid_id}.pt"
        out_latents_path = os.path.join(args.out_latents_dir, save_basename)

        if (not args.overwrite) and os.path.exists(out_latents_path):
            continue

        try:
            video_reader = VideoReader(input_video_path)
            video_length = len(video_reader)
            clip_length = min(video_length, (args.sample_n_frames - 1) * args.sample_stride + 1)
            batch_index = np.linspace(0, clip_length - 1, args.sample_n_frames, dtype=int)
            np_video = video_reader.get_batch(batch_index).asnumpy()
            del video_reader

            preprocessed_video = (
                video_processor.preprocess(
                    [Image.fromarray(x) for x in np_video],
                    height=args.height,
                    width=args.width,
                )
                .permute(1, 0, 2, 3)
                .unsqueeze(0)
                .to(device, dtype=vae_dtype)
            )

            with torch.no_grad():
                latents = vae.encode(preprocessed_video).latent_dist.sample(generator)

            torch.save(latents.cpu(), out_latents_path)

        except Exception as e:
            ddp_log(rank, f"Error on global index {i} ({input_video_path}): {e}")
        finally:
            if device.type == "cuda":
                try:
                    del preprocessed_video
                except Exception:
                    pass
                if "latents" in locals():
                    del latents
                torch.cuda.empty_cache()


def main(args):
    rank, world_size, device = device_and_rank(args)

    # High-level startup banner per-rank
    ddp_log(rank, f"LOCAL_RANK={rank} WORLD_SIZE={world_size} "
                  f"args.device={args.device} args.dtype={args.dtype}")

    process_partition(rank, world_size, device, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate input latents (torchrun multi-GPU, ROCm/SLURM friendly).")
    parser.add_argument("--base_model_path", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--input_video_dir", type=str, required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--out_latents_dir", type=str, required=True)
    parser.add_argument("--sample_stride", type=int, default=2)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--sample_n_frames", type=int, default=81)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16"])
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    main(args)
