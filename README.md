# Introduction

This repository includes both training and inference scripts for **Wan2.2 5b i2v** pipeline with ControlNet.

## Installation
Clone the repository
```bash
git clone https://github.com/slava-qw/wan2.2-controlnet
cd wan2.2-controlnet
```

Install Miniconda: https://docs.anaconda.com/miniconda/

Create venv  
```bash
conda create -n my_env python=3.10
conda activate my_env
```
  
Install requirements: firstly, torch and torchvision according to your GPU setup:
```bash
pip install torch>=2.4.0 torchvision>=0.19.0
pip install -r requirements.txt
```

## Training

To speed up the training, one needs to precompute the VAE latents of the whole video and the first conditional frames, resize the conditional control net videos, and calculate the text embeddings. The entire preparation code is located in the `./train/prepare` folder (it includes two versions for single- and multi-GPU setups).

### Precalculations
Scripts to run each file:

For VAE latents of training videos:
```bash
#!/bin/bash
export NPROC=${SLURM_GPUS_PER_TASK:-${SLURM_GPUS_ON_NODE:-4}}
export DATA_BASE="<path-to-save-latents>"

# Encode video into vae latents:
echo 'start with latents'
mkdir -p "$DATA_BASE/latents_48ch_81fr"
torchrun --standalone --nproc_per_node="$NPROC" \
./train/prepare/prepare_vae_latents_ddp.py \
--input_video_dir "$DATA_BASE/videos_eval" \
--csv_path "$DATA_BASE/metadata.csv" \
--out_latents_dir "$DATA_BASE/latents_48ch_81fr" \
--base_model_path "Wan-AI/Wan2.2-TI2V-5B-Diffusers" \
--sample_stride 2 \
--width 832 \
--height 480 \
--sample_n_frames 81 \
--seed 42 \
--device "cuda" \
--dtype "fp32" \

echo 'done with latents'
```

For VAE latents of the initial frames of training videos:
```bash
#!/bin/bash
export NPROC=${SLURM_GPUS_PER_TASK:-${SLURM_GPUS_ON_NODE:-4}}
export DATA_BASE="<path-to-save-latents>"

# Encode video into vae latents 1st frames:
echo 'start with latents frames'
mkdir -p "$DATA_BASE/latents_48ch_1st_frames"
torchrun --standalone --nproc_per_node="$NPROC" \
./train/prepare/prepare_vae_frames_ddp.py \
--input_video_dir "$DATA_BASE/init_frames_eval" \
--csv_path "$DATA_BASE/metadata.csv" \
--out_latents_dir "$DATA_BASE/latents_48ch_1st_frames" \
--base_model_path "Wan-AI/Wan2.2-TI2V-5B-Diffusers" \
--width 832 \
--height 480 \
--seed 42 \
--device "cuda" \
--dtype "fp32" \

echo 'done with latents'
```

For ControlNet video resizing:

```bash
#!/bin/bash
export NPROC=${SLURM_GPUS_PER_TASK:-${SLURM_GPUS_ON_NODE:-4}}
export DATA_BASE="<path-to-save-latents>"

# Preprocess original videos with controlnet processor (in this case, just reshape them).
echo 'start with controlnets vids'
mkdir -p "$DATA_BASE/controlnet_latents"
torchrun --standalone --nproc_per_node="$NPROC" \
.train/prepare/prepare_controlnet_video_ddp.py \
--input_video_dir "$DATA_BASE/vel_masks" \
--csv_path "$DATA_BASE/real_metadata.csv" \
--out_controlnet_video_dir "$DATA_BASE/controlnet_latents" \
--sample_stride 2 \
--width 832 \
--height 480 \
--sample_n_frames 81 \

echo 'done with controlnets vids'
```

For text embedding precalculation:

```bash
#!/bin/bash
export DATA_BASE="<path-to-save-latents>"

# dataset preparation:
echo 'start with embeddings'
mkdir -p "$DATA_BASE/text_embeds"
export NEG_PROMPT="bad quality, worst quality"

python ./train/prepare/prepare_text_embeddings.py \
--csv_path "$DATA_BASE/real_metadata.csv" \
--out_embeds_dir "$DATA_BASE/text_embeds" \
--base_model_path "Wan-AI/Wan2.2-TI2V-5B-Diffusers" \
--device "cuda" \
--dtype "bf16" \
--negative_prompt "$NEG_PROMPT"

echo 'done with embeddings'
```
### ControlNet training

After all preparations, the data should be placed in a separate folder. To start training, provide the paths to the saved latents and run the following script:

```bash
#!/bin/bash
export MODEL_PATH="Wan-AI/Wan2.2-TI2V-5B-Diffusers"
export DATA_BASE="<path-to-saved-latents>"

export WANDB_API_KEY="<api-key-for-logs>"
mkdir -p "$DATA_BASE/wan_controlnet_checkpoints"
mkdir -p "$DATA_BASE/eval_vids"

# take some validation prompts from val dataset
export VAL_PROMPT0="..."
export VAL_PROMPT1="..."
export VAL_PROMPT2="..."

accelerate launch --config_file ./train/accelerate_config_machine_single.yaml --multi_gpu \
  --num_machines 1 \
  --num_processes 4 \
  ./train/train_controlnet.py \
  --tracker_name "wan2.2-controlnet" \
  --pretrained_model_name_or_path $MODEL_PATH \
  --weighting_scheme 'weighted' \
  --seed 17 \
  --mixed_precision bf16 \
  --output_dir "$DATA_BASE/wan_controlnet_checkpoints" \
  --keep_n_checkpoints 3 \
  --latents_dir "$DATA_BASE/latents_48ch_81fr" \
  --text_embeds_dir "$DATA_BASE/text_embeds" \
  --controlnet_video_dir "$DATA_BASE/controlnet_latents" \
  --control_video_dir "$DATA_BASE/latents_48ch_1st_frames" \
  --controlnet_transformer_num_layers 8 \
  --controlnet_input_channels 3 \
  --downscale_coef 16 \
  --vae_channels 48 \
  --num_frames 49 \
  --controlnet_weights 0.5 \
  --controlnet_stride 3 \
  --save_checkpoint_postfix "_3stride_8blocks" \
  --init_from_transformer \
  --train_batch_size 5 \
  --stage_2_training \
  --sample_type "non_uniform" \
  --dataloader_num_workers 0 \
  --num_train_epochs 4 \
  --validation_steps 125 \
  --checkpointing_steps 125 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-5 \
  --lr_scheduler cosine_with_restarts \
  --lr_warmup_steps 200 \
  --lr_num_cycles 1 \
  --optimizer AdamW \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --max_grad_norm 1.0 \
  --allow_tf32 \
  --gradient_checkpointing \
  --report_to wandb \
  --wandb_key $WANDB_API_KEY \
  --validation_prompt "<some-embeds-to-log>" \
  --validation_prompt_log "$VAL_PROMPT0;$VAL_PROMPT1;$VAL_PROMPT2" \
  --negative_prompt_latent "<path-to-latent-of-negative-prompt>" \
  --video_path "<controlnet-conditions-to-log>" \
  --validation_image "<starting-condition-frames-for-validation>" \
  --output_path "$DATA_BASE/eval_vids" \
  --resume_from_checkpoint "<path-to-checkpoint>" \
  --wandb_run_id "<provide-to-resume-training>" \
  --wandb_project_name "wan2.2-controlnet"
  
  # --wandb_run_id <provide-to-resume-training> \
```

### Inference

For single GPU:
```bash
export MODEL_PATH="Wan-AI/Wan2.2-TI2V-5B-Diffusers"
export DATA_BASE="<path-to-saved-latents>"

export WANDB_API_KEY="<api-key-for-logs>"

accelerate launch --config_file ./train/accelerate_config_machine_single.yaml --multi_gpu \
  ./train/validation.py \
  --tracker_name "wan2.2-controlnet" \
  --pretrained_model_name_or_path $MODEL_PATH \
  --seed 43 \
  --mixed_precision bf16 \
  --output_dir "$HOME_DATA_BASE/wan_controlnet_checkpoints" \
  --stage_2_training \
  --controlnet_transformer_num_layers 8 \
  --controlnet_input_channels 3 \
  --downscale_coef 16 \
  --vae_channels 48 \
  --num_frames 49 \
  --controlnet_weights 0.5 \
  --guidance_scale 6.0 \
  --controlnet_stride 3 \
  --init_from_transformer \
  --gradient_accumulation_steps 2 \
  --allow_tf32 \
  --validation_prompt "$DATA_BASE/text_embeds"\
  --validation_prompt_log "$DATA_BASE/descriptions.txt" \
  --negative_prompt_latent "$DATA_BASE/negative_prompt_latent.pt" \
  --validation_image "$DATA_BASE/latents_48ch_1st_frames" \
  --video_path "$DATA_BASE/controlnet_latents" \
  --output_path "$DATA_BASE/eval_vids" \
  --resume_from_checkpoint "<path-to-checkpoint>"
```

For multi-gpu:

```bash
export NPROC=${SLURM_GPUS_PER_TASK:-${SLURM_GPUS_ON_NODE:-4}}
export MODEL_PATH="Wan-AI/Wan2.2-TI2V-5B-Diffusers"
export MAIN_DATA_PATH="<path-to-save>"

accelerate launch \
  --config_file ./train/accelerate_config_machine_single.yaml \
  --multi_gpu \
  --num_processes "$NPROC" \
  ./train/validation_ddp.py \
  --tracker_name "wan2.2-controlnet" \
  --pretrained_model_name_or_path "$MODEL_PATH" \
  --seed 17 \
  --mixed_precision bf16 \
  --output_dir "$HOME_DATA_BASE/wan_controlnet_checkpoints" \
  --controlnet_transformer_num_layers 8 \
  --controlnet_input_channels 3 \
  --downscale_coef 16 \
  --vae_channels 48 \
  --num_frames 49 \
  --controlnet_weights 0.5 \
  --controlnet_stride 3 \
  --init_from_transformer \
  --gradient_accumulation_steps 2 \
  --allow_tf32 \
  --validation_prompt "$DATA_BASE/text_embeds" \
  --validation_prompt_log "$DATA_BASE/descriptions.txt" \
  --negative_prompt_latent "$DATA_BASE/negative_prompt_latent.pt" \
  --validation_images "$DATA_BASE/latents_48ch_1st_frames" \
  --video_path "$DATA_BASE/controlnet_latents" \
  --output_path "$DATA_BASE/val_vids_gen_stage_2" \
  --resume_from_checkpoint "<path-to-checkpoint>" \
  --stage_2_training
```

## Acknowledgements
Some parts of the code were borrowed from [here](https://github.com/TheDenk/wan2.2-controlnet) and fully adapted for the i2v pipeline. Original diffusers inference code of [Wan2.2](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers).  


## Citations
```
@misc{slava,
    title={Wan2.2 Controlnet},
    author={Viacheslav Iablochnikov},
    url={https://github.com/slava-qw/wan2.2-controlnet},
    publisher={Github},
    year={2025}
}
```
