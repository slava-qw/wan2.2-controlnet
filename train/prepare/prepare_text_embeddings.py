import os
import re
import argparse

import html
import ftfy
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, UMT5EncoderModel


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()

def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text
    
@torch.no_grad
def extract_text_embeddings(tokenizer, text_encoder, prompt, max_sequence_length=512, device=torch.device("cuda"), dtype=torch.bfloat16):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt = [prompt_clean(u) for u in prompt]

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
    seq_lens = mask.gt(0).sum(dim=1).long()

    try:
        prompt_embeds = text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )
        
        # Clean up intermediate tensors
        del text_input_ids, mask, seq_lens
        if device.type == "cuda":
            torch.cuda.empty_cache()
            
        return prompt_embeds
    except Exception as e:
        print(f"Error in extract_text_embeddings: {e}")
        if device.type == "cuda":
            print(f"GPU memory at error: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        raise


def main(args):
    torch_dtype = torch.bfloat16
    if args.dtype == "fp32":
        torch_dtype = torch.float32
    elif args.dtype == "fp16":
        torch_dtype = torch.bfloat16
    
    device = torch.device("cuda" if args.device == "cuda" else "cpu")
    
    # Clear GPU memory before loading
    if device.type == "cuda":
        torch.cuda.empty_cache()
        print(f"GPU memory before loading: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print(f"GPU memory available: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
    
    print('START MODEL DOWNLOADING')
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, subfolder="tokenizer")
        print("Tokenizer loaded successfully")
        
        # Load model with low_cpu_mem_usage to reduce memory pressure
        text_encoder = UMT5EncoderModel.from_pretrained(
            args.base_model_path, 
            subfolder="text_encoder", 
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            device_map="auto" if device.type == "cuda" else None
        )
        
        if device.type == "cuda":
            text_encoder = text_encoder.to(device=device)
            print(f"GPU memory after loading: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
            print(f"GPU memory reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
        
        print(f"MODEL HAS BEEN LOADED TO {device} WITH DTYPE: {torch_dtype}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        if device.type == "cuda":
            print(f"GPU memory at error: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        raise

    os.makedirs(args.out_embeds_dir, exist_ok=True)

    if not args.save_one:
        df = pd.read_csv(args.csv_path)
        for i, row in tqdm(df.iterrows(), total=df.shape[0]):
            assert (vid_id := row['ids']) == i, f'Wronng order of videos and ids in dataset; got {i=}, {vid_id=}'
            
            out_embeds_path = os.path.join(args.out_embeds_dir, f'{row["video"].split("_" if "_" in row["video"] else ".")[0]}_{vid_id}.pt')  # for reference see the .csv file
            if os.path.exists(out_embeds_path):
                continue

            prompt_embeds = extract_text_embeddings(tokenizer, text_encoder, row["caption"]).cpu()
            torch.save(prompt_embeds, out_embeds_path)
    else:
        prompt_embeds = extract_text_embeddings(tokenizer, text_encoder, args.negative_prompt).cpu()
        out_embeds_path = os.path.join(args.out_embeds_dir, 'negative_prompt_latent.pt')
        torch.save(prompt_embeds, out_embeds_path)

# CUDA_VISIBLE_DEVICES=0 python prepare_text_embeddings.py \
# --csv_path "path to csv" \
# --out_embeds_dir "path to output dir" \
# --base_model_path "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
# --device "cuda" \
# --dtype "bf16"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a text embeddings for training.")
    parser.add_argument(
        "--base_model_path", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", help="The path of the pre-trained model with text encoder"
    )
    parser.add_argument("--csv_path", type=str, required=True, help="Path to csv file from OpenVid dataset or other with text in column 'caption'.")
    parser.add_argument("--out_embeds_dir", type=str, required=True, help="Directory for embeddings")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--dtype", type=str, default="bf16", help="fp32, fp16 or bf16")

    parser.add_argument("--negative_prompt", type=str, default="", help="negative promt to calculate text embeds only once")
    parser.add_argument("--save_one", type=bool, default=False, help="Flag to indicate wether save only one embedding or all of them, needed for validation encoding")
    args = parser.parse_args()
    main(args)
