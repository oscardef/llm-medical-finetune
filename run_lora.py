"""
run_lora.py

Load a LoRA-fine­tuned Mistral-7B (or TinyMistral-248M) and enter a prompt loop.
Each time you type a new prompt, the script generates and prints a response.

Requirements:
  • HF_HUB_TOKEN must be set if the base model is gated.
  • Python packages: transformers, peft, torch, bitsandbytes, accelerate

Example usage:

  export HF_HUB_TOKEN="hf_XXXXXXXXXXXXXXXXXXXXXXXX"
  python3 run_lora.py \
    --base_model mistralai/Mistral-7B-v0.1 \
    --lora_dir   /scratch/lora_ckpts_mistral7b \
    --max_new_tokens 64 \
    --temperature 0.8 \
    --top_k 50
"""

import os
import warnings

# ──────────────────────────────────────────────────────────────────────────────
# Silence all Python/TF/PEFT/BnB warnings at import time
warnings.filterwarnings("ignore")

from transformers import logging as hf_logging
from peft import logging as peft_logging
import bitsandbytes as bnb

hf_logging.set_verbosity_error()
peft_logging.set_verbosity_error()
bnb.logging.set_verbosity_error()
# ──────────────────────────────────────────────────────────────────────────────

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run interactive inference with a LoRA-fine­tuned Mistral-7B (or TinyMistral-248M)."
    )
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="HF ID of the base model (e.g. mistralai/Mistral-7B-v0.1)."
    )
    parser.add_argument(
        "--lora_dir",
        type=str,
        required=True,
        help="Path to folder with adapter_config.json + adapter_model.safetensors"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Max number of tokens to generate per input."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (e.g. 0.8)."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling parameter (e.g. 50)."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1) Read HF token if needed
    hf_token = os.getenv("HF_HUB_TOKEN")
    if hf_token is None:
        raise RuntimeError("HF_HUB_TOKEN is not set. Please export it before running.")

    # 2) Decide device + dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    device_map = "auto" if device.type == "cuda" else None

    # 3) Load base model (FP16 on GPU or FP32 on CPU)
    print(f">>> Loading base model {args.base_model} (dtype={dtype}, device_map={device_map})…")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
        use_auth_token=hf_token,
        trust_remote_code=True,
        local_files_only=False,
    )

    # 4) Apply LoRA adapter
    print(f">>> Applying LoRA adapter from {args.lora_dir} …")
    model = PeftModel.from_pretrained(
        base_model,
        args.lora_dir,
        torch_dtype=dtype,
        device_map=device_map,
        use_auth_token=hf_token,
        trust_remote_code=True,
        local_files_only=True,  # don’t attempt remote config lookups
    )
    model.eval()

    # 5) Load tokenizer (right-padding for consistency with training)
    print(">>> Loading tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        use_fast=True,
        use_auth_token=hf_token,
        trust_remote_code=True,
        local_files_only=False,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 6) Interactive loop
    print("\n>>> Enter prompt (empty line to quit):\n")
    while True:
        try:
            user_input = input(">>> ")
        except EOFError:
            print("\nExiting.")
            break

        if user_input.strip() == "":
            print("Exiting.")
            break

        # 7) Build the exact same "[INST]…[/INST]" wrapper used during training
        prompt_text = (
            "<s>[INST] You are a medical doctor expert. "
            "Here is a conversation with a patient—give them a solution or an answer: "
            + user_input
            + " [/INST] "
        )

        # 8) Tokenize wrapped prompt
        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding="max_length",
        ).to(device)

        # 9) Keep a decoded version of the prompt (no special tokens) to strip later
        prompt_decoded = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

        # 10) Generate output
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_k=args.top_k,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # 11) Decode full output (prompt + answer), then strip the prompt
        full_decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        answer = full_decoded.replace(prompt_decoded, "").strip()

        print("\n--- Model response ---\n")
        print(answer)
        print("\n----------------------\n")


if __name__ == "__main__":
    main()
