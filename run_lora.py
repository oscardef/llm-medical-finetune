# run_lora.py

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def parse_args():
    p = argparse.ArgumentParser(
        description="Run inference on a LoRA‐finetuned causal LM"
    )
    p.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="The HuggingFace ID of your base model (e.g. TinyMistral-248M)"
    )
    p.add_argument(
        "--lora_dir",
        type=str,
        required=True,
        help="Path to the folder containing adapter_config.json + adapter_model.safetensors"
    )
    p.add_argument(
        "--prompt",
        type=str,
        default="Hello, how are you?",
        help="Text prompt to feed into the model"
    )
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Number of tokens to generate after the prompt"
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature"
    )
    p.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling parameter"
    )
    return p.parse_args()


def main():
    args = parse_args()

    # 1) Load base model in half‐precision if a GPU is available (else full precision)
    dtype = (
        torch.float16
        if torch.cuda.is_available()
        else torch.float32
    )
    device_map = "auto" if torch.cuda.is_available() else None

    print(f">>> Loading base model {args.base_model} (dtype={dtype}, device_map={device_map})…")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )

    # 2) Wrap with PeftModel to load ONLY the LoRA adapter weights
    print(f">>> Applying LoRA adapter from {args.lora_dir} …")
    model = PeftModel.from_pretrained(
        base,
        args.lora_dir,
        torch_dtype=dtype,
        device_map=device_map
    )
    model.eval()

    # 3) Load tokenizer (must come from the same base_model)
    print(">>> Loading tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # 4) Tokenize the prompt
    inputs = tokenizer(
        args.prompt,
        return_tensors="pt",
    ).to(model.device)

    # 5) Generate
    print(">>> Generating…")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            top_k=args.top_k,
            temperature=args.temperature,
        )

    # 6) Decode & print
    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("\n=== OUTPUT ===\n")
    print(generated)
    print("\n=== END ===\n")


if __name__ == "__main__":
    main()
