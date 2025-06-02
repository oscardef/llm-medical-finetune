#!/usr/bin/env python
"""
finetune_lora.py

Fine-tune a pretrained causal-LM with LoRA on two JSON files:
  1) Conversation data (data_conversation/*.json)
  2) MCQ data (data_questions/*.json)

All hyperparameters are passed in via command-line, so you can run
identical commands on your Mac (CPU/MPS) or on RCP (GPU).

Requirements (Mac/venv or RCP Docker):
    pip install transformers datasets sentencepiece peft bitsandbytes accelerate

Example usage (Mac or RCP):

    python3 finetune_lora.py \
      --base_model   Locutusque/TinyMistral-248M \
      --conv_json    data_conversation/train_conversations.json \
      --qcm_json     data_questions/train_formatted.json \
      --output_dir   lora_ckpts \
      --train_pct    1 \
      --epochs       1 \
      --bsz          2 \
      --grad_accum   4 \
      --learning_rate 2e-4 \
      --max_len      1024 \
      --save_steps   200
"""

import argparse
import math
import os
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, TaskType, get_peft_model


def parse_args():
    p = argparse.ArgumentParser(description="LoRA-fine­tune a pretrained causal-LM on JSON data")
    # Required arguments
    p.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="HuggingFace model ID (e.g. TinyMistral-248M or mistralai/Mistral-7B-v0.1)",
    )
    p.add_argument(
        "--conv_json", type=str, required=True, help="Path to conversation JSON file"
    )
    p.add_argument(
        "--qcm_json", type=str, required=True, help="Path to MCQ JSON file"
    )
    p.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save the LoRA-adapted model"
    )

    # Hyperparameters
    p.add_argument(
        "--train_pct",
        type=float,
        default=100.0,
        help="Percentage (0-100) of each dataset to keep",
    )
    p.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    p.add_argument("--bsz", type=int, default=2, help="Per-device train batch size")
    p.add_argument(
        "--grad_accum", type=int, default=4, help="Gradient accumulation steps"
    )
    p.add_argument(
        "--learning_rate", type=float, default=2e-4, help="Learning rate"
    )
    p.add_argument(
        "--max_len", type=int, default=1024, help="Max token length (padding/truncation)"
    )
    p.add_argument(
        "--save_steps", type=int, default=200, help="Save checkpoint every N steps"
    )

    return p.parse_args()


def load_8bit_lora(base_id: str):
    """
    1) Load config and force model_parallel=False/tensor_parallel=False.
    2) Load tokenizer.
    3) If CUDA is available, attempt 8-bit + LoRA. If not (MPS or CPU),
       skip 8-bit and load full-precision + LoRA.
    """

    # 1) Load config and disable model_parallel if necessary
    config = AutoConfig.from_pretrained(base_id)
    config.model_parallel = False
    setattr(config, "tensor_parallel", False)

    # 2) Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3) Decide whether to use 8-bit
    use_8bit = False
    if torch.cuda.is_available():
        use_8bit = True
    elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
        # MPS does not support bitsandbytes GPU quantization
        use_8bit = False
    else:
        # CPU only
        use_8bit = False

    if use_8bit:
        print(">>> CUDA detected: loading model in 8-bit quantized mode + LoRA")
        bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_id,
            config=config,
            device_map="auto",
            quantization_config=bnb_cfg,
            
        )
    else:
        device_map = "auto" if (torch.cuda.is_available() or torch.backends.mps.is_built()) else None
        dtype = torch.float16 if (torch.cuda.is_available() or torch.backends.mps.is_built()) else torch.float32
        print(">>> No CUDA (or MPS only): loading full-precision model + LoRA")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_id,
            config=config,
            device_map=device_map,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )

    # 4) Apply LoRA
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(base_model, lora_cfg)

    return tokenizer, model


def subset(dataset, pct: float, seed: int = 42):
    """
    If pct < 100, shuffle and keep only first `pct`% of rows.
    """
    if pct >= 100.0:
        return dataset
    keep_n = int(len(dataset) * pct / 100.0)
    return dataset.shuffle(seed=seed).select(range(keep_n))


def make_tokenizer_fn(tokenizer, max_len):
    """
    Returns a function that takes one example {"instruction","input","output"}
    and returns a dict of:
      - input_ids (length=max_len, padded/truncated)
      - attention_mask (length=max_len)
      - labels (length=max_len, with prompt tokens masked as -100)
    """

    def _fn(ex):
        prompt = f"<s>[INST] {ex['instruction']} {ex['input']} [/INST] "
        answer = f"{ex['output']} </s>"

        # 1) Tokenize prompt + answer
        enc_full = tokenizer(
            prompt + answer,
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_attention_mask=True,
        )
        input_ids = enc_full["input_ids"]
        attention_mask = enc_full["attention_mask"]

        # 2) Tokenize prompt alone to find prompt length
        enc_prompt = tokenizer(
            prompt,
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_attention_mask=False,
        )["input_ids"]
        prompt_len = sum(1 for tok_id in enc_prompt if tok_id != tokenizer.pad_token_id)

        # 3) Build labels = input_ids but mask prompt tokens
        labels = input_ids.copy()
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return _fn


def build_train_dataset(tokenizer, args):
    """
    1) Load conversation JSON + MCQ JSON (each a list of {"instruction","input","output"}).
    2) Subset to `train_pct`.
    3) Concatenate and shuffle.
    4) Map through our tokenization function (max_len, pad/truncation).
    """
    ds_conv = load_dataset("json", data_files=args.conv_json)["train"]
    ds_qcm = load_dataset("json", data_files=args.qcm_json)["train"]

    ds_conv = subset(ds_conv, args.train_pct)
    ds_qcm = subset(ds_qcm, args.train_pct)

    train_ds = concatenate_datasets([ds_conv, ds_qcm]).shuffle(seed=42)

    token_fn = make_tokenizer_fn(tokenizer, args.max_len)
    train_ds = train_ds.map(
        token_fn,
        remove_columns=train_ds.column_names,
        num_proc=4,
    )
    return train_ds


def main():
    args = parse_args()

    # 1) Load tokenizer + (8-bit or full) base model + LoRA
    print(f">>> Loading model {args.base_model} …")
    tokenizer, model = load_8bit_lora(args.base_model)

    # 2) Build the training dataset
    print(">>> Preparing dataset (tokenization)…")
    train_dataset = build_train_dataset(tokenizer, args)

    # 3) Set up TrainingArguments
    print(">>> Configuring Trainer…")
    os.makedirs(args.output_dir, exist_ok=True)
    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        save_steps=args.save_steps,
        save_total_limit=3,
        logging_steps=50,
        fp16=(torch.cuda.is_available()),  # MPS doesn’t support fp16 properly
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_dataset,
    )

    # 4) Train
    print(">>> Starting training…")
    trainer.train()

    # 5) Save final LoRA weights + tokenizer
    print(f">>> Saving LoRA-adapter + tokenizer to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\n✔ Training complete!")


if __name__ == "__main__":
    main()