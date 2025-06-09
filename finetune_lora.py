#!/usr/bin/env python3
"""
finetune_lora.py

Fine-tune a pretrained causal-LM with LoRA + DeepSpeed Stage 3 on RCP.
Eliminates HF/PEFT/BnB warnings at the root, batches & caches tokenization,
uses FP16 + 8-bit AdamW, and scales across multiple GPUs.

Usage under torchrun (8 GPUs):
  export HF_HUB_TOKEN="hf_XXXXX"
  export MASTER_ADDR=127.0.0.1
  export MASTER_PORT=29500
  torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    finetune_lora.py \
      --base_model   mistralai/Mistral-7B-v0.1 \
      --conv_json    /scratch/data/data_conversation/train_conversations.json \
      --qcm_json     /scratch/data/data_questions/train_formatted.json \
      --output_dir   /scratch/lora_ckpts_mistral7b \
      --train_pct    100 \
      --epochs       10 \
      --bsz          2 \
      --grad_accum   8 \
      --learning_rate 1e-4 \
      --max_len      512 \
      --save_steps   500
"""

import os, warnings
# ──────────────────────────────────────────────────────────────────────────────
# 1) Eliminate Python‐level warnings at import time
warnings.filterwarnings("ignore")

from transformers import logging as hf_logging
from peft import logging as peft_logging
import bitsandbytes as bnb

hf_logging.set_verbosity_error()     # silence 🤗 Transformers below ERROR
peft_logging.set_verbosity_error()   # silence 🤗 PEFT below ERROR
bnb.logging.set_verbosity_error()    # silence bitsandbytes logs below ERROR
# ──────────────────────────────────────────────────────────────────────────────

import argparse
import torch
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model",    type=str,   required=True)
    p.add_argument("--conv_json",     type=str,   required=True)
    p.add_argument("--qcm_json",      type=str,   required=True)
    p.add_argument("--output_dir",    type=str,   required=True)
    p.add_argument("--train_pct",     type=float, default=100.0)
    p.add_argument("--epochs",        type=int,   default=10)
    p.add_argument("--bsz",           type=int,   default=8)
    p.add_argument("--grad_accum",    type=int,   default=2)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--max_len",       type=int,   default=512)
    p.add_argument("--save_steps",    type=int,   default=1000)
    return p.parse_args()


def load_8bit_lora(base_id: str, auth_token: str = None):
    # 1) Config
    config = AutoConfig.from_pretrained(
        base_id,
        use_auth_token=auth_token,
        trust_remote_code=True,
    )
    config.model_parallel = False
    setattr(config, "tensor_parallel", False)

    # 2) Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_id,
        use_fast=True,
        use_auth_token=auth_token,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 3) Base model (8-bit if CUDA else FP16/FP32)
    use_8bit = torch.cuda.is_available()
    if use_8bit:
        print(">>> Loading model in 8-bit + LoRA")
        bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_id,
            config=config,
            quantization_config=bnb_cfg,
            device_map="auto",
            torch_dtype=torch.float16,
            use_cache=False,
            low_cpu_mem_usage=True,
            use_auth_token=auth_token,
            trust_remote_code=True,
        )
    else:
        print(">>> Loading model in full precision + LoRA")
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        base_model = AutoModelForCausalLM.from_pretrained(
            base_id,
            config=config,
            torch_dtype=dtype,
            device_map=None,
            use_cache=False,
            low_cpu_mem_usage=True,
            use_auth_token=auth_token,
            trust_remote_code=True,
        )

    # 4) Apply LoRA + gradient checkpointing + k-bit prep
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
    )
    model = get_peft_model(base_model, lora_cfg)
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    return tokenizer, model, use_8bit


def subset(ds: Dataset, pct: float, seed: int = 42):
    if pct >= 100.0:
        return ds
    n = int(len(ds) * pct / 100.0)
    return ds.shuffle(seed=seed).select(range(n))


def make_tokenizer_fn(tokenizer, max_len):
    def fn(batch):
        prompts = [
            f"<s>[INST] {i} {inp} [/INST] "
            for i, inp in zip(batch["instruction"], batch["input"])
        ]
        answers = [out + " </s>" for out in batch["output"]]
        full = [p + a for p, a in zip(prompts, answers)]

        enc_full = tokenizer(
            full,
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_attention_mask=True,
        )
        enc_prompt = tokenizer(
            prompts,
            truncation=True,
            max_length=max_len,
            padding="max_length",
        )["input_ids"]

        labels = []
        for ids, pids in zip(enc_full["input_ids"], enc_prompt):
            lab = ids.copy()
            plen = sum(1 for t in pids if t != tokenizer.pad_token_id)
            for i in range(plen):
                lab[i] = -100
            labels.append(lab)

        return {
            "input_ids":      enc_full["input_ids"],
            "attention_mask": enc_full["attention_mask"],
            "labels":         labels,
        }
    return fn


def build_train_dataset(tokenizer, args):
    ds_c = load_dataset("json", data_files=args.conv_json)["train"]
    ds_q = load_dataset("json", data_files=args.qcm_json)["train"]

    ds_c = subset(ds_c, args.train_pct)
    ds_q = subset(ds_q, args.train_pct)
    ds  = concatenate_datasets([ds_c, ds_q]).shuffle(42)

    ds = ds.map(
        make_tokenizer_fn(tokenizer, args.max_len),
        batched=True,
        batch_size=2000,
        remove_columns=ds.column_names,
        num_proc=2,
    )
    return ds


def main():
    args     = parse_args()
    hf_token = os.getenv("HF_HUB_TOKEN") or os.getenv("HF_TOKEN")
    assert hf_token, "Set HF_HUB_TOKEN"

    print(f">>> Loading {args.base_model} …")
    tokenizer, model, use_8bit = load_8bit_lora(args.base_model, auth_token=hf_token)

    print(">>> Tokenizing dataset…")
    train_ds = build_train_dataset(tokenizer, args)

    print(">>> Configuring Trainer…")
    os.makedirs(args.output_dir, exist_ok=True)

    ds_config = {
      "zero_optimization": {
        "stage": 3,
        "stage3_gather_16bit_weights_on_model_save": True,
        "offload_optimizer": {"device": "cpu","pin_memory": True},
      },
      "fp16": {"enabled": True},
      "gradient_clipping": 1.0,
      "train_micro_batch_size_per_gpu": args.bsz,
      "gradient_accumulation_steps": args.grad_accum,
    }

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        save_steps=args.save_steps,
        save_total_limit=3,
        fp16=True,
        logging_steps=200,
        deepspeed=ds_config,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=default_data_collator,
        label_names=["labels"],   # suppress missing label_names warning
    )

    print(">>> Starting training…")
    trainer.train()

    print(f">>> Saving adapter + tokenizer to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("✔ Done!")

if __name__ == "__main__":
    main()
