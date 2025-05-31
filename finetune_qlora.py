#!/usr/bin/env python
# ────────────────────────────────────────────────────────────────────────────────
# finetune_qlora.py
#
# Supervised fine-tuning of a causal LLaMA / Mistral model
# using the QLoRA method:
#   • Base model weights quantized in 4-bit NF4
#   • LoRA adapters (r = 16) trained in float16
#
# Designed for: GPUs with 8–16 GB or A100/RTX-series.
# Inputs     : Two JSON datasets with fields {instruction, input, output}
# Outputs    : Intermediate checkpoints + final adapter folder
# ────────────────────────────────────────────────────────────────────────────────

import argparse, math, torch
from datasets import load_dataset, concatenate_datasets
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          BitsAndBytesConfig,
                          TrainingArguments, Trainer)
from peft import LoraConfig, TaskType, get_peft_model


# ╭──────────────────────────────────────────────────────────────────────────────╮
# │ 1.  Command-line arguments                                                   │
# ╰──────────────────────────────────────────────────────────────────────────────╯
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model",
                   default="mistralai/Mistral-7B-v0.1",
                   help="HF model to fine-tune (7B max recommended for 16GB VRAM)")
    p.add_argument("--conv_json",
                   default="data_conversation/train_conversations.json",
                   help="Path to conversation dataset")
    p.add_argument("--qcm_json",
                   default="data_questions/train_formatted.json",
                   help="Path to medical MCQ dataset")
    p.add_argument("--output_dir", default="qlora_ckpts",
                   help="Output directory")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--bsz", type=int, default=1,
                   help="Batch size per GPU (set to 1 if limited VRAM)")
    p.add_argument("--grad_accum", type=int, default=8,
                   help="Gradient accumulation steps (effective batch size = bsz * grad_accum)")
    p.add_argument("--max_len", type=int, default=1024,
                   help="Max (token) length for truncated samples")
    p.add_argument("--save_steps", type=int, default=500)
    return p.parse_args()


# ╭──────────────────────────────────────────────────────────────────────────────╮
# │ 2.  QLoRA model construction                                                 │
# ╰──────────────────────────────────────────────────────────────────────────────╯
def load_qlora_model(base_id: str):
    # 2-a  4-bit NF4 quantization
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit            = True,
        bnb_4bit_quant_type     = "nf4",
        bnb_4bit_compute_dtype  = torch.float16,
        bnb_4bit_use_double_quant = False
    )

    tokenizer = AutoTokenizer.from_pretrained(base_id, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_id,
        device_map         = "auto",
        quantization_config = bnb_cfg,
        trust_remote_code   = True
    )
    base_model.gradient_checkpointing_enable()   # ↓ memory, ↑ training time
    base_model.config.use_cache = False          # required for checkpointing

    # 2-b  LoRA configuration
    lora_cfg = LoraConfig(
        r               = 16,
        lora_alpha      = 32,
        lora_dropout    = 0.05,
        bias            = "none",
        task_type       = TaskType.CAUSAL_LM,
        target_modules  = ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"]  # includes MLP
    )

    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()
    return tokenizer, model


# ╭──────────────────────────────────────────────────────────────────────────────╮
# │ 3.  Dataset preparation                                                      │
# ╰──────────────────────────────────────────────────────────────────────────────╯
def build_dataset(tok, conv_path: str, qcm_path: str, max_len: int):
    conv_ds = load_dataset("json", data_files=conv_path)["train"]
    qcm_ds  = load_dataset("json", data_files=qcm_path)["train"]
    raw_ds  = concatenate_datasets([conv_ds, qcm_ds]).shuffle(seed=42)

    def format_example(ex):
        prompt  = f"<s>[INST] {ex['instruction']} {ex['input']} [/INST] "
        answer  = f"{ex['output']} </s>"
        full    = prompt + answer

        ids = tok(full, truncation=True, max_length=max_len)
        labels = ids["input_ids"].copy()

        prompt_len = len(tok(prompt)["input_ids"])
        labels[:prompt_len] = [-100] * prompt_len  # mask the prompt portion
        ids["labels"] = labels
        return ids

    proc_ds = raw_ds.map(format_example,
                         remove_columns=raw_ds.column_names,
                         num_proc=4,
                         desc="Tokenization + masking")
    return proc_ds


# ╭──────────────────────────────────────────────────────────────────────────────╮
# │ 4.  Fine-tuning                                                              │
# ╰──────────────────────────────────────────────────────────────────────────────╯
def main():
    args = parse_args()

    tokenizer, model = load_qlora_model(args.base_model)
    train_ds = build_dataset(tokenizer, args.conv_json, args.qcm_json, args.max_len)

    train_args = TrainingArguments(
        output_dir                    = args.output_dir,
        per_device_train_batch_size   = args.bsz,
        gradient_accumulation_steps   = args.grad_accum,
        num_train_epochs              = args.epochs,
        learning_rate                 = 2e-4,
        lr_scheduler_type             = "cosine",
        warmup_ratio                  = 0.05,
        logging_steps                 = 50,
        save_steps                    = args.save_steps,
        save_total_limit              = 3,
        fp16                          = torch.cuda.is_available(),
        bf16                          = False,
        report_to                     = "none"
    )

    trainer = Trainer(
        model           = model,
        args            = train_args,
        train_dataset   = train_ds 
    )

    trainer.train()
    trainer.save_model(args.output_dir)          # adapter_model.bin + config
    tokenizer.save_pretrained(args.output_dir)   # same tokenizer


if __name__ == "__main__":
    main()
