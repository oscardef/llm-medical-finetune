#!/usr/bin/env python
"""
Fine-tune (SFT) a HuggingFace model with LoRA 8-bit on two datasets:
  - data_conversation  (chat-style data between patient <-> doctor)
  - data_questions     (medical MCQs)

Automatically saves checkpoints and the final adapter.
"""

#
# LoRA 8-bit
#


import argparse, os, torch, json, random
from datasets import load_dataset, concatenate_datasets
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          TrainingArguments, Trainer,
                          BitsAndBytesConfig)
from peft import LoraConfig, TaskType, get_peft_model

# ---------- 1. CLI Arguments ----------
parser = argparse.ArgumentParser()
parser.add_argument("--base_model", default="mistralai/Mistral-7B-v0.1")
parser.add_argument("--conv_json",  default="data_conversation/train_conversations.json")
parser.add_argument("--qcm_json",   default="data_questions/train_formatted.json")
parser.add_argument("--output_dir", default="lora_ckpts")
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--bsz",    type=int, default=2)
parser.add_argument("--save_steps", type=int, default=500)
parser.add_argument("--max_len", type=int, default=1024)
args = parser.parse_args()

# ---------- 2. Tokenizer & 8-bit base model ----------
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token  # required for padding

base_model = AutoModelForCausalLM.from_pretrained(
    args.base_model,
    device_map="auto",
    quantization_config=bnb_config
)

# ---------- 3. LoRA Adapter ----------
lora_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj","k_proj","v_proj","o_proj"]
)
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

# ---------- 4. Dataset processing ----------
def format_sample(example):
    prompt  = f"<s>[INST] {example['instruction']} {example['input']} [/INST] "
    answer  = f"{example['output']} </s>"
    full_text = prompt + answer
    tokenized = tokenizer(full_text, truncation=True, max_length=args.max_len)
    labels  = tokenized["input_ids"].copy()
    # mask the prompt part to ignore it in the loss computation
    prompt_len = len(tokenizer(prompt)["input_ids"])
    labels[:prompt_len] = [-100] * prompt_len
    tokenized["labels"] = labels
    return tokenized

conv_dataset = load_dataset("json", data_files=args.conv_json)["train"]
qcm_dataset  = load_dataset("json", data_files=args.qcm_json )["train"]
train_dataset = concatenate_datasets([conv_dataset, qcm_dataset]).shuffle(seed=42)
train_dataset = train_dataset.map(format_sample, remove_columns=train_dataset.column_names, num_proc=4)

# ---------- 5. Training ----------
training_args = TrainingArguments(
    output_dir           = args.output_dir,
    per_device_train_batch_size = args.bsz,
    gradient_accumulation_steps = 8 // args.bsz,
    num_train_epochs     = args.epochs,
    learning_rate        = 2e-4,
    logging_steps        = 50,
    save_steps           = args.save_steps,
    save_total_limit     = 3,
    fp16                 = torch.cuda.is_available(),
    bf16                 = False,
    report_to            = "none"
)
trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
trainer.train()
trainer.save_model(args.output_dir)      # => adapter_model.bin + config
tokenizer.save_pretrained(args.output_dir)
