#!/usr/bin/env python
# Fine-tune with 8-bit + LoRA, honouring explicit train / valid / test files
# and allowing a percentage sub-sample of the train split.

import argparse, math, torch
from datasets import load_dataset, concatenate_datasets
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          BitsAndBytesConfig,
                          TrainingArguments, Trainer)
from peft import LoraConfig, TaskType, get_peft_model

# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    # model & I/O
    p.add_argument("--base_model", default="mistralai/Mistral-7B-v0.1")
    p.add_argument("--conv_train", required=True,
                   help="train_conversations.json")
    p.add_argument("--conv_test",  required=True,
                   help="test_conversations.json")
    p.add_argument("--qcm_train",  required=True,
                   help="train_formatted.json")
    p.add_argument("--qcm_valid",  required=True,
                   help="validation_formatted.json")
    p.add_argument("--qcm_test",   required=True,
                   help="test_formatted.json")
    p.add_argument("--output_dir", default="lora_ckpts")
    # training hyper-params
    p.add_argument("--train_pct", type=float, default=100,
                   help="Percentage of each *train* split to keep (0-100)")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--bsz",    type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--save_steps", type=int, default=500)
    return p.parse_args()

# ──────────────────────────────────────────────────────────────────────────────
def load_8bit_lora(base_id):
    bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
    tok = AutoTokenizer.from_pretrained(base_id, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        base_id, device_map="auto", quantization_config=bnb_cfg)
    lora_cfg = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj","k_proj","v_proj","o_proj"])
    model = get_peft_model(base, lora_cfg)
    return tok, model

# ──────────────────────────────────────────────────────────────────────────────
def mask_and_tokenise(tok, max_len):
    def _inner(ex):
        prompt  = f"<s>[INST] {ex['instruction']} {ex['input']} [/INST] "
        answer  = f"{ex['output']} </s>"
        ids = tok(prompt + answer, truncation=True, max_length=max_len)
        labels = ids["input_ids"].copy()
        prompt_len = len(tok(prompt)["input_ids"])
        labels[:prompt_len] = [-100] * prompt_len
        ids["labels"] = labels
        return ids
    return _inner

def subset(dataset, pct, seed=42):
    if pct >= 100:                          # keep everything
        return dataset
    take = int(len(dataset) * pct / 100)
    return dataset.shuffle(seed=seed).select(range(take))

def build_splits(tok, a):
    conv_train = load_dataset("json", data_files=a.conv_train)["train"]
    conv_test  = load_dataset("json", data_files=a.conv_test )["train"]
    qcm_train  = load_dataset("json", data_files=a.qcm_train )["train"]
    qcm_valid  = load_dataset("json", data_files=a.qcm_valid )["train"]
    qcm_test   = load_dataset("json", data_files=a.qcm_test  )["train"]

    # sub-sample if requested
    conv_train = subset(conv_train, a.train_pct)
    qcm_train  = subset(qcm_train,  a.train_pct)

    train_ds = concatenate_datasets([conv_train, qcm_train]).shuffle(seed=42)
    valid_ds = qcm_valid                                 # simple choice
    test_ds  = concatenate_datasets([conv_test, qcm_test])

    proc = mask_and_tokenise(tok, a.max_len)
    train_ds = train_ds.map(proc, remove_columns=train_ds.column_names, num_proc=4)
    valid_ds = valid_ds.map(proc, remove_columns=valid_ds.column_names, num_proc=4)
    test_ds  = test_ds.map(proc,  remove_columns=test_ds.column_names,  num_proc=4)
    return train_ds, valid_ds, test_ds

# ──────────────────────────────────────────────────────────────────────────────
def compute_metrics(out):
    loss = out["eval_loss"]
    return {"perplexity": math.exp(loss) if loss < 20 else float("inf")}

def main():
    a = parse_args()
    tok, model = load_8bit_lora(a.base_model)
    train_ds, valid_ds, test_ds = build_splits(tok, a)

    targs = TrainingArguments(
        output_dir              = a.output_dir,
        per_device_train_batch_size = a.bsz,
        gradient_accumulation_steps = a.grad_accum,
        num_train_epochs        = a.epochs,
        learning_rate           = 2e-4,
        evaluation_strategy     = "epoch",
        save_steps              = a.save_steps,
        save_total_limit        = 3,
        logging_steps           = 50,
        fp16                    = torch.cuda.is_available(),
        report_to               = "none")

    trainer = Trainer(model=model, args=targs,
                      train_dataset=train_ds,
                      eval_dataset =valid_ds,
                      compute_metrics=compute_metrics)
    trainer.train()
    trainer.save_model(a.output_dir)
    tok.save_pretrained(a.output_dir)

    test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
    print(f"\n★ Test-set perplexity : {test_metrics['test_perplexity']:.2f}")

if __name__ == "__main__":
    main()
