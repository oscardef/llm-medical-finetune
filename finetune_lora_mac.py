#!/usr/bin/env python3
"""
finetune_lora_mac.py

A Mac / CPU / MPS–compatible version of finetune_lora.py that:
  • Omits all 8-bit / bitsandbytes code (since that doesn’t run on MPS/CPU).
  • Forces device_map=None to avoid Mistral’s “NoneType is not iterable” bug.
  • Loads the TinyMistral (or any HuggingFace) model in FP16/FP32 on MPS or CPU.
  • Subsamples your train JSONs by a percentage (--train_pct) so that it runs quickly.

Usage example:
  python3 finetune_lora_mac.py \
    --base_model Locutusque/TinyMistral-248M-v2.5 \
    --conv_train data_conversation/train_conversations.json \
    --conv_test  data_conversation/test_conversations.json \
    --qcm_train  data_questions/train_formatted.json \
    --qcm_valid  data_questions/validation_formatted.json \
    --qcm_test   data_questions/test_formatted.json \
    --output_dir lora_test_local \
    --train_pct  1 \
    --epochs     1 \
    --bsz        2 \
    --grad_accum 4 \
    --max_len    1024 \
    --save_steps 200
"""

import argparse
import math
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, TaskType, get_peft_model


def parse_args():
    p = argparse.ArgumentParser()
    # Model & I/O arguments
    p.add_argument(
        "--base_model",
        default="mistralai/Mistral-7B-v0.1",
        help="HuggingFace‐style model ID (e.g. Locutusque/TinyMistral-248M-v2.5)",
    )
    p.add_argument(
        "--conv_train", required=True, help="Path to train_conversations.json"
    )
    p.add_argument(
        "--conv_test", required=True, help="Path to test_conversations.json"
    )
    p.add_argument(
        "--qcm_train", required=True, help="Path to train_formatted.json"
    )
    p.add_argument(
        "--qcm_valid", required=True, help="Path to validation_formatted.json"
    )
    p.add_argument(
        "--qcm_test", required=True, help="Path to test_formatted.json"
    )
    p.add_argument(
        "--output_dir", default="lora_ckpts_mac", help="Where to save LoRA adapters"
    )
    # Training hyperparameters
    p.add_argument(
        "--train_pct",
        type=float,
        default=100.0,
        help="What percent of *each* train split to keep (0–100).",
    )
    p.add_argument("--epochs", type=int, default=1, help="Number of epochs to train")
    p.add_argument("--bsz", type=int, default=2, help="Batch size per device")
    p.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation")
    p.add_argument("--max_len", type=int, default=1024, help="Max seq length")
    p.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps",
    )
    return p.parse_args()


def load_cpu_lora(base_id: str, device: torch.device):
    """
    Load the tokenizer and the base model *without* 8-bit quantization.
    We explicitly pass device_map=None so Transformers does not attempt to parallelize.
    Then wrap it in a LoRA adapter.
    """
    # 1) Load the tokenizer (fast=False for consistency with LoRA code).
    tok = AutoTokenizer.from_pretrained(base_id, use_fast=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # 2) Load the base model in FP16 (if MPS) or FP32 (if CPU). No device_map/8bit.
    dtype = torch.float16 if device.type == "mps" else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        base_id,
        torch_dtype=dtype,
        device_map=None,          # DO NOT let HF try to parallelize
        low_cpu_mem_usage=True,   # helps reduce footprint on Mac
        revision="main",          # ensure you’re loading the main branch
    )
    # move to MPS/CPU explicitly
    base_model.to(device)

    # 3) Create a LoRA config (same as you used before)
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(base_model, lora_cfg)
    return tok, model


def mask_and_tokenize(tok, max_len):
    """
    Given a tokenzier `tok` and a maximum length, return a function that:
      • concatenates [INST] instruction + input + [/INST] + output
      • tokenizes+truncates to max_len
      • sets labels to -100 for the prompt portion so that loss is only computed on the answer.
    """
    def _inner(ex):
        prompt = f"<s>[INST] {ex['instruction']} {ex['input']} [/INST] "
        answer = f"{ex['output']} </s>"

        enc = tok(prompt + answer, truncation=True, max_length=max_len)
        labels = enc["input_ids"].copy()
        prompt_len = len(tok(prompt)["input_ids"])
        labels[:prompt_len] = [-100] * prompt_len
        enc["labels"] = labels
        return enc

    return _inner


def subset(dataset, pct: float, seed: int = 42):
    """
    If pct < 100, randomly select only `pct%` of the examples.
    Otherwise return dataset unchanged.
    """
    if pct >= 100.0:
        return dataset
    take = int(len(dataset) * pct / 100.0)
    return dataset.shuffle(seed=seed).select(range(take))


def build_splits(tok, args):
    """
    Loads JSON files via `datasets`, sub-samples train splits by args.train_pct,
    concatenates conversation + QCM datasets for train/test as needed,
    then applies the masking/tokenization pipeline.
    Returns (train_ds, valid_ds, test_ds).
    """
    conv_train = load_dataset("json", data_files=args.conv_train)["train"]
    conv_test = load_dataset("json", data_files=args.conv_test)["train"]
    qcm_train = load_dataset("json", data_files=args.qcm_train)["train"]
    qcm_valid = load_dataset("json", data_files=args.qcm_valid)["train"]
    qcm_test = load_dataset("json", data_files=args.qcm_test)["train"]

    # 1) Sub-sample the *train* splits by percentage.
    conv_train = subset(conv_train, args.train_pct)
    qcm_train = subset(qcm_train, args.train_pct)

    # 2) Concatenate conv_train + qcm_train into a single train_ds
    train_ds = concatenate_datasets([conv_train, qcm_train]).shuffle(seed=42)

    # 3) Use qcm_valid as valid set, and concatenate conv_test + qcm_test as test set
    valid_ds = qcm_valid
    test_ds = concatenate_datasets([conv_test, qcm_test])

    # 4) Map the masking/tokenization to each split
    fn = mask_and_tokenize(tok, args.max_len)
    train_ds = train_ds.map(fn, remove_columns=train_ds.column_names, num_proc=4)
    valid_ds = valid_ds.map(fn, remove_columns=valid_ds.column_names, num_proc=4)
    test_ds = test_ds.map(fn, remove_columns=test_ds.column_names, num_proc=4)

    return train_ds, valid_ds, test_ds


def compute_metrics(eval_preds):
    """
    Compute perplexity from eval_loss. (Only used if evaluation_strategy != "no".)
    """
    loss = eval_preds["eval_loss"]
    return {"perplexity": math.exp(loss) if loss < 20 else float("inf")}


def main():
    args = parse_args()

    # 1) Detect device: use MPS if available, else CPU.
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f">>> Using device: {device}", flush=True)

    # 2) Load tokenizer + (LoRA‐wrapped) model on CPU/MPS
    tok, model = load_cpu_lora(args.base_model, device)

    # 3) Build train/val/test Dataset objects
    train_ds, valid_ds, test_ds = build_splits(tok, args)

    # 4) Create TrainingArguments for HuggingFace Trainer
    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=2e-4,
        evaluation_strategy="epoch",
        save_steps=args.save_steps,
        save_total_limit=3,
        logging_steps=50,
        fp16=(device.type == "mps"),  # enable fp16 only on MPS
        bf16=False,
        report_to="none",
        push_to_hub=False,
    )

    # 5) Instantiate Trainer
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        compute_metrics=compute_metrics,
    )

    # 6) Train
    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)

    # 7) Evaluate on test set
    test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
    print(f"\n★ Test-set perplexity : {test_metrics['test_perplexity']:.2f}", flush=True)


if __name__ == "__main__":
    main()
