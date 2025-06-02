#!/usr/bin/env python
# finetune_lora_mac.py
#
# Fine-tune with LoRA on TinyMistral (no 8-bit quantization) under macOS/MPS or CPU.
# This version disables BitsAndBytes entirely and forces the model onto MPS (if available).
#
import argparse
import math
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, TaskType, get_peft_model


def parse_args():
    p = argparse.ArgumentParser()
    # Model & I/O
    p.add_argument(
        "--base_model",
        default="Locutusque/TinyMistral-248M-v2.5",
        help=" Hugging Face repo ID of your TinyMistral model",
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
    p.add_argument("--qcm_test", required=True, help="Path to test_formatted.json")
    p.add_argument(
        "--output_dir",
        default="lora_ckpts_mac",
        help="Where to save LoRA checkpoints + tokenizer",
    )

    # Training hyperparameters
    p.add_argument(
        "--train_pct",
        type=float,
        default=1.0,
        help="Percentage (0–100) of the train split to subsample",
    )
    p.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    p.add_argument("--bsz", type=int, default=2, help="Batch size per device")
    p.add_argument(
        "--grad_accum",
        type=int,
        default=4,
        help="Gradient accumulation steps (to simulate a larger batch)",
    )
    p.add_argument(
        "--max_len", type=int, default=1024, help="Max token length for prompt + answer"
    )
    p.add_argument(
        "--save_steps", type=int, default=200, help="Steps between saving LoRA weights"
    )
    return p.parse_args()


def load_lora_no8bit(base_id: str):
    """
    1) Grabs config + forces model_parallel=False
    2) Loads tokenizer
    3) Loads the full-precision model with device_map='auto'
    4) Wraps it in LoRA
    """
    # ——— 1) Get config & patch out model_parallel/tensor_parallel
    config = AutoConfig.from_pretrained(base_id)
    config.model_parallel = False
    config.tensor_parallel = False

    # ——— 2) Load tokenizer (this also requires sentencepiece for Mistral)
    tok = AutoTokenizer.from_pretrained(base_id, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # ——— 3) Load the full-precision (fp16/float32) model onto MPS/CPU
    #     We do NOT pass any quantization_config here, so no 8-bit.
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f">>> Loading model on device: {device}")
    base = AutoModelForCausalLM.from_pretrained(
        base_id,
        config=config,
        device_map="auto",       # HF will pick 'mps' or 'cpu'
        torch_dtype=torch.float16 if device == "mps" else torch.float32,
    )

    # ——— 4) Attach LoRA
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(base, lora_cfg)
    return tok, model


def mask_and_tokenise(tok, max_len):
    def _inner(ex):
        # Build the "<s>[INST] ... [/INST] answer </s>" format
        prompt = f"<s>[INST] {ex['instruction']} {ex['input']} [/INST] "
        answer = f"{ex['output']} </s>"

        # Tokenize + truncate
        ids = tok(prompt + answer, truncation=True, max_length=max_len)
        labels = ids["input_ids"].copy()

        # Hide the prompt tokens in the labels (so LM only computes loss on answer)
        prompt_len = len(tok(prompt)["input_ids"])
        labels[:prompt_len] = [-100] * prompt_len
        ids["labels"] = labels
        return ids

    return _inner


def subset(dataset, pct, seed=42):
    if pct >= 100:
        return dataset
    take = int(len(dataset) * pct / 100)
    return dataset.shuffle(seed=seed).select(range(take))


def build_splits(tok, args):
    # Load each split from JSON
    conv_train = load_dataset("json", data_files=args.conv_train)["train"]
    conv_test = load_dataset("json", data_files=args.conv_test)["train"]
    qcm_train = load_dataset("json", data_files=args.qcm_train)["train"]
    qcm_valid = load_dataset("json", data_files=args.qcm_valid)["train"]
    qcm_test = load_dataset("json", data_files=args.qcm_test)["train"]

    # Subsample if requested
    conv_train = subset(conv_train, args.train_pct)
    qcm_train = subset(qcm_train, args.train_pct)

    # Concatenate conversation + QCM train, shuffle
    train_ds = concatenate_datasets([conv_train, qcm_train]).shuffle(seed=42)
    valid_ds = qcm_valid
    test_ds = concatenate_datasets([conv_test, qcm_test])

    proc = mask_and_tokenise(tok, args.max_len)
    train_ds = train_ds.map(
        proc, remove_columns=train_ds.column_names, num_proc=4
    )
    valid_ds = valid_ds.map(
        proc, remove_columns=valid_ds.column_names, num_proc=4
    )
    test_ds = test_ds.map(
        proc, remove_columns=test_ds.column_names, num_proc=4
    )

    return train_ds, valid_ds, test_ds


def compute_metrics(eval_out):
    loss = eval_out["eval_loss"]
    return {"perplexity": math.exp(loss) if loss < 20 else float("inf")}


def main():
    args = parse_args()

    # 1) Load tokenizer + TinyMistral + LoRA (no bitsandbytes)
    tok, model = load_lora_no8bit(args.base_model)

    # 2) Prepare train/valid/test datasets
    train_ds, valid_ds, test_ds = build_splits(tok, args)

    # 3) Set up HF Trainer
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f">>> Training on device: {device}")
    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=2e-4,
        save_steps=args.save_steps,
        save_total_limit=3,
        logging_steps=50,
        fp16=False,  # Use MPS‐half precision if on MPS; else full precision on CPU
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        compute_metrics=compute_metrics,
    )

    # 4) Fine-tune
    trainer.train()

    # 5) Save LoRA + tokenizer
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)

    # 6) Final evaluation on test set
    test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
    print(f"\n★ Test-set perplexity : {test_metrics['test_perplexity']:.2f}")


if __name__ == "__main__":
    main()
