import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset, concatenate_datasets
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
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        description="LoRA multi-GPU fine-tuning (DeepSpeed-ZeRO-3, 8 bit)"
    )

    # Required paths / model id
    p.add_argument("--base_model",    required=True, type=str, help="HF model ID (e.g. mistralai/Mistral-7B-v0.1)")
    p.add_argument("--conv_json",     required=True, type=str, help="Path to conversation JSON file")
    p.add_argument("--qcm_json",      required=True, type=str, help="Path to MCQ JSON file")
    p.add_argument("--output_dir",    required=True, type=str, help="Folder to save LoRA adapters & tokenizer")

    # Hyper‑params
    p.add_argument("--train_pct",      type=float, default=100.0, help="Percent of each JSON to keep")
    p.add_argument("--epochs",         type=int,   default=3,      help="#epochs")
    p.add_argument("--bsz",            type=int,   default=4,      help="Per GPU micro batch size")
    p.add_argument("--grad_accum",     type=int,   default=8,      help="Gradient accumulation steps")
    p.add_argument("--learning_rate",  type=float, default=1e-4,   help="Learning rate")
    p.add_argument("--max_len",        type=int,   default=1024,   help="Max token len")
    p.add_argument("--save_steps",     type=int,   default=500,    help="Save checkpoint every N steps")

    return p.parse_args()


def load_lora_model(base_id: str, auth_token: str | None = None):
    """Load base model in 8‑bit, add LoRA adapters, enable Flash‑Attention 2."""

    # 1) 8‑bit quant config
    bnb_cfg = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
    )

    # 2) Load config (disable model_parallel flags that break DeepSpeed)
    cfg = AutoConfig.from_pretrained(base_id, trust_remote_code=True, use_auth_token=auth_token)
    cfg.model_parallel = False
    setattr(cfg, "tensor_parallel", False)

    # 3) Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_id, use_auth_token=auth_token, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id



    # 4) Base model (8‑bit, Flash‑Attention 2, bf16)
    model = AutoModelForCausalLM.from_pretrained(
        base_id,
        config=cfg,
        device_map="balanced",
        attn_implementation=None,  # flash_attention_2
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_cfg,
        use_auth_token=auth_token,
        trust_remote_code=True,
    )

    model.gradient_checkpointing_enable()

    # 5) LoRA
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model = prepare_model_for_kbit_training(model)

    print("→ Trainable parameters after LoRA:")
    model.print_trainable_parameters()

    return tokenizer, model

def subset(ds, pct: float):
    if pct >= 100.0:
        return ds
    keep = int(len(ds) * pct / 100)
    return ds.shuffle(seed=42).select(range(keep))


def build_dataset(tokenizer, args):
    ds_conv = load_dataset("json", data_files=args.conv_json, split="train")
    ds_qcm  = load_dataset("json", data_files=args.qcm_json,  split="train")

    ds_conv = subset(ds_conv, args.train_pct)
    ds_qcm  = subset(ds_qcm,  args.train_pct)

    ds = concatenate_datasets([ds_conv, ds_qcm]).shuffle(seed=42)

    def tok_fn(ex):
        prompt  = f"<s>[INST] {ex['instruction']} {ex['input']} [/INST] "
        answer  = f"{ex['output']} </s>"
        enc     = tokenizer(
            prompt + answer,
            truncation=True,
            max_length=args.max_len,
            padding="max_length",
            return_attention_mask=True,
        )
        # Mask prompt tokens in labels
        prompt_ids = tokenizer(prompt, truncation=True, max_length=args.max_len)["input_ids"]
        prompt_len = len(prompt_ids)
        labels = enc["input_ids"].copy()
        for i in range(prompt_len):
            labels[i] = -100
        enc["labels"] = labels
        return enc

    ds = ds.map(tok_fn, remove_columns=ds.column_names, num_proc=4)
    return ds



def main():
    args = parse_args()

    # HF token (for gated models)
    hf_token = os.getenv("HF_HUB_TOKEN") or os.getenv("HF_TOKEN")

    # 1) Model + tokenizer
    tokenizer, model = load_lora_model(args.base_model, auth_token=hf_token)

    # 2) Dataset
    train_ds = build_dataset(tokenizer, args)

    # 3) DeepSpeed ZeRO‑3 config
    ds_cfg = {
        "zero_optimization": {
            "stage": 3,
            "stage3_param_persistence_threshold": 1e5,
            "stage3_gather_16bit_weights_on_model_save": True,
        },
        "bf16": {"enabled": True},
        "gradient_clipping": 1.0,
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": args.grad_accum,
    }

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        save_steps=args.save_steps,
        save_total_limit=3,
        logging_steps=50,
        bf16=True,
        report_to="none",
        deepspeed=ds_cfg,
        remove_unused_columns=False,
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=default_data_collator,
    )

    model.is_parallelizable = True
    model.model_parallel = True

    trainer.train()

    # Save adapters + tokenizer
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\n✔ Training complete! Adapters saved to", args.output_dir)


if __name__ == "__main__":
    main()

