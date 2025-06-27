
import argparse, itertools, os, torch
from dataclasses import dataclass
from typing import Dict, List

from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

#  Prompt helpers 
INST_OPEN, INST_CLOSE, EOS = "<s>[INST] ", " [/INST]", "</s>"

def format_prompt(entry: Dict[str, str]) -> str:
    """Builds instruction + optional input section."""
    instruction = entry["instruction"].strip()
    input_part  = entry.get("input", "").strip()
    prompt = (
        f"{INST_OPEN}Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}"
    )
    if input_part:
        prompt += f"\n\n### Input:\n{input_part}"
    prompt += INST_CLOSE
    return prompt

def format_response(entry: Dict[str, str]) -> str:
    return f"\n\n### Response:\n{entry['output'].strip()} {EOS}"

#  Dataset 
class InstructionDataset(torch.utils.data.Dataset):
    def __init__(self, json_files: List[str], tokenizer, max_len: int = 1024,
                 pct: float = 100.0, seed: int = 42):
        self.tok, self.max_len = tokenizer, max_len
        data = []
        for jf in json_files:
            ds = load_dataset("json", data_files=jf, split="train")
            if pct < 100:
                keep = int(len(ds) * pct / 100)
                ds = ds.shuffle(seed=seed).select(range(keep))
            data.extend(ds)
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        prompt, completion = format_prompt(ex), format_response(ex)
        full_enc = self.tok(
            prompt + completion,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_attention_mask=True,
        )
        # length of prompt only
        prompt_ids = self.tok(
            prompt,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_attention_mask=False,
        )["input_ids"]
        prompt_len = sum(tok != self.tok.pad_token_id for tok in prompt_ids)
        labels = full_enc["input_ids"].copy()
        labels[:prompt_len] = [-100] * prompt_len  # mask prompt tokens
        return {
            "input_ids": torch.tensor(full_enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(full_enc["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

#  Custom collate fn 
@dataclass
class CollateCfg:
    pad_id: int
    ignore_index: int = -100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

def custom_collate(batch: List[Dict[str, torch.Tensor]], cfg: CollateCfg):
    keys = batch[0].keys()
    max_len = max(len(x["input_ids"]) for x in batch)
    out = {k: [] for k in keys}
    for ex in batch:
        for k in keys:
            seq = ex[k]
            if len(seq) < max_len:
                pad_val = cfg.pad_id if k != "labels" else cfg.ignore_index
                pad = torch.full((max_len - len(seq),), pad_val, dtype=seq.dtype)
                seq = torch.cat([seq, pad])
            out[k].append(seq)
    for k in keys:
        out[k] = torch.stack(out[k]).to(cfg.device)
    return out

# LoRA loader

def load_8bit_lora(model_id: str, hf_token: str | None):
    print(">>> Loading base model & applying LoRA …")
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True, use_auth_token=hf_token)
    cfg.model_parallel, cfg.tensor_parallel = False, False

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, use_auth_token=hf_token, trust_remote_code=True)
    tok.pad_token_id = tok.eos_token_id if tok.pad_token_id is None else tok.pad_token_id
    tok.padding_side = "right"

    quant = BitsAndBytesConfig(load_in_8bit=True) if torch.cuda.is_available() else None
    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=cfg,
        device_map="auto",
        use_auth_token=hf_token,
        trust_remote_code=True,
        quantization_config=quant,
    )

    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    base = prepare_model_for_kbit_training(base)
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()
    return tok, model

#  Main 

def parse_args():
    ap = argparse.ArgumentParser(description="Fine‑tune causal‑LM with LoRA on medical JSON datasets")
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--conv_json", required=True)
    ap.add_argument("--qcm_json", required=True)
    ap.add_argument("--output_dir", required=True)

    ap.add_argument("--train_pct", type=float, default=100)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bsz", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=1e-4)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--save_steps", type=int, default=200)
    return ap.parse_args()


def main():
    args = parse_args()
    hf_token = os.getenv("HF_HUB_TOKEN") or os.getenv("HF_TOKEN")

    tokenizer, model = load_8bit_lora(args.base_model, hf_token)

    print(">>> Building dataset …")
    train_ds = InstructionDataset(
        [args.conv_json, args.qcm_json], tokenizer,
        max_len=args.max_len, pct=args.train_pct,
    )

    collate_cfg = CollateCfg(pad_id=tokenizer.pad_token_id)
    data_collator = lambda batch: custom_collate(batch, collate_cfg)

    os.makedirs(args.output_dir, exist_ok=True)

    fp16_flag = False  # 8‑bit quant + fp16 is unstable; keep full precision grads

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        save_steps=args.save_steps,
        logging_steps=20,
        logging_strategy="steps",
        save_total_limit=6,
        fp16=fp16_flag,
        report_to="none",
        dataloader_pin_memory=False,
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=data_collator,
    )

    print(">>> Starting training …")
    trainer.train()

    print(f">>> Saving adapter & tokenizer to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("✔ Training complete! ")


if __name__ == "__main__":
    main()
