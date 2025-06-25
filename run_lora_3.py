#!/usr/bin/env python3
"""
run_lora_chat.py ────────────────────────────────────────────────────────────
Chat interactif avec un modèle causal-LM (Mistral-7B, TinyMistral-248M, …)
finement-ajusté par LoRA.

• Charge le modèle de base en 8-bit si un GPU CUDA est dispo.
• Applique l’adapter LoRA sauvegardé (adapter_config.json + adapter_model.safetensors)
• Conserve l’historique USER / ASSISTANT et le tronque pour rester ≤ max_len tokens.
• Utilise exactement le schéma de prompt vu pendant le fine-tune :
      <s>[INST] … [/INST]  Réponse </s>

Dépendances : transformers, peft, torch, bitsandbytes, accelerate
"""

import argparse, os, torch, collections
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
)
from peft import PeftModel


# ╭─────────────────────────── CLI ────────────────────────────╮
def parse_args():
    p = argparse.ArgumentParser("Chat with a LoRA-fine-tuned Mistral model")
    p.add_argument("--base_model", required=True,
                   help="e.g. mistralai/Mistral-7B-v0.1")
    p.add_argument("--lora_dir",   required=True,
                   help="Folder with adapter_config.json / adapter_model.safetensors")
    p.add_argument("--max_len",    type=int, default=896,
                   help="Max context length (prompt history)")
    p.add_argument("--max_gen",    type=int, default=128,
                   help="Max tokens to generate for each reply")
    p.add_argument("--temperature",type=float, default=0.7)
    p.add_argument("--top_k",      type=int,   default=0,
                   help="0 ⇒ désactivé (utilise top_p)")
    p.add_argument("--top_p",      type=float, default=0.9)
    p.add_argument("--min_gen",    type=int,   default=10,
                   help="Force at least N new tokens")
    return p.parse_args()


# ╭──────────────────── Prompt helper (matches training) ───────────────────╮
def build_prompt(user_msg: str) -> str:
    return (
        "<s>[INST] Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        f"{user_msg.strip()}\n"
        "[/INST] "
    )


# ╭─────────────────────── Model + tokenizer loader ─────────────────────────╮
def load_model_and_tok(base_id: str, lora_dir: str, token: str | None):
    cuda = torch.cuda.is_available()
    q_cfg = BitsAndBytesConfig(load_in_8bit=True) if cuda else None

    base = AutoModelForCausalLM.from_pretrained(
        base_id,
        device_map="auto" if cuda else None,
        quantization_config=q_cfg,
        torch_dtype=torch.float16 if cuda else torch.float32,
        low_cpu_mem_usage=True,
        use_auth_token=token,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(
        base, lora_dir, is_trainable=False,
        device_map="auto" if cuda else None,
        torch_dtype=base.dtype,
        use_auth_token=token,
        trust_remote_code=True,
    )
    model.eval();  model.config.use_cache = True

    tok = AutoTokenizer.from_pretrained(
        base_id, use_fast=True, use_auth_token=token, trust_remote_code=True
    )
    tok.padding_side = "right"
    tok.pad_token_id = tok.eos_token_id if tok.pad_token_id is None else tok.pad_token_id
    return tok, model


# ╭──────────────────── Assemble history into single prompt ─────────────────╮
def prompt_from_history(hist):
    # hist = deque([(role, msg), ...])  role ∈ {"user","assistant"}
    parts = []
    for role, msg in hist:
        if role == "user":
            parts.append(f"<s>[INST] {msg.strip()} [/INST] ")
        else:  # assistant
            parts.append(f"{msg.strip()} </s>")
    return "".join(parts)


# ╭──────────────────────────── Chat loop ───────────────────────────────────╮
def main():
    args = parse_args()
    hf_token = os.getenv("HF_HUB_TOKEN") or os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("Please export HF_HUB_TOKEN before running.")

    tok, model = load_model_and_tok(args.base_model, args.lora_dir, hf_token)
    history = collections.deque()       # (role, msg)

    print("\nChat started — empty line to quit.")
    while True:
        user_msg = input("\nUSER >>> ").strip()
        if user_msg == "":
            break

        history.append(("user", user_msg))
        prompt = prompt_from_history(history)

        # tokenise + raccourcir si dépasse max_len
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        while inputs.input_ids.shape[-1] > args.max_len and len(history) > 1:
            history.popleft()                        # retire le tour le plus ancien
            prompt = prompt_from_history(history)
            inputs = tok(prompt, return_tensors="pt").to(model.device)

        prompt_len = inputs.input_ids.shape[-1]

        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                min_new_tokens=args.min_gen,
                max_new_tokens=args.max_gen,
                temperature=args.temperature,
                top_k=args.top_k if args.top_k > 0 else None,
                top_p=args.top_p,
                do_sample=True,
                pad_token_id=tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
            )

        answer = tok.decode(gen_ids[0, prompt_len:], skip_special_tokens=True).strip()
        print("\nASSISTANT >>>", answer)
        history.append(("assistant", answer))


if __name__ == "__main__":
    main()
