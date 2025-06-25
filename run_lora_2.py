#!/usr/bin/env python3
"""
run_lora_chat.py – chat loop pour un modèle Mistral-7B + LoRA
──────────────────────────────────────────────────────────────
• Conserve l’historique USER / ASSISTANT.
• Utilise exactement le même schéma d’entraînement (« [INST] … [/INST] … »).
• Fonctionne en 8-bit sur GPU, full-precision sur CPU.
"""

import argparse, os, torch, collections
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


# ╭─ arguments CLI ───────────────────────────────────────────────────────────╮
def parse_args():
    p = argparse.ArgumentParser("Interactive medical chat with LoRA model")
    p.add_argument("--base_model",   required=True)
    p.add_argument("--lora_dir",     required=True)
    p.add_argument("--max_len",      type=int, default=1024)        # contexte max
    p.add_argument("--max_gen",      type=int, default=256)         # tokens générés
    p.add_argument("--temperature",  type=float, default=0.8)
    p.add_argument("--top_k",        type=int,   default=40)
    return p.parse_args()


# ╭─ helpers ─────────────────────────────────────────────────────────────────╮
def load_model(base_id, lora_dir, token):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    base = AutoModelForCausalLM.from_pretrained(
        base_id,
        device_map="auto" if dev == "cuda" else None,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True) if dev == "cuda" else None,
        torch_dtype=torch.float16 if dev == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
        use_auth_token=token,
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, lora_dir, is_trainable=False,
                                      device_map="auto" if dev == "cuda" else None,
                                      torch_dtype=base.dtype,
                                      use_auth_token=token, trust_remote_code=True)
    model.eval();  model.config.use_cache = True
    tok = AutoTokenizer.from_pretrained(base_id, use_fast=True,
                                        use_auth_token=token, trust_remote_code=True)
    tok.padding_side = "right"
    tok.pad_token_id = tok.eos_token_id if tok.pad_token_id is None else tok.pad_token_id
    return tok, model


def build_history_prompt(history):
    """
    history = deque([(role,msg), ...])  where role ∈ {"user","assistant"}
    Assemble <s>[INST] user1 [/INST] assistant1 </s><s>[INST] user2 …
    The last line is an unfinished [INST] ... waiting for assistant.
    """
    out = []
    for role, msg in history:
        if role == "user":
            out.append(f"<s>[INST] {msg.strip()} [/INST] ")
        else:  # assistant
            out.append(f"{msg.strip()} </s>")
    return "".join(out)


# ╭─ main loop ───────────────────────────────────────────────────────────────╮
def main():
    args = parse_args()
    hf_token = os.getenv("HF_HUB_TOKEN") or os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("Export HF_HUB_TOKEN first.")

    tok, model = load_model(args.base_model, args.lora_dir, hf_token)
    history = collections.deque()  # (role,msg)

    print("\nChat started – empty line to quit.")
    while True:
        user_msg = input("\nUSER >>> ").strip()
        if not user_msg:
            break

        # 1) add user message to history
        history.append(("user", user_msg))

        # 2) build prompt & truncate if > max_len
        prompt = build_history_prompt(history)
        ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)

        # truncate oldest exchanges if too long
        while ids.shape[-1] > args.max_len and len(history) > 1:
            history.popleft()           # retire le plus vieux
            prompt = build_history_prompt(history)
            ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)

        prompt_len = ids.shape[-1]

        # 3) generate
        with torch.no_grad():
            gen_ids = model.generate(
                ids,
                max_new_tokens=args.max_gen,
                temperature=args.temperature,
                top_k=args.top_k,
                do_sample=True,
                pad_token_id=tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
            )

            

        answer = tok.decode(gen_ids[0, prompt_len:], skip_special_tokens=True).strip()
        print("\nASSISTANT >>>", answer)

        # 4) add assistant reply to history
        history.append(("assistant", answer))


if __name__ == "__main__":
    main()


