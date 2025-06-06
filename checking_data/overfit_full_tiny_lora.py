#!/usr/bin/env python3
# overfit_full_tiny_lora_fixed.py

import torch
from datasets import Dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, PeftModel

# ─── 1) Five toy conversation examples ───────────────────────────────────────
examples = [
    {
      "instruction": "You are a medical doctor expert. Here is a conversation with a patient—give them a solution or an answer:",
      "input": (
          "Hi.My names Ahdieh.I m from a small city in Iran.My father had a heart "
          "attack on Sunday as doctor said cpr condition.after 40 minuts he came back "
          "to life. now he is not conscious. He is in ICU. His doctor said his conscious "
          "rate is 5. What do you think about his condition? If it is needed I can send "
          "you his cardiogram and blood tests and the drugs they are using for him in ICU. "
          "Kindly tell me can we carry him to a better hospital in Tehran (It takes 4 hours "
          "to reach there)?"
      ),
      "output": (
          "Hello Addie! Welcome and thank you for asking on Chat Doctor! I passed carefully "
          "through your question and would explain that your father's clinical situation "
          "could be related to prolonged brain hypoxia due to the cardiac arrest. The time "
          "(40 minutes) needed for his resuscitation is long and may have caused irreversible "
          "damage to his brain (hypoxic encephalopathy). For this reason, I recommend "
          "consulting with a neurologist and performing a brain MRI and EEG to examine his "
          "brain function. Hope you will find this answer helpful! Kind regards."
      ),
    },
    {
      "instruction": "You are a medical doctor expert. Here is a conversation with a patient—give them a solution or an answer:",
      "input": (
          "Hello and thank you for your time and assistance, I have severe fibromyalgia, "
          "migraines, adult-onset diabetes, cervical radiculopathy, C4-7 moderate to severe "
          "bi-lateral foraminal encroachment, post-concussion disorder, thyroiditis. Would you "
          "have ordered an EMG test for any reason considering the pain that it could possibly "
          "cause a continual cycle of pain?"
      ),
      "output": (
          "Hi, you need to focus more on doing exercises and allow the muscle fibers to regain "
          "normalcy. Fibromyalgia is a common term used nowadays by many as they cannot do much "
          "with pain medication alone. The pain you have is due to cervical disc degenerative "
          "changes, which have led to chronic inflammation in the trapezius muscle. Regular "
          "exercise to correct muscle imbalance and improve paraspinal stability should help. "
          "Hope I have answered your query. Let me know if I can assist you further. Jay In."
      ),
    },
    {
      "instruction": "You are a medical doctor expert. Here is a conversation with a patient—give them a solution or an answer:",
      "input": (
          "Yes! I had double knee replacements on Feb. 2008. At the time, the Dr. told my "
          "husband my left knee was too small and he had a hard time with it. Then I also was "
          "found to have a blood clot in that calf. That knee replacement has never worked right. "
          "Then, 8 weeks after the surgery, I suddenly began to have a very itchy, red bump-y rash "
          "from the knee down the front & inside of that leg. All these years I've been told I "
          "have dry skin even though I kept applying various creams. Then in Oct. 2012 it began "
          "the same way on my right leg; below the knee. I have asked if I could be allergic to "
          "the glue and mentioned that I am allergic to nickel, but was told there was no nickel. "
          "I am so frustrated. With constantly using steroid cream, my skin is very thin and red, "
          "and all my blood vessels now are very prominent! It is an ongoing discomfort. Do you "
          "think this could be from the glue or a bit of nickel in the prosthesis?"
      ),
      "output": (
          "Hi, hope this message finds you in good health. I have reviewed your complaints and "
          "understand your concern. You may indeed be having a low-grade allergic reaction to "
          "some chemical in the implant. A possibility of a low-level infection should also be "
          "considered. I recommend getting an X-ray to inspect the prosthetic wear and tear. If "
          "you suspect allergy, patch testing or consulting an allergist may help. In the meantime, "
          "gentle moisturizing and avoiding irritants is wise. If pain or rash persists, consult "
          "an orthopedic specialist. Thanks, take care, and God bless."
      ),
    },
    {
      "instruction": "You are a medical doctor expert. Here is a conversation with a patient—give them a solution or an answer:",
      "input": (
          "I’m extremely itchy ALL OVER! It constantly feels like bugs crawling and biting in random "
          "places, almost like an allergic reaction, but it’s all over my body. I’m very hairy, which "
          "doesn’t help. One night I woke up with my chest itching so badly I kept scratching until it hurt. "
          "I live on a lake, and it’s been about 104°F every day. Please help—I don’t know what to do, and I’m "
          "freaking out because it won’t stop!"
      ),
      "output": (
          "Your itching could be due to a variety of causes: heat rash, insect bites, allergic contact "
          "dermatitis (from clothing or lotions), or even scabies. First, try a cool bath with oatmeal soak "
          "or apply calamine lotion. Wear loose, breathable cotton clothing. If insect bites are suspected, "
          "use an insect repellent. A topical antihistamine cream or a short course of oral antihistamines "
          "can also help. If nothing improves, seek evaluation by a dermatologist or allergist. Take care "
          "and stay cool."
      ),
    },
    {
      "instruction": "You are a medical doctor expert. Here is a conversation with a patient—give them a solution or an answer:",
      "input": (
          "Hi Doctor, I need your help. I’m really worried about my mom—she has a brown discoloration in her gums "
          "under her front bottom teeth, and she says it stings (the bone, not the gum). She’s had dental issues "
          "before but has no insurance so she can’t go to a dentist. I really need advice to calm her down. Please help!"
      ),
      "output": (
          "Hello there, thanks for writing. Brownish discoloration of the gums can be caused by gingival inflammation "
          "(gingivitis) or a gingival abscess. Poor oral hygiene leads to bacterial plaque and calculus building up "
          "in the gum pockets, which causes redness, bleeding, swelling, pain, recession, and sensitivity. Treatment "
          "requires a thorough dental deep-clean (scaling and root planing) plus a course of antibiotics and analgesics. "
          "At home, she can rinse with lukewarm saline or an antiseptic mouthwash like chlorhexidine. Topical antiseptic "
          "gels can also help soothe. Encourage her to see a dentist or periodontist as soon as possible. Hope this helps—take care."
      ),
    },
]

# ─── 2) Build a HF Dataset from those 5 examples ──────────────────────────────
raw = {
    "instruction": [e["instruction"] for e in examples],
    "input":       [e["input"]       for e in examples],
    "output":      [e["output"]      for e in examples],
}
dataset = Dataset.from_dict(raw)

# ─── 3) Load TinyMistral tokenizer ─────────────────────────────────────────
base_model_id = "Locutusque/TinyMistral-248M"
tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
tokenizer.padding_side = "right"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# ─── 4) Tokenization + label masking ───────────────────────────────────────
def tokenize_and_mask(ex):
    prompt = f"<s>[INST] {ex['instruction']} {ex['input']} [/INST] "
    answer = f"{ex['output']} </s>"

    full_enc = tokenizer(
        prompt + answer,
        truncation=True,
        max_length=1024,
        padding="max_length",
        return_attention_mask=True,
    )
    prompt_enc = tokenizer(
        prompt,
        truncation=True,
        max_length=1024,
        padding="max_length",
        return_attention_mask=False,
    )["input_ids"]
    prompt_len = sum(1 for tok in prompt_enc if tok != tokenizer.pad_token_id)

    labels = full_enc["input_ids"].copy()
    for i in range(prompt_len):
        labels[i] = -100

    return {
        "input_ids":      full_enc["input_ids"],
        "attention_mask": full_enc["attention_mask"],
        "labels":         labels,
    }

train_ds = dataset.map(tokenize_and_mask, remove_columns=["instruction", "input", "output"])

# ─── 5) Load TinyMistral + attach LoRA ──────────────────────────────────────
config = AutoConfig.from_pretrained(base_model_id)
config.model_parallel = False
setattr(config, "tensor_parallel", False)

if torch.cuda.is_available():
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        config=config,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
else:
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        config=config,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )

# Print LoRA‐capable modules to verify
print("\n>>> AVAILABLE modules containing ‘self_attn.q_proj’, etc.:")
for name, module in base_model.named_modules():
    if name.endswith("q_proj") or name.endswith("k_proj") or name.endswith("v_proj") or name.endswith("o_proj"):
        print("  •", name)

lora_cfg = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
        "model.layers.0.self_attn.o_proj",
        "model.layers.1.self_attn.q_proj",
        "model.layers.1.self_attn.k_proj",
        "model.layers.1.self_attn.v_proj",
        "model.layers.1.self_attn.o_proj",
        "model.layers.2.self_attn.q_proj",
        "model.layers.2.self_attn.k_proj",
        "model.layers.2.self_attn.v_proj",
        "model.layers.2.self_attn.o_proj",
        "model.layers.3.self_attn.q_proj",
        "model.layers.3.self_attn.k_proj",
        "model.layers.3.self_attn.v_proj",
        "model.layers.3.self_attn.o_proj",
        "model.layers.4.self_attn.q_proj",
        "model.layers.4.self_attn.k_proj",
        "model.layers.4.self_attn.v_proj",
        "model.layers.4.self_attn.o_proj",
        "model.layers.5.self_attn.q_proj",
        "model.layers.5.self_attn.k_proj",
        "model.layers.5.self_attn.v_proj",
        "model.layers.5.self_attn.o_proj",
        "model.layers.6.self_attn.q_proj",
        "model.layers.6.self_attn.k_proj",
        "model.layers.6.self_attn.v_proj",
        "model.layers.6.self_attn.o_proj",
        "model.layers.7.self_attn.q_proj",
        "model.layers.7.self_attn.k_proj",
        "model.layers.7.self_attn.v_proj",
        "model.layers.7.self_attn.o_proj",
        "model.layers.8.self_attn.q_proj",
        "model.layers.8.self_attn.k_proj",
        "model.layers.8.self_attn.v_proj",
        "model.layers.8.self_attn.o_proj",
        "model.layers.9.self_attn.q_proj",
        "model.layers.9.self_attn.k_proj",
        "model.layers.9.self_attn.v_proj",
        "model.layers.9.self_attn.o_proj",
        "model.layers.10.self_attn.q_proj",
        "model.layers.10.self_attn.k_proj",
        "model.layers.10.self_attn.v_proj",
        "model.layers.10.self_attn.o_proj",
        "model.layers.11.self_attn.q_proj",
        "model.layers.11.self_attn.k_proj",
        "model.layers.11.self_attn.v_proj",
        "model.layers.11.self_attn.o_proj",
    ],
)

model = get_peft_model(base_model, lora_cfg)

if torch.cuda.is_available():
    # k‐bit preparation for GPU (if needed)
    model = prepare_model_for_kbit_training(model)

# ─── 6) TrainingArguments (higher LR + fewer epochs) ────────────────────────
training_args = TrainingArguments(
    output_dir="overfit_full_tiny_lora_results",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=100,       # halved from 200
    learning_rate=2e-4,
    save_strategy="no",
    logging_steps=2,
    fp16=torch.cuda.is_available(),
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
)

print(">>> Starting LoRA‐only overfit on TinyMistral (five examples)…")
trainer.train()
print(">>> LoRA overfit complete.\n")

# ─── 7) Save adapter and reload for inference ───────────────────────────────
model.save_pretrained("overfit_full_tiny_lora_results/adapter_model")

print(">>> Reloading base TinyMistral + LoRA adapter for inference…")
base_model_for_inference = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    config=config,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)
model = PeftModel.from_pretrained(
    base_model_for_inference,
    "overfit_full_tiny_lora_results/adapter_model",
    is_trainable=False,
)
model.eval()

# ─── 8) Generate from each prompt and compare ───────────────────────────────
for idx, ex in enumerate(examples):
    # Rebuild the prompt exactly as during training
    prompt_only = f"<s>[INST] {ex['instruction']} {ex['input']} [/INST] "
    # Tokenize the prompt (with explicit max_length to suppress warnings)
    inputs = tokenizer(
        prompt_only,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(model.device)

    # Decode the tokenized prompt (skip_special_tokens=True) so we know what to strip
    prompt_with_specials = tokenizer.decode(
        inputs["input_ids"][0],
        skip_special_tokens=True
    )

    # Generate output
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(out_ids[0], skip_special_tokens=True)

    # Remove the prompt_text portion from the generated string
    reply = generated.replace(prompt_with_specials, "").strip()

    print(f"\n--- EXAMPLE {idx+1} (LoRA) ---")
    print("PROMPT:    ", ex["input"])
    print("EXPECTED:  ", ex["output"])
    print("GENERATED: ", reply)
    print("-------------------------")
