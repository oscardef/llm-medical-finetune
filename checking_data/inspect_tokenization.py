# inspect_tokenization.py

from transformers import AutoTokenizer

# 1) Load tokenizer and force "right" padding
base_model = "Locutusque/TinyMistral-248M"
tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
tokenizer.padding_side = "right"

# 2) One example from your conversation JSON
instruction = (
    "You are a medical doctor expert. Here is a conversation with a patient—"
    "give them a solution or an answer:"
)
input_text = (
    "Hi. My names Ahdieh. I’m from a small city in Iran. My father had a heart attack on Sunday as doctor said CPR condition. "
    "After 40 minutes he came back to life. Now he is not conscious. He is in ICU. His doctor said his conscious rate is 5. "
    "What do you think about his condition? If it is needed I can send you his cardiogram and blood tests and the drugs they are using for him in ICU. "
    "Kindly tell me can we carry him to a better hospital in Tehran (It takes 4 hours to reach there)?"
)
output_text = (
    "Hello Addie! Welcome and thank you for asking on Chat Doctor! I passed carefully through your question and would explain "
    "that your father’s clinical situation could be related to prolonged brain hypoxia, due to the cardiac arrest. The time (40 minutes) "
    "needed for his resuscitation is long and may have caused irreversible damage to his brain, due to low blood flow (also called hypoxic encephalopathy). "
    "For this reason, I recommend consulting with a neurologist and performing a brain MRI and EEG to examine his brain function. Hope you will find this answer helpful! "
    "Kind regards,"
)

max_len = 1024

# 3) Build the “prompt” and the “answer” exactly as in training
full_prompt = f"<s>[INST] {instruction} {input_text} [/INST] "
full_answer = f"{output_text} </s>"
combined    = full_prompt + full_answer

# 4) Tokenize the entire sequence with right‐side padding
enc_full = tokenizer(
    combined,
    truncation=True,
    max_length=max_len,
    padding="max_length",
    return_attention_mask=True,
)
input_ids      = enc_full["input_ids"]
attention_mask = enc_full["attention_mask"]

# 5) Tokenize only the prompt to find prompt_len
enc_prompt = tokenizer(
    full_prompt,
    truncation=True,
    max_length=max_len,
    padding="max_length",
    return_attention_mask=False,
)["input_ids"]
prompt_len = sum(1 for tok_id in enc_prompt if tok_id != tokenizer.pad_token_id)

print("=== SAMPLE TOKENIZATION (RIGHT PADDING) ===\n")
print("Full prompt+answer tokens (first 50):", input_ids[:50])
print("Prompt length (non‐pad tokens):", prompt_len)
print("Decoded prompt (first prompt_len tokens):")
print("  ", tokenizer.decode(input_ids[:prompt_len]))
print("Decoded answer (next 20 tokens):")
print("  ", tokenizer.decode(input_ids[prompt_len : prompt_len + 20]))
print("Attention mask (first 50):", attention_mask[:50])

# 6) Show masking of labels
labels = input_ids.copy()
for i in range(prompt_len):
    labels[i] = -100
print("\nLabels (first 50):", labels[:50])
