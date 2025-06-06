import torch
from datasets import Dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)

# 1) Five small conversation examples
examples = [
    {
    "instruction": "You are a medical doctor expert. Here is a conversation with a patient—give them a solution or an answer:",
    "input": "Hi.My names Ahdieh.I m from a small city in Iran.My father had a heart attack on sunday as doctor said cpr condition.after 40 minuts he came back to life. now he isnot conscious.he is in ICU.his doctor said his conscious rate is 5.what do you think about his condition?If it is needed I can send you his cardiograf and blood test and the drugs they are using for him in ICU.Kindly tell me can we carry him to a better hospital in Tehran(It takes 4 hours to reach there)?",
    "output": "Hello Addie! Welcome and thank you for asking on Chat Doctor! I passed carefully through your question and would explain that your father clinical situation could be related to prolonged brain hypoxia, due to the cardiac arrest. The time (40 minutes) needed for his resuscitation is long and may have caused irreversible damage to his brain, due to low blood flow (also called hypoxic encephalopathy). For this reason, I recommend consulting with a neurologist and performing a brain MRI and EEG to examine his brain function. Hope you will find this answer helpful! Kind regards,"
  },
  {
    "instruction": "You are a medical doctor expert. Here is a conversation with a patient—give them a solution or an answer:",
    "input": "Hello and thank you for your time and assistance, I have severe fibromyalgia, migraines, adult onset diabetes, cervical radiculopathy, c4,5,6,7 moderate to severe bi-lateral foraminal encroachment , post concussion disorder, thyroiditis. Would you have ordered an emg test for any reason considering the pain that it could possibly cause a continual cycle of pain?",
    "output": "Hi, You need to focus more on doing exercises and allow the muscle fibers to regain the normalcy. Fibromyalgia is a common term used nowadays by most as they cannot do much with the pain medication. But fibromyalgia is a different terminology to ad Chat Doctor.  The pain you have is due to the cervical disc degenerative changes which has led to chronic inflammation in the trapezius muscle. Doing regular exercise and getting correction of the muscle imbalance due to weakness and improving the stability of the para spinal muscles should help further. Hope I have answered your query. Let me know if I can assist you further. Jay In"
  },
  {
    "instruction": "You are a medical doctor expert. Here is a conversation with a patient—give them a solution or an answer:",
    "input": "Yes! I had double knee replacements on Feb. 2008, At the time the Dr. told my husband my left knee was too small , and he had a hard time with it. Then I also was found to have a blood clot in that calf. That replacement has never worked right, THEN, 8 weeks after the surgery, I suddenly began to have a very itchy, red bumpy rash from the knee down the front & inside of that leg. and all these years I keep being told that I have dry skin even though I continually apply several types of creams, aquaphor, even steroid creams subscribed for me. Then in Oct. 2012 it began same way on my right leg; below the knee. I have asked & asked if I could be allergic to the glue & commented that I was not asked about Nickel, ( I am allergic to it), but was told that there was no Nickel. I am so frustrated, & with constantly using the steroid cream, my skin is very thin, red & all my blood vessels now are prominent! It is an ever on going discomfort. Do you think this could be from the glue or a bit of Nickel in the Prothesis?",
    "output": "Hi Hope this message finds you in good health. I have gone through your complaints and understand your concern. You are right, the redness may be due to allergic reaction to some type of chemicals used in the manufacturing of implants, A possibility of a low level infection should also be kept in mind. Get a X-ray done to see the wear and tear of implant material. Nothing to worry about. I hope your question has been answered. If you have any follow-up queries, feel free to consult me anytime. Thanks, Take care, God bless."
  },
  {
    "instruction": "You are a medical doctor expert. Here is a conversation with a patient—give them a solution or an answer:",
    "input": "I m extremely itchy ALL OVER the place? I m extremely itchy all over the place! It constantly fels like there are bugs crawling all over me and biting me in random places. It almost feels like the kind of itch you get from an allergic reaction . Except i feel it constantly in random places all over my body. I m a very hairy person, unfortunately, so I don t think that helps. One night while I was asleep i woke up from my chest itching so bad and I just kept scratching and digging into cause it wouldnt stop and it hurt really bad and i had to put hydrocortisone on it. I live in a house on the lake and the temperatures right now are about a constant 104 every day. Don t know if that helps anyone. But please help me. I don t know what to do and I m freaking out because it s starting to itch ALL OVER and WONT STOP!!",
    "output": "YOUR DURATION OF DISEASE IS NOT KNOWN FROM HISTORY IT CAN BE FROM MANY CAUSES OILS SHAMPOO BODY APPLICATION MATERIAL DYES RECOOKING OILS ANY MED TAKING FOR OTHER DISEASE LIKE PCM IBUPROFEN VITAMINS AND DIET SUPPLEMENTARY CLOTHING TO LIKE CHANCES OF DUST MITE ALLERGY NEED TO FIND STILL THEN YOU CAN SE BASTING 10 MG NIGHTS YP CPM OR TAB CPM AT NIGHTLY CALAMINE LOTION LOCALLY OVER SKIN AS MANY TIMES YOU NEED CAN CONTINUE LONG"
  },
  {
    "instruction": "You are a medical doctor expert. Here is a conversation with a patient—give them a solution or an answer:",
    "input": "Hi Doctor, I need your help, im really worried about my mom, she has a brown discoloration in her gums, like under her front bottom teeth, and she says she feels like it stings, her bone stings not her gum, she has had problems with her teeth before, she has no insurance so she cant go to the dentist, but i really need your advice and help to calm her down, Please help!",
    "output": "Hello there, Thanks for writing your query, Brownish discoloration of the gums is caused may be due to gingival inflammation or gingival abscess or periodontal access. It is caused due poor oral hygiene, improper brushing and flossing that leads to accumulation of bacterial plaque and calculus deep in the gum pockets and root surfaces of the teeth.  Gingivitis presents with symptoms of redness, bleeding gums, swelling, pain, recession and sensitivity. It needs to be treated by thorough professional deep scaling of gums and root planing of the teeth to remove all the irritants, this will help in resolving the symptoms followed by a course of antibiotic and analgesics.  I would suggest you to go for symptomatic treatment at home like gargling with lukewarm saline water or antiseptic mouthwash like chlorhexidine or Betadine.  You can apply topical oral antiseptics like denote over the affected which helps in giving soothing effect to the gums. I hope this helps, take care."
  },
]

# 2) Build a HuggingFace Dataset from these 5 examples
raw = {
    "instruction": [e["instruction"] for e in examples],
    "input":       [e["input"] for e in examples],
    "output":      [e["output"] for e in examples],
}
dataset = Dataset.from_dict(raw)

# 3) Load TinyMistral tokenizer
base_model_id = "Locutusque/TinyMistral-248M"
tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
tokenizer.padding_side = "right"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# 4) Tokenization + masking function
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

# 5) Load the full TinyMistral model (no LoRA)
config = AutoConfig.from_pretrained(base_model_id)
config.model_parallel = False
setattr(config, "tensor_parallel", False)

if torch.cuda.is_available():
    # On CUDA, load in fp16
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        config=config,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    use_fp16 = True
else:
    # On CPU (or MPS), stay in fp32
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        config=config,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    use_fp16 = False

# 6) TrainingArguments with saving turned off
training_args = TrainingArguments(
    output_dir="overfit_full_tiny_results",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=15,
    learning_rate=2e-4,
    save_strategy="no", # disable all checkpoints
    logging_steps=2,
    fp16=use_fp16,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
)

print(">>> Starting overfit on full TinyMistral (no LoRA)…")
trainer.train()
print(">>> Overfit training complete.\n")

# 7) Generate from each prompt and compare
model.eval()
for idx, ex in enumerate(examples):
    prompt_only = f"<s>[INST] {ex['instruction']} {ex['input']} [/INST] "
    # NOTE: no padding="max_length" here
    inputs = tokenizer(
        prompt_only,
        return_tensors="pt",
        truncation=True,
        # no padding argument at all (so it only returns the minimal prompt length)
    ).to(model.device)

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
        )
    generated = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    # now strip off the prompt prefix
    reply = generated.replace(prompt_only, "").strip()

    print(f"\n--- EXAMPLE {idx+1} ---")
    print("PROMPT:    ", ex["input"])
    print("EXPECTED:  ", ex["output"])
    print("GENERATED: ", reply)
    print("-------------------------")