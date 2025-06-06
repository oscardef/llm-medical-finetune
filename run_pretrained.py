import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",
        default="Locutusque/TinyMistral-248M",
        help="HF model ID (TinyMistral-248M or mistralai/Mistral-7B-v0.1)")
    p.add_argument("--prompt",
        default="Hello, how are you?",
        help="Text prompt")
    p.add_argument("--max_new_tokens", type=int, default=64)
    args = p.parse_args()

    # load
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.eval()

    # inference
    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            top_k=50,
            temperature=0.8,
        )
    print("\n" + tokenizer.decode(out[0], skip_special_tokens=True))

if __name__=="__main__":
    main()
