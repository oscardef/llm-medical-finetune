# TinyMistral Runner

A minimal example to run the TinyMistral-248M model for quick inference on MacBook or any machine.

## Requirements

* Python 3.8+
* pip-installed packages below

## Installation

```bash
pip install torch transformers accelerate
```

## Files

* **run\_pretrained.py**: Script that downloads and runs a specified HF model.

## Usage

1. **Run TinyMistral-248M** (fits on M1/M2 with 16 GB RAM)

   ```bash
   python run_pretrained.py \
     --model Locutusque/TinyMistral-248M \
     --prompt "Explain diabetes like I'm five years old." \
     --max_new_tokens 100
   ```

2. **Run full Mistral-7B** (GPU recommended)

   ```bash
   python run_pretrained.py \
     --model mistralai/Mistral-7B-v0.1 \
     --prompt "Patient: I have a headache and fever. Doctor:" \
     --max_new_tokens 128
   ```

## Options

* `--model`: HF model identifier (default: `Locutusque/TinyMistral-248M`)
* `--prompt`: Text prompt to generate from
* `--max_new_tokens`: Number of tokens to generate (default: 64)

---