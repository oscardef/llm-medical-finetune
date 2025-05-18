# Research: Base Model Selection & Evaluation

## Base Model Options

### Mistral AI “Mixtral” 8×7B

* **Performance:** Outperforms LLaMA‑2 70B on ARC Challenge, HellaSwag & MMLU (Fr, De, Es, It) ([mistral.ai](https://mistral.ai/news/mistral-large), [arxiv.org](https://arxiv.org/pdf/2401.04088))
* **Speed:** ≈6× faster inference; supports 32K token context
* **Pros:** State‑of‑the‑art multilingual reasoning; moderate per‑token cost
* **Cons:** Sparse MoE layers complicate fine‑tuning; requires expert routing; 46B params total

### Mistral AI Dense 7B

* **Performance:** Beats LLaMA‑2 13B on all evaluated benchmarks (commonsense, STEM, code) ([arxiv.org](https://arxiv.org/abs/2310.06825))
* **Speed & Memory:** Low footprint, fast on single GPUs
* **Pros:** Simple dense architecture; excellent speed/accuracy trade‑off
* **Cons:** Slightly lower reasoning capacity vs. Mixtral

### Meta LLaMA‑2 / LLaMA‑3 (7B–70B)

* **Performance:** LLaMA‑2 70B scores \~68.9% on MMLU ([huggingface.co](https://huggingface.co/meta-llama/Llama-2-70b))
* **Pros:** Mature HF ecosystem (Transformers, LoRA, DeepSpeed); well‑documented
* **Cons:** Large variants (34B–70B) need multi‑GPU or offload; license restrictions

### Google Gemma 7B

* **Performance:** \~64.3% on MMLU (HELM) ([crfm.stanford.edu](https://crfm.stanford.edu/2024/05/01/helm-mmlu.html))
* **Speed & Memory:** Fits single 24 GB GPU; very fast inference
* **Pros:** Top performance for 7B; open weights; instruction‑tunable
* **Cons:** No larger variants yet; may lag on complex tasks

### Alibaba Qwen 7B

* **Performance:** \~56.7% on MMLU (5‑shot) ([paperswithcode.com](https://paperswithcode.com/sota/multi-task-language-understanding-on-mmlu))
* **Pros:** Multilingual + code‑focused; rich community support
* **Cons:** Chinese‑centric pretraining can affect English nuance; moderate inference cost

---

## Evaluation Strategy

* **Medical QA Benchmarks**

  * Use HF datasets: `openlifescienceai/medqa` ([huggingface.co](https://huggingface.co/datasets/openlifescienceai/medqa)), `openlifescienceai/medmcqa` ([huggingface.co](https://huggingface.co/datasets/openlifescienceai/medmcqa)), `qiaojin/PubMedQA` ([github.com](https://github.com/huggingface/open-r1/issues/31))
  * Follow the Open Medical‑LLM Leaderboard protocol (accuracy on MedQA, MedMCQA, PubMedQA, MMLU subsets) ([huggingface.co](https://huggingface.co/blog/leaderboard-medicalllm))

* **Conversational Quality**

  * Human or LLM‑based scoring of doctor–patient chats (coherence, empathy, factuality)
  * Side‑by‑side prompts to your model vs. MediTron; judge via GPT‑4 or clinician feedback

* **Baseline Models**

  * Load MediTron weights: `epfl-llm/meditron-7b` ([huggingface.co](https://huggingface.co/epfl-llm/meditron-7b)) and `epfl-llm/meditron-70b` ([huggingface.co](https://huggingface.co/epfl-llm))
  * Run identical QA & chat evaluations for direct comparison

---

## Setup & Fine‑Tuning Steps

1. **Select & Load Base Model**

   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer

   model = AutoModelForCausalLM.from_pretrained(
       "mistral-inference/mixtral-8x7b", trust_remote_code=True
   )
   tokenizer = AutoTokenizer.from_pretrained("mistral-inference/mixtral-8x7b")
   ```

2. **Prepare Data**

   * **Dialogues:** `<patient_msg> → <doctor_response>`
   * **MCQs:**

     ```
     Question: ...
     Options: A) ... B) ... C) ... D) ...
     Answer:
     ```

3. **Fine‑Tune**

   * Use Raschka’s PyTorch loop or HF `Trainer` + LoRA/PEFT
   * Enable gradient checkpointing & mixed precision (fp16)

4. **Evaluate**

   * Compute accuracy on MedQA/MedMCQA/PubMedQA etc.
   * Generate dialogue replies; compare vs. MediTron with human/automated judges

5. **Iterate & Optimize**

   * Address hallucinations via retrieval‑augmented data or data filtering
   * Optionally add small RLHF or extra instruction‑tuning



