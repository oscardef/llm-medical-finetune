# Contributions by each of us

## Oscar

* Research/Planning
  * Did research on pretrained models to use (see `research.md`) and decided on Mistral7B (+ TinyMistral for testing)
  * tested pretrained models to see how it worked to download/work with/modify them (see `run_pretrained.py`)
  * Read + made notes on RCP Cluster Documentation (See `README.md`)
* Setup
  * Created this GitHub repository
  * Did extensive research + testing with RCP to figure out how to configure it, set it up, use it (from everything needed to be done locally; Dockerfile, etc. to NAS1 and NAS3 storage to runai + kubectl to submitting jobs + viewing their output, etc.)
    * Created `README.md` with extensive instructions on how to do this all step-by-step
  * Created RCP Repository `llm-medical-finetune` where deployed docker images get sent to (see `https://registry.rcp.epfl.ch/harbor/projects/648/repositories/med-llm`)
  * Fifure out NAS storage to store data used for finetuning (can be found on `jumphost.rcp.epfl.ch` in `/mnt/sdsc-ge/scratch/`)
* Finetuning
  * Rewrote `finetune_lora.py` so that it properly trains chosen model localy on mac M1 as well as on RCP cluster when deployed
    * Tested this and it works with 1% of the data and 1 epoch with TinyMistral locally
  * Created run_lora.py which will run the finetuned model so it can be tested
    * Was able to rune the finetuned model with this
  * Created Docker image (and figured out necessary python versions, etc.) and deployed to RCP Repository `llm-medical-finetune`
  * Pushed data to NAS storage
  * 




## Ali

Below we highlight contributions from **Ali** (responsible for finetuning, experimentation, and model optimization in this project):

* **Research & Design Choices**
  * Explored different finetuning approaches (LoRA, QLoRA, AdaLoRA, etc.) and decided to use **LoRA with 8 bit models** as the best compromise between memory usage and training efficiency.
  * Tested Flash Attention 2 versus standard attention; after stability issues on A100 GPUs, recommended using standard attention instead.
  * Helped define the final LoRA configuration (`r=8`, `alpha=16`, `dropout=0.05`) used across all large scale training runs.

* **Data Cleaning & Formatting**
  * Cleaned and reformatted raw JSON files (dialogues and MCQs) into the structure needed for instruction based finetuning.
  * Applied standard preprocessing (removing bad formatting, fixing whitespaces, etc ...) and converted examples into the format `<s>[INST] instruction [/INST] response </s>`, masking non target parts for proper loss computation.
  * Added a data sampling option to quickly test small training subsets before launching longer runs.

* **Training Code & Experiments**
  * Wrote and debugged `finetune_lora2.py`, a script that prepares the dataset, loads quantized models, applies LoRA, and launches training using Hugging Face's Trainer API.
  * Carried out **extensive hyperparameter tuning**, testing many combinations of learning rate, batch size, gradient accumulation steps, and training percentages to find stable and effective settings.
  * Improved the training pipeline by identifying and fixing bugs in masking, tokenization, save/load logic, and compatibility with quantized models.

* **Multi GPU Training Support**
  * Contributed to adapting the training script for multi GPU runs with DeepSpeed (in `finetune_lora_multigpu.py`), including gradient checkpointing and automatic memory scaling.
  * Helped debug GPU specific errors and ensured compatibility across different environments.
  * Created alternative versions to bypass Flash Attention issues.

* **Inference & Chat Interface**
  * Developed and tested `run_lora_3.py`, a script for interactive chats with the finetuned model.
  * Implemented context/history handling and prompt formatting so users can have a natural conversation with the model.
  * Verified correct loading of LoRA adapters and ensured smooth generation on 8 bit models using GPU.

* **General Debugging & Optimization**
  * Fixed several bugs related to quantization, padding, training stability, and memory usage.
  * Optimized data loading, logging, and model saving to make the pipeline more reliable and reproducible.
