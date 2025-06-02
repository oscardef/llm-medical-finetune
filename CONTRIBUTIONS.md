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