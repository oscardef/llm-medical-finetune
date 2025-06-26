# LLM Medical Finetune (Mistral-7B & TinyMistral-248M)

This repository contains code and documentation for fine-tuning large language models (LLMs) on medical dialogue and question-answering data. The project’s goal is to adapt state-of-the-art models (like **Mistral-7B** and a smaller **TinyMistral-248M**) to the medical domain by training on two types of data: (1) **medical instruction-based conversations** and (2) **medical multiple-choice questions (MCQs)**. We leverage **LoRA** (Low-Rank Adaptation) to fine-tune these models efficiently on limited hardware, enabling training on a single GPU or even on a Mac M1 by using 8-bit quantization when available. The end result is a specialized medical assistant model that can engage in doctor-patient conversations and answer medical questions with domain-specific knowledge.

**Key features of this project include:**

* A **LoRA fine-tuning script** that combines medical dialogue and Q\&A datasets to train the model, with all hyperparameters controllable via command-line (making it easy to run on different environments). Fine-tuning is done by injecting trainable LoRA adapters into the base model’s layers instead of updating all model weights.
* **Inference scripts** for both the fine-tuned model (with LoRA adapters) and the original pretrained models, allowing you to interactively prompt the model or evaluate it on given inputs.
* **Containerization and cluster orchestration** for running training and inference on EPFL’s RCP cloud infrastructure using Docker and Run\:AI. Detailed instructions are provided for building the Docker image, pushing it to the registry, configuring the RCP Kubernetes cluster access, uploading data to the cluster storage, and submitting jobs to run training or inference on GPUs.
* A tested workflow on both **local machines** (e.g. a MacBook with Apple M1, using CPU/MPS) and the **EPFL RCP GPU cluster**, ensuring new users can reproduce results in either environment.
* **Hugging Face Hub integration:** The code supports loading models from Hugging Face (including gated models like Mistral-7B) by using an authentication token. Please ensure you set your `HF_HUB_TOKEN` in the environment when needed (more on this below).

## Repository Structure and Key Files

This section explains the main scripts and utilities in the repository, and how they contribute to the fine-tuning workflow:

* **`finetune_lora.py` – LoRA Fine-Tuning Script:** This is the core training script to fine-tune a causal language model on the medical datasets with LoRA. It expects two JSON files as input: one with **conversation data** (`--conv_json`) and one with **MCQ (multiple-choice) data** (`--qcm_json`), each containing a list of records with `{"instruction": ..., "input": ..., "output": ...}` fields. The script loads a **base model** from Hugging Face (`--base_model`, e.g. `mistralai/Mistral-7B-v0.1` or `Locutusque/TinyMistral-248M`), and prepares it for 8-bit training if a GPU is available. Specifically, it uses **BitsAndBytes** to load the model in 8-bit mode when on CUDA, which greatly reduces memory usage. A Hugging Face token is passed (via env var `HF_HUB_TOKEN`) for gated models like Mistral-7B. The script then loads the tokenizer and ensures proper configuration (setting `padding_side="right"` and defining a `pad_token` if missing). We use a special prompt format for training: each example’s `instruction` and `input` are wrapped in the `<s>[INST] ... [/INST]` tags, and the `output` is appended with a closing `</s>` tag. This mirrors the format expected by Mistral/LLaMA models for instruction prompts. The training dataset is built by loading the JSON files with 🤗 **Datasets**, subsetting to a percentage if specified (`--train_pct`) and then concatenating and shuffling them. We then map a tokenization function to transform each example into model inputs (input IDs, attention mask) and labels, where the prompt tokens are masked out in the labels (set to -100) so that only the response (`output`) tokens incur loss.

  LoRA configuration is defined inside the script: by default it uses rank `r=8`, alpha `16`, dropout `0.05`, and targets the model’s **query, key, value, and output projection layers** of each transformer block. These settings, along with the use of `prepare_model_for_kbit_training`, ensure the base model is ready for mixed-int8 precision training with LoRA adapters. We disable any existing model parallelism flags for compatibility on single GPU. Training is done using Hugging Face’s **Trainer** API: it reads hyperparameters like epochs, batch size (`--bsz`), gradient accumulation (`--grad_accum`), learning rate, etc., from the command-line arguments. We take care to disable FP16 automatic mixed precision if using 8-bit mode, because **bitsandbytes 8-bit** conflicts with FP16 training. During training, checkpoints are saved periodically (`--save_steps`) and at the end we save the final LoRA adapter weights and the tokenizer to the specified `--output_dir`. (The final saved model is actually a set of LoRA adapter weights – you need to combine them with the base model for inference, as done in `run_lora.py`.) The script prints progress messages, and upon completion you should see `✔ Training complete!` in the logs.

  *Example usage:* To fine-tune Mistral-7B on the full dataset for 1 epoch with batch size 2 (gradient accumulation 4 to simulate effectively batch size 8), you might run:

  ```bash
  python3 finetune_lora.py \
    --base_model mistralai/Mistral-7B-v0.1 \
    --conv_json data_conversation/train_conversations.json \
    --qcm_json data_questions/train_formatted.json \
    --output_dir lora_ckpts_mistral7b \
    --train_pct 100 \
    --epochs 1 \
    --bsz 2 \
    --grad_accum 4 \
    --learning_rate 2e-4 \
    --max_len 1024 \
    --save_steps 200
  ```

  This will load the Mistral-7B model (make sure you have `HF_HUB_TOKEN` set if required), use the full training data, and save the LoRA adapter weights to `lora_ckpts_mistral7b`. You can adjust these parameters as needed. On a Mac M1 (no CUDA), the script will automatically fall back to full precision CPU (or MPS) mode for the model – in that case you might want to use the smaller `TinyMistral-248M` model for experimentation due to hardware limitations.

* **`run_lora.py` – LoRA Inference Script:** After fine-tuning, use this script to load the base model **plus** the LoRA weights and run an interactive prompt loop for inference. It requires the `--base_model` argument (same original model used for training, e.g. `mistralai/Mistral-7B-v0.1`) and `--lora_dir` which is the path to the folder containing the saved LoRA adapter (the output directory from `finetune_lora.py`, which should contain files like `adapter_model.safetensors` and `adapter_config.json`). **Important:** Before running, set the environment variable `HF_HUB_TOKEN` to your Hugging Face token if the base model is gated. The script will check for this and refuse to run if the token is not set when needed. It then loads the base model from HF Hub (with 16-bit precision on GPU if available, or 32-bit on CPU) and **applies the LoRA adapter weights** on top of it using `PeftModel.from_pretrained`. The tokenizer is loaded similarly and configured for right-padding with a pad token (to match how it was during training). Once everything is ready, the script enters an interactive loop: it prompts you to **“Enter prompt (empty line to quit)”**. For each prompt you enter, it will internally prepend the same instruction that was used during training (telling the model it’s a medical doctor and to have a conversation with the patient) and wrap your input into the `[INST] ... [/INST]` format. This ensures the model sees a prompt formatted just like the training examples it saw. It then generates a response using `model.generate` with the specified decoding parameters (`--max_new_tokens`, `--temperature`, `--top_k`). The generated text (excluding the prompt) is printed out as the model’s answer. You’ll see an output like:

  ```text
  >>> Enter prompt (empty line to quit):

  >>> How can I treat a common cold at home?

  --- Model response ---
  To treat a common cold, you should get plenty of rest and stay hydrated. You can use over-the-counter medications like pain relievers for aches and fever, and decongestants for a stuffy nose. Drinking warm fluids like tea with honey can soothe your throat. If symptoms persist or worsen, consult a doctor for advice.
  ----------------------
  ```

  (This is an example – the actual response will vary since the model generates stochastically.) Enter an empty line to exit the loop. This script is great for quick interactive testing of the fine-tuned model.

* **`run_pretrained.py` – Pretrained Model Runner:** This is a convenience script to test inference on a pretrained model *without* any fine-tuning, mainly used for baseline comparisons or to sanity-check model loading. By default it points to the **TinyMistral-248M** model and a simple prompt, but you can specify `--model` to any Hugging Face model ID and provide a `--prompt` of your choosing. It will download the model and tokenizer, then generate text from the prompt and print the output to stdout. For example, you can run:

  ```bash
  python3 run_pretrained.py --model Locutusque/TinyMistral-248M --prompt "Explain diabetes like I'm five years old." --max_new_tokens 100
  ```

  which will load the 248M model (fits in CPU memory on a laptop) and produce an answer explaining diabetes in simple terms. Or, if you have access to a GPU, try the full 7B model:

  ```bash
  python3 run_pretrained.py --model mistralai/Mistral-7B-v0.1 --prompt "Patient: I have a headache and fever. Doctor:" --max_new_tokens 128
  ```

  (For the 7B model, ensure `HF_HUB_TOKEN` is set and you have enough GPU memory or use 8-bit loading via the other scripts.) The script uses a simple sampling strategy (temperature 0.8, top\_k 50 by default) for generation. It’s a quick way to verify the base model’s behavior or to ensure everything is installed correctly.

* **Dockerfile** – *Container Configuration*: The repository includes a Dockerfile to build a container image for running on the RCP cluster. It is based on NVIDIA’s PyTorch 23.11 base image (CUDA 12.6, PyTorch 2.7.0) for GPU support. The Dockerfile is configured to integrate with EPFL’s infrastructure: it accepts build arguments for your EPFL LDAP username, UID, group, and GID in order to create a matching user inside the container. This ensures that any files written to mounted volumes (scratch or home directories) have the correct permissions. The Dockerfile installs the Python dependencies listed in `requirements.txt` (see below), then copies all repository code into `/home/$USER` in the image. It sets the entrypoint to run the fine-tuning script by default (`ENTRYPOINT ["python3", "finetune_lora.py"]`). The default command is `--help`, so running the container with no arguments will just show the script options. When submitting jobs on the cluster, we often override the entrypoint/command to run specific tasks (as shown in the Run\:AI examples below).

* **`requirements.txt` – Dependencies:** This file pins specific versions of the required Python packages. Key dependencies include Hugging Face Transformers, Datasets, and Accelerate, as well as **PEFT** (for LoRA) and **BitsAndBytes** (for 8-bit quantization). The versions have been tested on the RCP cluster; for example, `transformers==4.52.4` and `peft==0.15.2`. (Accelerate 0.23+ is required so that `Trainer` can handle device offloading properly.) If you install these exact versions on your local machine, you should get reproducible behavior. Newer versions may also work, but those were the versions used during development. Additionally, the code uses the `torch` and `sentencepiece` packages. **Note:** The base Docker image already includes PyTorch and CUDA, and we add `torchvision==0.22.0` to match the installed PyTorch version.

* **Utilities and Data Checks:** In the `checking_data/` folder, there are a couple of helper scripts used during development. For instance, `sanity_check.py` will load a few samples from the conversation and QCM JSON files and print their content to verify the format. This is useful to ensure that the dataset JSONs are structured as expected (each entry should have `instruction`, `input`, and `output` keys, and you can eyeball that the text looks correct). Another utility, `inspect_tokenization.py`, demonstrates how an example from the dataset is tokenized into the prompt format and helps confirm that special tokens (e.g. the `<s>` and `</s>` tags and `[INST]` markers) are handled correctly by the tokenizer. These scripts aren’t needed for running the training or inference, but can be helpful for understanding the data preprocessing. There is also an **experimental** `finetune_qlora.py` script which was used to explore 4-bit (QLoRA) fine-tuning (quantizing the model to 4-bit NF4 precision during training). However, the primary focus of this project is on the standard LoRA approach (`finetune_lora.py`), which was thoroughly tested.

With the above files and scripts, you have the components to train the model on your data and then run it to get model responses. Next, we detail how to set up the environment on EPFL’s RCP cluster and run these tools in a containerized workflow.

## Setting Up and Running on EPFL RCP (Run\:AI Cluster)

The EPFL **RCP (Research Computing Platform)** provides a Kubernetes-based platform (CaaS) with GPU nodes, and we use **Run\:AI** to schedule and manage jobs on this cluster. Below is a step-by-step guide for building the Docker image, pushing it to EPFL’s container registry, configuring access to the cluster, uploading your data, and submitting training/inference jobs via the Run\:AI CLI. This guide assumes you have the required permissions on RCP (an RCP account with access to the SDSC-GE project and a personal namespace).

> **Prerequisites:** Make sure you have Docker installed locally, and the `runai` and `kubectl` CLI tools set up on your machine. You’ll also need VPN/SSH access to the EPFL network to reach the cluster. And importantly, get your Hugging Face authentication token ready (if you plan to use the Mistral 7B model) – sign in to Hugging Face and grab your token from your account settings.

### 1. Login to the RCP Container Registry

First, authenticate with EPFL’s private container registry (Harbor) so you can push your Docker image to it:

```bash
docker login registry.rcp.epfl.ch
# → Enter your GASPAR (EPFL) username and password when prompted
```

**Why?** RCP’s Kubernetes nodes pull images from a private registry (`registry.rcp.epfl.ch`), so you must have access. This step ensures Docker can push to (and pull from) that registry using your credentials.

### 2. Build the Docker Image

In the root of this repository (where the Dockerfile is located), build your container image. Use the provided command but replace the build-arg placeholders with your actual EPFL username/UID and group GID:

```bash
docker build --platform linux/amd64 \
  --build-arg LDAP_USERNAME=<YOUR_USERNAME> \
  --build-arg LDAP_UID=<YOUR_UID> \
  --build-arg LDAP_GROUPNAME=rcp-runai-sdsc-ge \
  --build-arg LDAP_GID=<YOUR_GID> \
  -t registry.rcp.epfl.ch/llm-medical-finetune/med-llm:0.1 .
```

Fill in `<YOUR_USERNAME>` with your Gaspar login (e.g. `jdoe`), `<YOUR_UID>` with your numerical UID, and `<YOUR_GID>` with the group ID (the SDSC-GE RunAI group). These values ensure the container’s internal user matches your host identity. We tag the image as `registry.rcp.epfl.ch/llm-medical-finetune/med-llm:0.1` – you can choose a different version tag if needed.

**Why?** The `--platform linux/amd64` flag forces the image to be built for x86\_64 (since you might be building on an ARM-based Mac). The build args inject your user info so that when Kubernetes mounts your volumes, file permissions stay consistent. Tagging the image with the registry URL and project name prepares it for pushing to RCP’s registry.

> *Note:* The file `requirements.txt` is copied first and dependencies installed to leverage Docker layer caching. If you modify the requirements or Dockerfile, the build may re-run those steps. The rest of the repository code is copied afterwards into the image.

### 3. Push the Image to the EPFL Registry

Once the build finishes successfully, push the image to the remote registry so the cluster can access it:

```bash
docker push registry.rcp.epfl.ch/llm-medical-finetune/med-llm:0.1
```

This will upload the image layers. It may take a while on first push (since the base image is large). If you update the image later, pushing with the same tag will upload only the changed layers.

**Why?** The Kubernetes cluster nodes will pull your image from this registry when running jobs. Pushing it ensures the image is available to all node workers.

### 4. (Optional) Test the Container Locally

You can do a quick local test run of the container to make sure everything is in order:

```bash
docker run -it registry.rcp.epfl.ch/llm-medical-finetune/med-llm:0.1 sh
```

This drops you into a shell inside the container (which is running on your local machine). From here you can manually invoke `python3 run_pretrained.py` or other commands to verify that the environment and scripts work as expected. For example, try `python3 run_pretrained.py --model Locutusque/TinyMistral-248M --prompt "Test prompt"` and see if it returns an output. Once done, exit the container.

**Why?** This test lets you catch any issues with the image (missing dependencies, etc.) before deploying. It’s easier to debug locally than on the cluster.

### 5. Logout (Cleanup Local Credentials)

After pushing the image, you can log out from the registry (optional):

```bash
docker logout registry.rcp.epfl.ch
```

This removes your stored credentials for the EPFL registry from your Docker client. It’s a good practice if you’re on a shared machine.

---

With the image now available on the registry, the next steps involve configuring your local environment to interact with the RCP Kubernetes cluster and using the Run\:AI CLI to submit jobs.

## RCP Cluster Access Setup

These steps ensure that your local `kubectl` and `runai` are pointing to the EPFL RCP cluster and that you’re authenticated.

### 6. Get the Kubernetes Config (`~/.kube/config`)

You need a kubeconfig file for the RCP cluster. If you don’t have one yet, create the directory and download the config:

```bash
mkdir -p ~/.kube
wget https://wiki.rcp.epfl.ch/public/files/kube-config.yaml -O ~/.kube/config && chmod 600 ~/.kube/config
```

This fetches the kubeconfig from EPFL’s documentation and stores it as your local config file. The `chmod 600` restricts access to you, since the file contains sensitive authentication info (certificates/tokens for the cluster).

**Why?** The kubeconfig file tells `kubectl` (and `runai`, which uses `kubectl` under the hood) how to communicate with the cluster’s API server. Without it, your CLI commands cannot reach the RCP cluster.

### 7. Verify the `kubectl` Client and Contexts

It’s good to check that `kubectl` is installed and see what contexts are available:

```bash
kubectl version --client
kubectl config get-contexts
```

The first command should print the kubectl client version (no need to contact the cluster). The second will list contexts in your kubeconfig. You should see an entry for `rcp-caas-prod` (the production RCP CaaS cluster). If that’s present, your config is in place.

**Why?** This verifies that the kubeconfig is recognized and you have the context for RCP. It’s easy to accidentally have multiple kubeconfigs or contexts; here we ensure the correct one is set.

### 8. Select the RCP Context

Make sure your kubectl and runai are targeting the RCP cluster:

```bash
runai config cluster rcp-caas-prod
kubectl config use-context rcp-caas-prod
```

The first command tells runai to use the `rcp-caas-prod` cluster (if it isn’t already), and the second does the same for kubectl. From now on, any `kubectl` or `runai` command will interact with the RCP cluster.

**Why?** If you have multiple clusters or if runai was pointing to a different default, this ensures we explicitly target the EPFL cluster for the next steps.

### 9. Login with Run\:AI

Authenticate the Run\:AI CLI with your EPFL credentials:

```bash
runai login
runai whoami
```

`runai login` will open a browser or prompt for SSO – follow the steps to login with your EPFL account. After that, `runai whoami` should show your username and current project if login succeeded.

**Why?** Run\:AI commands (like submitting jobs) require an authentication token. The login step obtains a token via EPFL Single Sign-On and caches it. `whoami` is just to double-check you’re logged in under the correct user.

### 10. Select Your Project

Run\:AI uses the concept of projects which map to Kubernetes namespaces (in our case, likely `sdsc-ge-<YourUsername>`):

```bash
runai list project        # shows all projects you have access to
runai config project sdsc-ge-<YOUR_USERNAME>
runai config view         # optional, to see current config
kubectl config get-contexts  # you should see a context like runai-sdsc-ge-<YOUR_USERNAME>
```

Replace `<YOUR_USERNAME>` accordingly (e.g., `sdsc-ge-jdoe`). The first command lists projects, which helps you find the exact project name. The second sets your current project. The `kubectl config get-contexts` will show that a new context (with runai and your project) is active.

**Why?** Your GPU quota and resources on RCP are organized by project. You must select your personal project (or the relevant group project) to submit jobs to the correct namespace. This also ensures you have permission to launch jobs there.

### 11. Check Your Persistent Volumes (Storage)

On RCP, you typically have two persistent storage areas: a **scratch** (NAS3) and a **home** (NAS1). Check that these are present:

```bash
kubectl get pvc -n runai-sdsc-ge-<YOUR_USERNAME>
```

This will list PersistentVolumeClaims in your namespace. You should see something like:

* `<YOUR_USERNAME>-sdsc-ge-scratch` – a scratch volume (high-speed, temporary storage, often named `sdsc-ge-scratch`)
* `<YOUR_USERNAME>-home` – your home NAS volume for long-term storage

**Why?** We need the exact names of these PVCs to mount them in jobs. The scratch space is where we’ll put training data and possibly write output models, because it’s faster and larger. The home space is your standard network drive. Knowing the names (especially if they differ slightly from default) is important for the next steps.

---

At this point, you have your image in the registry, your local environment configured to talk to the cluster, and you know your storage volumes. Next, let’s upload the dataset and then run some jobs.

## Data Preparation on the Cluster

### 12. Upload Your Data to Scratch (NAS3)

Copy your training and evaluation data onto the cluster’s scratch storage. From your local machine (or whichever has the data), use SCP to transfer files to the RCP jump host:

```bash
scp -r /path/to/my-data-files <YOUR_USERNAME>@jumphost.rcp.epfl.ch:/mnt/sdsc-ge/scratch/
```

This will prompt for your EPFL password and then copy the entire `my-data-files` directory (or it could be individual JSON files) to the scratch space.

**Why?** Placing data on `/mnt/sdsc-ge/scratch/` (NAS3) makes it accessible to all RCP compute nodes via the `sdsc-ge-scratch` PVC. The scratch filesystem is high-performance, suitable for heavy I/O during training. We don’t include large data in the Docker image (to keep it lean), so we upload it separately.

### 13. Verify the Data on the Jump Host

It’s a good idea to confirm that the files are indeed on the scratch volume. SSH into the jump host and list the files:

```bash
ssh <YOUR_USERNAME>@jumphost.rcp.epfl.ch
cd /mnt/sdsc-ge/scratch/
ls
```

You should see your uploaded `train_conversations.json`, `train_formatted.json`, or whatever data directory you scp’d over. You can even open one of the files to ensure it’s not corrupted. Note that on the jump host, the path is `/mnt/sdsc-ge/scratch/…`, but when we mount this volume into a job, it will appear at a mount path we specify (we mount it to `/scratch` inside the container).

**Why?** This manual check gives peace of mind that the data is in the right place and named correctly. When the Kubernetes job mounts the PVC, it will surface the same files to the container.

*(Side note: The jump host is a gateway machine; you can’t run heavy computations there, but it’s useful for managing files.*)

---

Now we have data available on the cluster’s shared storage. Let’s run some jobs to ensure everything works: first a simple test job, then the actual fine-tuning and inference jobs.

## Submitting Jobs with Run\:AI

We will use the `runai submit` command to run jobs on the cluster. Below are examples for different scenarios. In each case, we specify resource requests, the Docker image (the one you built), volume mounts for data, and the command to execute inside the container.

### 14. Submit a GPU Test Job (sleep test)

Before doing any real work, it’s wise to do a minimal test to verify that the cluster can schedule a job with your image:

```bash
runai submit \
  --name test-sleep \
  --image registry.rcp.epfl.ch/llm-medical-finetune/med-llm:0.1 \
  --gpu 0.1 \
  --existing-pvc claimname=sdsc-ge-scratch,path=/scratch \
  --command -- /bin/bash -c "sleep 60"
```

Here we submit a job named “test-sleep” that uses our container. We request `--gpu 0.1` which on RCP means one tenth of a GPU. You could also request a full GPU with `--gpu 1` if you prefer. We attach the scratch PVC (`sdsc-ge-scratch`) to `/scratch` in the container. The job simply runs `sleep 60` for a minute and then exits.

**Why?** This quick job does almost nothing, but it tests that a GPU can be allocated, the container starts up, and volumes mount properly. If this succeeds, it’s a good sign that more complex jobs will run. If it fails, you can inspect events to see what went wrong (image pull issues, scheduling, etc.).

### 15. Submit a Fine-Tuning Job (LoRA Training on RCP)

Now, to run the actual fine-tuning on the cluster, submit a job that uses `finetune_lora.py`. You can either rely on the Docker entrypoint or explicitly call the script. We’ll explicitly call it here for clarity and to pass arguments:

```bash
runai submit \
  --name train-mistral7b \
  --image registry.rcp.epfl.ch/llm-medical-finetune/med-llm:0.1 \
  --gpu 1 \
  --env HF_HUB_TOKEN=<YOUR_HF_TOKEN> \
  --existing-pvc claimname=sdsc-ge-scratch,path=/scratch \
  --command -- bash -c "cd /home/<YOUR_USERNAME> && \
    python3 finetune_lora.py \
      --base_model mistralai/Mistral-7B-v0.1 \
      --conv_json /scratch/train_conversations.json \
      --qcm_json /scratch/train_formatted.json \
      --output_dir /scratch/lora_ckpts_mistral7b \
      --epochs 3 --bsz 1 --grad_accum 8 --learning_rate 2e-4 --max_len 1024 --save_steps 200"
```

Let’s break down what this does:

* We name the job “train-mistral7b” and use our image tag. We request a full GPU (`--gpu 1`).
* We pass in the Hugging Face token securely via an environment variable. **Do not hardcode your token**; replace `<YOUR_HF_TOKEN>` with the real token string. This env var will be picked up by the script inside (it checks `HF_HUB_TOKEN`).
* We mount the scratch PVC to `/scratch`. This is where our training data resides (as uploaded in step 12) and also where we plan to save outputs. We don’t necessarily need the home PVC here since our code is already baked into the image, but mounting home as well doesn’t hurt (it could be used if, say, you wanted to write final models to your home directory for long-term storage).
* The command we run inside the container: we `cd` to the working directory (the code is in `/home/<YOUR_USERNAME>` inside the image) and invoke `finetune_lora.py` with appropriate arguments. We point `--base_model` to `mistralai/Mistral-7B-v0.1`. We use the data files from the scratch volume (`/scratch/train_conversations.json` etc., assuming you uploaded those exact filenames). The `--output_dir` is set to `/scratch/lora_ckpts_mistral7b` – this means the model checkpoints and final adapter will be saved to the scratch storage (persisting after the job ends, because the scratch is mounted) rather than inside the container ephemeral filesystem. Hyperparameters here are just an example (3 epochs, batch size 1, grad\_accum 8 to simulate effective batch size 8, etc.). You can modify them or add others as needed.

**Why?** This job will actually perform the fine-tuning on the cluster GPU. We included the HF token so the model can download Mistral-7B, and by saving outputs to the PVC we can retrieve the fine-tuned model later. Monitor the job (as shown in steps below) – it will likely take some time depending on data size and epochs. You should see logs indicating the training progress, and eventually the `✔ Training complete!` message when done.

### 16. Submit an Inference Job (TinyMistral Example)

As another example, let’s run a quick inference on the cluster using the smaller model, just to see everything working end-to-end:

```bash
runai submit \
  --name infer-tiny \
  --image registry.rcp.epfl.ch/llm-medical-finetune/med-llm:0.1 \
  --gpu 0.1 \
  --existing-pvc claimname=sdsc-ge-scratch,path=/scratch \
  --command -- bash -c "cd /home/<YOUR_USERNAME> && \
    python3 run_pretrained.py \
      --model Locutusque/TinyMistral-248M \
      --prompt 'Explain diabetes like I am five years old.' \
      --max_new_tokens 100"
```

This job uses the same container but runs `run_pretrained.py` instead, on the TinyMistral model. We allocate a small fraction of a GPU (0.1) since the 248M model is lightweight. We mount scratch just in case (not strictly needed for this, since it downloads the model from HF Hub and doesn’t use any dataset). We run a prompt through the model (a simple test prompt).

**Why?** This serves as a quick integration test – it will download the model inside the cluster environment and generate an output. The output (the explanation of diabetes) will appear in the job logs. It’s a way to ensure the inference code and environment works on the cluster as expected. For a real use-case, you might instead run `run_lora.py` with the LoRA weights on the cluster or run batch inference jobs on a set of prompts, etc.

*(You could also submit a job to run `run_lora.py` for interactive mode, but typically for interactive usage you’d just attach to a running training pod or use `runai bash` as shown below, rather than submit a non-terminating interactive job.)*

### 17. Check Job Logs

To see the output of your jobs (training progress, printouts, or inference result), use:

```bash
runai logs train-mistral7b    # view logs for the training job
runai logs infer-tiny         # view logs for the inference job
```

This will stream the stdout/stderr from the job’s container. For the training job, you should see messages like model loading, dataset preparation, training iterations, etc., as printed by our script (e.g., `>>> Loading model ...`, `>>> Starting training…`, and so on). For the inference job, you’ll see the model’s generated text output printed by `run_pretrained.py`. Checking logs is crucial to ensure everything ran correctly. If something failed, the logs will contain error messages to help debug.

**Why?** Logs let you monitor the progress and catch errors. On RCP, jobs don’t have a live interactive console by default, so `runai logs` is how you peek at what’s happening inside the container.

### 18. Monitor Job Status

List your jobs or describe them for more details:

```bash
runai list jobs
runai describe job train-mistral7b -p sdsc-ge-<YOUR_USERNAME>
```

The `list jobs` command will show an overview of all jobs in your project (name, status, GPU usage, etc.). The `describe job` provides detailed info on a specific job, including resource usage, conditions, events, etc. (Here `-p` specifies the project/namespace, which we set earlier, but include it to be explicit.)

**Why?** This helps you verify that the job is running or has completed. You can see if a job is pending (e.g., waiting for resources), running, or finished, and whether it succeeded or failed. It’s especially useful for the training job which might run for a long time – you can check that it’s still active and using the GPU as expected.

### 19. Attach to a Running Pod (Interactive Shell)

Run\:AI allows you to exec into a running job’s pod for debugging or interactive exploration:

```bash
runai bash train-mistral7b
```

This will give you a shell inside the container for the training job (if it’s still running, or even after completion if the pod hasn’t been cleaned up yet). You’ll be in the container’s environment, in case you want to inspect files, run `nvidia-smi` to check GPU usage, or even launch Python for interactive work.

**Why?** This is extremely useful for debugging – you can poke around in the live container, check that files are where they should be (e.g., verify that `/scratch` data is visible inside, check that the output directory has been created, etc.), and run additional commands if needed. It’s like SSHing into the container. If your training job is hanging, you could use this to investigate. If the job completed and you forgot to copy some output, you might use this to manually retrieve it (as long as the pod still exists).

*Remember to exit the shell (`Ctrl+D` or `exit`) when done. Also, note that once a job’s pod is completed and removed by Kubernetes (depending on the backoffLimit or ttl), you may not be able to bash into it, so use it while the job is running or shortly after completion.*

---

By following steps 1–19, you have:

1. Built and pushed a Docker image with your fine-tuning code.
2. Configured your local environment to interact with the EPFL RCP cluster via kubectl/runai.
3. Uploaded your datasets to the cluster’s scratch storage.
4. Submitted both a test job and actual fine-tuning/inference jobs.
5. Monitored the jobs, checked outputs, and even hopped into a container interactively.

This covers the full workflow of training a medical-domain LLM on the cluster and getting results out. In practice, you might iterate on training (adjust hyperparameters, try different data, etc.) and then use the inference script to evaluate the model’s answers.

**Hugging Face Token Note:** If you are using models that require accepting a license (like Mistral-7B) or any private models, you must provide your Hugging Face authentication token to the jobs. In the examples above, we did this by using `--env HF_HUB_TOKEN=<YOUR_HF_TOKEN>` in the runai submit command for training. Alternatively, you could configure the token globally on the cluster or use a secret. For interactive use (like `runai bash`), you can also manually run `huggingface-cli login` inside the container to authenticate. But using the environment variable is straightforward since our scripts will automatically use it. Always keep your token secure and don’t hardcode it in code or public config files.

## Running the Code Locally (Optional)

While the primary target is the RCP cluster, you can also run this project on a local machine for development or small-scale tests. Here are some tips for local execution:

* **Setup:** You’ll need Python 3.8+ and to install the required packages. It’s recommended to use a virtual environment. You can install dependencies with:

  ```bash
  pip install -r requirements.txt
  ```

  Ensure that your environment has PyTorch installed (with CUDA if you have a GPU). On Mac silicon (M1/M2), PyTorch with MPS support can be used – the `finetune_lora.py` script will detect the absence of CUDA and use CPU (or MPS) accordingly.

* **Data:** Place your `train_conversations.json` and `train_formatted.json` in the `data_conversation/` and `data_questions/` directories (or anywhere, but then provide the correct path to the scripts). Since these files might be large, you might work with a subset locally.

* **Finetuning on CPU/MPS:** Fine-tuning a 7B model without a GPU is **very** slow and may not be feasible due to memory constraints. However, you can test the pipeline with the smaller TinyMistral-248M model on CPU. For example, run:

  ```bash
  python3 finetune_lora.py \
    --base_model Locutusque/TinyMistral-248M \
    --conv_json data_conversation/train_conversations.json \
    --qcm_json data_questions/train_formatted.json \
    --output_dir lora_ckpts_tiny \
    --train_pct 5 --epochs 1
  ```

  This would fine-tune on 5% of the data for 1 epoch, which is a reasonable quick test. You should see output indicating the training steps and it should finish relatively quickly on CPU for the tiny model. Make sure to export `HF_HUB_TOKEN` if the model is gated (TinyMistral-248M is usually open, but Mistral-7B is gated).

* **Running inference locally:** You can use `run_pretrained.py` to test the base model responses, as shown in the examples above, or `run_lora.py` to test an adapter. For `run_lora.py`, if you fine-tuned on Mac, you can load the resulting `lora_ckpts_tiny` with:

  ```bash
  export HF_HUB_TOKEN="<YOUR_HF_TOKEN>"
  python3 run_lora.py --base_model Locutusque/TinyMistral-248M --lora_dir lora_ckpts_tiny
  ```

  and then enter prompts interactively. The script will load the base TinyMistral model and apply your LoRA; this should run on CPU (or MPS) albeit slowly. The first time you run a model, it will download from Hugging Face, which can take time – subsequent runs will use the cached model.

* **Expected outputs:** When running locally, you’ll see similar prints as on the cluster. During training, the Trainer will not output a progress bar by default (since we set `logging_steps=50` and `report_to="none"` to keep logs clean), but you will see our custom print statements like `>>> Loading model...`, `>>> Starting training…`, and eventually the completion message. In interactive inference, you’ll see the prompt `>>> Enter prompt` and the model’s response as shown earlier.

Running locally is mainly useful for development, debugging, or small-scale experimentation. For full training on the 7B model, you’ll want to use a GPU on RCP (or another GPU machine) due to the hardware requirements.

## Contributions

Below we highlight contributions from **Oscar** (the author of this README and a contributor for the infrastructure and implementation in this repo):

* **Research & Planning**

  * Researched various open-source pretrained language models and selected **Mistral-7B** for full-scale medical fine-tuning and **TinyMistral-248M** for local testing and proof-of-concept.
  * Created `run_pretrained.py` to experiment with downloading, loading, and generating from those base models, verifying compatibility with Hugging Face APIs and ensuring that the inference workflow works before any fine-tuning.
  * Read EPFL RCP cluster documentation in detail and took extensive notes on authentication, storage mounts (NAS1 / NAS3), GPU allocation, and Run\:AI / `kubectl` usage. Those notes informed the step-by-step RCP instructions in this README.

* **Repository & Environment Setup**

  * Initialized and organized the GitHub repository, including directories for code (`.`), data checks (`checking_data/`), and documentation.
  * Drafted this comprehensive `README.md` that covers cloning the repo, container build/push, RCP cluster configuration, Run\:AI job submission, log/status inspection, and interactive debugging - aiming to make the entire workflow reproducible for new users.
  * Developed the Docker environment for RCP:

    * Wrote the `Dockerfile` that builds from NVIDIA’s PyTorch 23.11 CUDA 12.6 base image, installs all Python dependencies (from `requirements.txt`), and creates a container user matching EPFL LDAP credentials (`LDAP_USERNAME`, `LDAP_UID`, `LDAP_GROUPNAME`, `LDAP_GID`) to avoid file-permission issues when mounting NAS volumes.
    * Verified that all required packages (Transformers, Datasets, PEFT, BitsAndBytes, Accelerate, etc.) install correctly and that the code inside (training and inference scripts) runs without errors in the container.
  * Created the EPFL RCP Harbor project `llm-medical-finetune` and pushed the initial Docker image (`med-llm:0.1`) to `registry.rcp.epfl.ch/llm-medical-finetune/med-llm:0.1`. Tagged subsequent images (e.g. `:0.4`) to reflect updates.
  * Determined and documented the storage strategy on RCP: use NAS3 scratch (`/mnt/sdsc-ge/scratch/`) for large training data and model outputs, and NAS1 home (`/mnt/nas1/home/…`) for any long-term storage or small config files.
  * Uploaded the necessary JSON datasets (`train_conversations.json`, `train_formatted.json`, etc.) to NAS3 scratch via `scp` and verified their paths on the RCP jump host.

* **Fine-Tuning Pipeline Implementation (`finetune_lora.py`)**

  * Rewrote and refactored the fine-tuning script to handle both local (Mac M1 CPU/MPS) and RCP (GPU + 8-bit quantization) environments seamlessly.

    * Added logic to detect CUDA and load the base model in 8-bit via BitsAndBytes (when on GPU), or full-precision on CPU/MPS. Ensured that the HF token and `trust_remote_code=True` are passed when loading gated models (e.g., `mistralai/Mistral-7B-v0.1`).
    * Configured the tokenizer to use **right-padding** (`tokenizer.padding_side = "right"`) and set `pad_token_id = eos_token_id` if missing, to avoid training errors when padding input sequences.
    * Implemented the `<s>[INST] {instruction} {input} [/INST] {output} </s>` prompt schema exactly, then masked out prompt tokens in the labels (setting them to `-100`) so that the model only computes loss on the generated response.
    * Defined a LoRA configuration (`r=8`, `lora_alpha=16`, `lora_dropout=0.05`, `target_modules=["q_proj","k_proj","v_proj","o_proj"]`) to inject trainable adapters into the model’s self-attention projections.
    * Wrapped the model with PEFT’s `get_peft_model` and called `prepare_model_for_kbit_training` to properly handle 8-bit quantization adjustments.
    * Diagnosed and fixed a critical training bug where `fp16=True` and 8-bit quantization conflicted; forced `fp16=False` whenever loading the model in 8-bit mode to prevent runtime errors.
    * Thoroughly tested local fine-tuning on TinyMistral-248M (using 1% of the data for 1 epoch) on a Mac M1. Verified that JSON loading, tokenization, data sharding, Trainer loop, and output saving all work correctly on CPU/MPS.
  * Configured the Trainer to use gradient accumulation, a customizable learning rate, and a flexible batch size. Added command-line arguments (`--epochs`, `--bsz`, `--grad_accum`, etc.) so the same script can be run on Mac or RCP with minimal changes.

* **Inference & Utility Scripts**

  * Developed `run_lora.py` to load the base model (Mistral-7B or TinyMistral-248M), apply the saved LoRA adapter weights (from `lora_adapter` directory), and offer an **interactive chat loop** with user prompts wrapped in the same `<s>[INST]…[/INST]` format. Ensured that:

    * The base model and adapter are loaded to `torch_dtype=torch.float16` on CUDA or `torch.float32` on CPU.
    * The LoRA adapter is applied using `PeftModel.from_pretrained(...)` with `trust_remote_code=False`.
    * The tokenizer is re-initialized with the same padding settings and a valid `pad_token_id`.
    * Generation uses `max_new_tokens`, `temperature`, and `top_k` parameters to control decoding.
  * Verified that the fine-tuned model can generate coherent medical advice in interactive mode, demonstrating the complete train>save>load>generate pipeline.
  * Wrote data-checking utilities under `checking_data/`:

    * `sanity_check.py` to print out a few raw JSON examples from the conversation and QCM datasets to confirm field names and text formatting.
    * `inspect_tokenization.py` to show exactly how a single example is tokenized into input IDs, attention masks, prompt lengths, and labels. This helped catch early tokenization issues (e.g., missing special tokens, incorrect mask indexing).

* **Hugging Face Hub Integration**

  * Ensured all training and inference scripts check for the `HF_HUB_TOKEN` environment variable and pass it into any `from_pretrained(...)` calls when loading gated models (such as Mistral-7B).
  * Documented the need to `export HF_HUB_TOKEN="hf_XXXXXXXXXXXX"` before running training or inference on the cluster or locally. This avoids manual login prompts inside the container and prevents “access denied” errors when fetching model files.
  * When cluster jobs require the token, we passed it via the `--e HF_HUB_TOKEN=<token>` flag in `runai submit` commands, ensuring seamless model downloads at runtime.

* **Cluster Orchestration & Testing**

  * Created and validated **Run\:AI submission commands** for multiple scenarios:

    * **Test sleep job** that simply requests a fractional GPU (`--gpu 0.1`) and runs `sleep 60`, verifying that the Docker image pulls correctly, the GPU is allocated, and volumes mount properly.
    * **TinyMistral fine-tuning job** (`--gpu 1 --epochs 2 --bsz 2 --grad_accum 8`) to sanity-check that training on a small model works end-to-end on RCP.
    * **Mistral-7B fine-tuning job** (`--gpu 1 --epochs 10 --bsz 2 --grad_accum 8 --learning_rate 1e-4`) to run full-scale LoRA training on the cluster’s GPU nodes, loading data from NAS3 scratch and saving adapter weights back to scratch.
    * **Inference job** for TinyMistral (`run_pretrained.py`), allocating a small GPU fraction and generating a short explanation of diabetes, confirming that inference works in the cluster environment.
    * **Interactive jobs** by using `runai bash <job-name>` to open a shell inside a running container, enabling live debugging (checking `nvidia-smi`, inspecting `/scratch` contents, etc.).
  * Debugged volume mount permission issues by adjusting container user mapping (via `LDAP_USERNAME`, `LDAP_UID`, `LDAP_GID`, `LDAP_GROUPNAME` build arguments) so that files written to `/scratch` and `/home` have the correct owner and permissions.
  * Monitored job logs diligently (`runai logs <job>`), identified and fixed runtime errors (e.g., missing `config.json` warnings, label\_names warning from `PeftModelForCausalLM`), and iterated until the pipeline ran smoothly from data load through training to model export.
  * Documented best practices for inspecting Kubernetes contexts (`kubectl config get-contexts`), verifying PVCs (`kubectl get pvc -n runai-sdsc-ge-<USERNAME>`), and diagnosing scheduling issues or resource shortages.

In summary, this project’s contributions cover everything from initial model selection and local testing to building a production-ready container, orchestrating jobs on EPFL’s RCP cluster, and writing the scripts that train and deploy a medical-domain LoRA-adapted LLM. The bullet points above capture each discrete task that was completed to make the entire workflow reproducible and robust for both local and cloud environments.

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


---

**We hope this README serves as a complete guide** for anyone (likely neededing certain EPFL access) to get started. By following the instructions above, one should be able to reproduce the fine-tuning of a medical LLM on the EPFL RCP platform and interact with the resulting model. If you encounter any issues or have questions, feel free to reach out (or consult the EPFL RCP documentation and Hugging Face docs as needed). Good luck and happy LLM-ing on RCP
