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

## Running on RCP Cluster

Below are step-by-step instructions for building, pushing, and running the Docker image on EPFL’s RCP CaaS (GPU) infrastructure via `runai`. Each step includes a brief explanation of its purpose.

### 1. LOGIN to the RCP Container Registry

```bash
docker login registry.rcp.epfl.ch
# → Enter your GASPAR username and password
```

**Why?** This authenticates with EPFL’s private registry so you can push and pull images.

### 2. BUILD the Docker Image

```bash
docker build --platform linux/amd64 \
  --build-arg LDAP_USERNAME=USERNAME \
  --build-arg LDAP_UID=YOUR_UID \
  --build-arg LDAP_GROUPNAME=rcp-runai-sdsc-ge \
  --build-arg LDAP_GID=YOUR_GID \
  -t registry.rcp.epfl.ch/llm-medical-finetune/med-llm:0.1 \
  .
```

**Why?**

* `--platform linux/amd64` ensures the image is compatible with RCP’s x86\_64 nodes (necessary when building on an M1/M2 laptop).
* The `--build-arg` flags set up your EPFL user inside the container so that mounted volumes have correct permissions.
* Tagging as `registry.rcp.epfl.ch/...:0.1` prepares the image for pushing.

### 3. PUSH the Image to the EPFL Registry

```bash
docker push registry.rcp.epfl.ch/llm-medical-finetune/med-llm:0.1
```

**Why?** Make your custom container available to RCP CaaS.

### 4. (Optional) RUN the Container Locally

```bash
docker run -it registry.rcp.epfl.ch/llm-medical-finetune/med-llm:0.1 sh
```

**Why?** Kick the tires locally to verify that Python, `run_pretrained.py`, and dependencies work before deploying to the cluster.

### 5. LOGOUT from the Registry

```bash
docker logout registry.rcp.epfl.ch
```

**Why?** Optional cleanup when you’re done pushing.

---

## RCP CaaS SETUP

Follow these steps to configure your local environment to talk to the RCP Kubernetes cluster and use Run\:AI.

### 6. DOWNLOAD & INSTALL `~/.kube/config`

If you do not have `~/.kube/config`, create it first:

```bash
mkdir -p ~/.kube
```

Then fetch the RCP kubeconfig and secure it:

```bash
wget https://wiki.rcp.epfl.ch/public/files/kube-config.yaml \
     -O ~/.kube/config \
  && chmod 600 ~/.kube/config
```

**Why?** Kubernetes clients (`kubectl`, `runai`) need this file to know how to authenticate against RCP’s API.

### 7. VERIFY `kubectl` CLIENT

```bash
kubectl version --client
kubectl config get-contexts
```

**Why?** Confirm you can communicate with the correct contexts. You should see `rcp-caas-prod` as one of the contexts.

### 8. SELECT RCP CONTEXT

```bash
runai config cluster rcp-caas-prod
kubectl config use-context rcp-caas-prod
```

**Why?** Ensure that subsequent `kubectl` or `runai` commands target the RCP cluster rather than a local or other cluster.

### 9. AUTHENTICATE with Run\:AI

```bash
runai login
runai whoami
```

**Why?** SSO‐based login retrieves a token so `runai` CLI can submit jobs and manage your project.

### 10. VERIFY OR SELECT YOUR PROJECT

```bash
runai list project
runai config project sdsc-ge-USERNAME
runai config view
kubectl config get-contexts   # ensure you see runai-sdsc-ge-USERNAME
```

**Why?** RCP CaaS segregates GPU quota by “project.” Make sure you’ve selected your personal project namespace (e.g. `sdsc-ge-USERNAME`).

### 11. CHECK YOUR PVCs

```bash
kubectl get pvc -n runai-sdsc-ge-USERNAME
```

You should see two PersistentVolumeClaims (PVCs):

* `sdsc-ge-scratch`: This is NAS3 (scratch) for fast read/write during jobs.
* `home`: This points to your NAS1 share for long‐term storage.
  **Why?** Knowing the exact PVC names allows you to mount them into your Run\:AI jobs.

---

## Data Workflow on RCP

### 12. UPLOAD YOUR DATA to NAS3

From your workstation:

```bash
scp -r ~/Documents/my-data USERNAME@jumphost.rcp.epfl.ch:/mnt/sdsc-ge/scratch/
```

**Why?** Copy training or inference data onto the high-performance scratch storage so GPU nodes can access it.

### 13. VERIFY ON JUMPHOST

SSH into the jump host:

```bash
ssh USERNAME@jumphost.rcp.epfl.ch
cd /mnt/sdsc-ge/scratch/
ls
```

**Why?** Confirm your data is in `/mnt/sdsc-ge/scratch/` so that pods mounting `sdsc-ge-scratch` will see it under `/scratch`.

---

## Running Jobs on RCP

Below are examples for submitting a “sleep” test and an actual TinyMistral inference.

### 14. SUBMIT A SLEEP TEST (JOB JUST SLEEPS)

```bash
runai submit \
  --name test-1 \
  --image registry.rcp.epfl.ch/llm-medical-finetune/med-llm:0.1 \
  --gpu 0.1 \
  --existing-pvc claimname=sdsc-ge-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/USERNAME \
  --command -- /bin/bash -ic "sleep 600"
```

**Why?** Quickly verify that a GPU is allocated, volumes mount correctly, and the container starts without issues.

### 15. SUBMIT TINY MISTRAL INFERENCE (RUN YOUR SCRIPT)

```bash
runai submit \
  --name test-2 \
  --image registry.rcp.epfl.ch/llm-medical-finetune/med-llm:0.1 \
  --gpu 0.1 \
  --existing-pvc claimname=sdsc-ge-scratch,path=/scratch \
  --command -- bash -lc 'cd /home/USERNAME && \
    python3 run_pretrained.py \
      --model Locutusque/TinyMistral-248M \
      --prompt "Explain diabetes like I am five years old." \
      --max_new_tokens 100'
```

**Why?** This invokes `run_pretrained.py` directly.

* We only mount `sdsc-ge-scratch` under `/scratch`.
* We do not re-mount `/home/USERNAME`, so the code baked into the image remains intact.

### 16. CHECK JOB LOGS

```bash
runai logs test-2
```

**Why?** View TinyMistral’s generated output or any error messages.

### 17. CHECK JOB STATUS

```bash
runai list jobs
# or for detailed info:
runai describe job test-2 -p sdsc-ge-USERNAME
```

**Why?** Monitor GPU usage, pod status, and ensure your job succeeded.

### 18. ATTACH TO A RUNNING POD (INTERACTIVE SHELL)

```bash
runai bash test-2
```

**Why?** Get a live shell inside your pod to debug, modify inputs, or run additional commands.

---

Now you have a complete, end-to-end guide:

1. Build & push your container
2. Configure `kubectl` & `runai`
3. Stage data on NAS3
4. Submit both a simple sleep-600 test and real TinyMistral inference
5. Inspect logs, status, and attach interactively

Feel free to copy-paste this entire file into your GitHub repository as `README.md`.
Happy LLM-ing on RCP!
