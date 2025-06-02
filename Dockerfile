#####################################
# Use NVIDIA’s official PyTorch 23.11 image
#####################################
FROM nvcr.io/nvidia/pytorch:23.11-py3

#####################################
# RCP CaaS requirement (Storage / user mapping)
#####################################
ARG LDAP_USERNAME
ARG LDAP_UID
ARG LDAP_GROUPNAME
ARG LDAP_GID

# 1) Create the EPFL user/group inside the container
RUN groupadd "${LDAP_GROUPNAME}" --gid "${LDAP_GID}" \
 && useradd  -m -s /bin/bash -g "${LDAP_GROUPNAME}" -u "${LDAP_UID}" "${LDAP_USERNAME}"

# 2) Copy your local code into /home/<user>
RUN mkdir -p /home/${LDAP_USERNAME}
COPY ./ /home/${LDAP_USERNAME}

# 3) Fix ownership
RUN chown -R ${LDAP_USERNAME}:${LDAP_GROUPNAME} /home/${LDAP_USERNAME}

WORKDIR /home/${LDAP_USERNAME}

#####################################
# By default, NVIDIA’s image already has:
#   • CUDA 12.6 runtime + cuDNN + NCCL 
#   • Torch 2.7.0 + torchvision (matching CUDA version)
#   • bitsandbytes (unsure of version—just re‐install 0.42.0 below)
#   • (No guarantee on accelerate/transformers versions)
#####################################

# 4) As root: upgrade pip + setuptools, install exactly our requirements.txt
#    (so that any pre‐installed transformers/accelerate get overridden).
RUN python3 -m pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt \
 && rm -rf /root/.cache/pip

#####################################
# Switch to the unprivileged EPFL user
#####################################
USER ${LDAP_USERNAME}

# 5) ENTRYPOINT to your training script
ENTRYPOINT ["python3", "finetune_lora.py"]
CMD ["--help"]
