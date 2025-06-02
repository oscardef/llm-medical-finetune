#####################################
# Use NVIDIA’s official PyTorch 23.11 GPU image
#####################################
FROM nvcr.io/nvidia/pytorch:23.11-py3

#####################################
# RCP CaaS requirement (Storage / user mapping)
#####################################
ARG LDAP_USERNAME
ARG LDAP_UID
ARG LDAP_GROUPNAME
ARG LDAP_GID

# Create the EPFL user/group inside the container
RUN groupadd "${LDAP_GROUPNAME}" --gid "${LDAP_GID}" \
 && useradd  -m -s /bin/bash -g "${LDAP_GROUPNAME}" -u "${LDAP_UID}" "${LDAP_USERNAME}"

# Copy your local code into the container
RUN mkdir -p /home/${LDAP_USERNAME}
COPY ./ /home/${LDAP_USERNAME}

# Make sure the EPFL user owns everything
RUN chown -R ${LDAP_USERNAME}:${LDAP_GROUPNAME} /home/${LDAP_USERNAME}

WORKDIR /home/${LDAP_USERNAME}

#####################################
# Switch to the unprivileged EPFL user
#####################################
USER ${LDAP_USERNAME}

# By default, NVIDIA’s PyTorch image already has:
#   • CUDA 12.6 runtime + cuDNN + NCCL 
#   • Torch 2.7.0 + torchvision (matching CUDA version)
#   • bitsandbytes (if you chose the “-bpc” or “-bfloat” variants; otherwise pip‐install it)
#
# So you only need to pip install your extra Python requirements.

# Copy only the files we actually need (no data_*/ directories)
COPY finetune_lora.py requirements.txt /home/${LDAP_USERNAME}/

# Upgrade pip, then install Python dependencies
RUN python3 -m pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && rm -rf ~/.cache/pip

# Finally, point ENTRYPOINT at your training/fine-tuning script
ENTRYPOINT ["python3","finetune_lora.py"]
CMD ["--help"]
