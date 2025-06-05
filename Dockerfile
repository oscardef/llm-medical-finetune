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
 && useradd -m -s /bin/bash -g "${LDAP_GROUPNAME}" -u "${LDAP_UID}" "${LDAP_USERNAME}"

WORKDIR /home/${LDAP_USERNAME}

#####################################
# 2) Copy only requirements.txt first (to leverage Docker cache)
#####################################
#    If requirements.txt doesn't change, this layer is cached and 
#    we skip re-installing Python packages on every code update.
COPY requirements.txt /home/${LDAP_USERNAME}/requirements.txt

# 3) As root: install Python dependencies
USER root
RUN pip install -r requirements.txt

#####################################
# 4) Copy the rest of your application code
#####################################
COPY --chown=${LDAP_USERNAME}:${LDAP_GROUPNAME} . /home/${LDAP_USERNAME}/

#####################################
# 5) Switch to the unprivileged EPFL user
#####################################
USER ${LDAP_USERNAME}
WORKDIR /home/${LDAP_USERNAME}

#####################################
# 6) ENTRYPOINT to your training script
#####################################
ENTRYPOINT ["python3", "finetune_lora.py"]
CMD ["--help"]
