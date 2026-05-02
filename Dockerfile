# Default: CUDA 12.6 + PyTorch cu126 (works across a wide range of NVIDIA GPUs on supported drivers).
# Optional — CUDA 12.8 + cu128 (some PyTorch cu128 wheels drop support for older architectures):
#   docker build --build-arg CUDA_IMAGE_TAG=12.8.0-cudnn-runtime-ubuntu24.04 \
#     --build-arg PYTORCH_CUDA=cu128 \
#     --build-arg FLASH_ATTN_URL=https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.4/flash_attn-2.8.3%2Bcu128torch2.11-cp313-cp313-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl \
#     -t vibevoice-realtime-openai-api:cu128 .
ARG CUDA_IMAGE_TAG=12.6.3-cudnn-runtime-ubuntu24.04
FROM nvidia/cuda:${CUDA_IMAGE_TAG}

# Install system packages
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    sudo git curl ffmpeg ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Create ubuntu user (UID/GID 1000 for volume compatibility)
RUN (getent group 1000 || groupadd -g 1000 ubuntu) && \
    (getent passwd 1000 || useradd -m -s /bin/bash -u 1000 -g 1000 ubuntu) && \
    echo "ubuntu ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/ubuntu && \
    chmod 0440 /etc/sudoers.d/ubuntu && \
    usermod -aG video ubuntu && \
    chown -R ubuntu:ubuntu /home/ubuntu

# Switch to ubuntu user
USER ubuntu
WORKDIR /home/ubuntu/app

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Append to .bashrc (for interactive sessions)
RUN cat >> /home/ubuntu/.bashrc << 'EOF'

# Environment setup
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
EOF

RUN touch /home/ubuntu/.sudo_as_admin_successful

# Set ENV for non-interactive CMD (HOME must be set; BuildKit does not assume a login user)
ENV CUDA_HOME=/usr/local/cuda
ENV HOME=/home/ubuntu
ENV PATH=$CUDA_HOME/bin:$HOME/.local/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Install Python 3.13
RUN /home/ubuntu/.local/bin/uv python install 3.13

ARG PYTORCH_CUDA=cu126
ENV UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/${PYTORCH_CUDA}

# Only dependency manifest before the heavy layer — keeps cache when app/UI code changes
COPY --chown=ubuntu:ubuntu requirements.txt .

ARG FLASH_ATTN_URL=https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.4/flash_attn-2.8.3%2Bcu126torch2.11-cp313-cp313-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl

# Save wheel with a valid PEP 427 filename (uv rejects names like flash_attn-local.whl — no py tag)
RUN mkdir -p prebuilt-wheels && \
    FN=$(echo "${FLASH_ATTN_URL}" | sed 's#.*/##;s/?.*//;s/%2B/+/g') && \
    curl -fL -o "prebuilt-wheels/${FN}" "${FLASH_ATTN_URL}" && \
    /home/ubuntu/.local/bin/uv venv .venv --python 3.13 --seed && \
    . .venv/bin/activate && \
    /home/ubuntu/.local/bin/uv pip install -r requirements.txt && \
    /home/ubuntu/.local/bin/uv pip install "prebuilt-wheels/${FN}" && \
    rm -rf ./prebuilt-wheels && \
    /home/ubuntu/.local/bin/uv cache clean

# Application code last — rebuild from here on Python / UI / entrypoint changes only
COPY --chown=ubuntu:ubuntu entrypoint.sh .
COPY --chown=ubuntu:ubuntu vibevoice_realtime_openai_api.py .
COPY --chown=ubuntu:ubuntu static ./static

RUN chmod +x entrypoint.sh

ENV CFG_SCALE=1.25
ENV MODELS_DIR=/home/ubuntu/app/models

VOLUME /home/ubuntu/app/models

EXPOSE 8880

CMD ["./entrypoint.sh"]
