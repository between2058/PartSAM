# =============================================================================
# PartSAM FastAPI Microservice — Docker Image
#
# Target hardware : NVIDIA RTX PRO 6000 (Blackwell, sm_120)
# CUDA toolkit    : 12.8.1
# cuDNN           : 9
# Python          : 3.11
# PyTorch         : 2.7.1 + cu128
#
# Build:
#   docker build -t partsam:latest .
#
# Run:
#   docker run --gpus all -p 8000:8000 partsam:latest
# =============================================================================

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

# -- Build-time arguments ------------------------------------------------------
# sm_120 = RTX PRO 6000 (Blackwell). Add more archs for multi-GPU compat.
ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0;12.0"
ARG MAX_JOBS=4

ARG http_proxy=""
ARG https_proxy=""
ARG no_proxy="localhost,127.0.0.1"

# -- Environment variables -----------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_HOME=/usr/local/cuda-12.8 \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    MAX_JOBS=${MAX_JOBS} \
    HF_HOME=/home/partsam/.cache/huggingface \
    http_proxy=${http_proxy} \
    https_proxy=${https_proxy} \
    HTTP_PROXY=${http_proxy} \
    HTTPS_PROXY=${https_proxy} \
    no_proxy=${no_proxy} \
    NO_PROXY=${no_proxy}

# -- System packages -----------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    build-essential \
    ninja-build \
    cmake \
    git \
    wget \
    curl \
    libx11-6 \
    libgl1 \
    libxrender1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# -- Python 3.11 as default ----------------------------------------------------
RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
 && python -m pip install --upgrade --no-cache-dir pip setuptools wheel

# -- Ensure CUDA_HOME symlink exists -------------------------------------------
RUN test -d /usr/local/cuda-12.8 \
 || ln -sf /usr/local/cuda /usr/local/cuda-12.8

# -- Configure git proxy (corporate networks may block direct GitHub access) ----
RUN if [ -n "$http_proxy" ]; then \
      git config --global http.proxy "$http_proxy" && \
      git config --global https.proxy "$https_proxy"; \
    fi

# -- Create non-root user ------------------------------------------------------
RUN groupadd -r partsam && useradd -r -g partsam -m -s /bin/bash partsam

WORKDIR /app

# =============================================================================
# STEP 1 — PyTorch 2.7.1 + CUDA 12.8 (sm_120 / Blackwell support)
# =============================================================================
RUN pip install --no-cache-dir \
    torch==2.7.1 \
    torchvision==0.22.1 \
    torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128

# =============================================================================
# STEP 2 — Pure-Python dependencies
# =============================================================================
RUN pip install --no-cache-dir \
    lightning==2.2 \
    h5py yacs trimesh scikit-image loguru boto3 \
    plyfile einops simple_parsing safetensors \
    hydra-core omegaconf accelerate timm igraph ninja \
    huggingface_hub \
    fastapi uvicorn[standard] pydantic-settings python-multipart

# =============================================================================
# STEP 3 — Lock torch ABI before compiling CUDA extensions
#
# Some deps above may pull in CPU-only torch. Force-reinstall to lock ABI.
# =============================================================================
RUN pip install --no-cache-dir --force-reinstall \
    torch==2.7.1 \
    torchvision==0.22.1 \
    torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128 \
 && pip install --no-cache-dir "numpy<2.0"

# =============================================================================
# STEP 4 — torch-scatter (pre-built wheel for PyTorch 2.7.1 + CUDA 12.8)
# =============================================================================
RUN pip install --no-cache-dir \
    torch-scatter -f https://data.pyg.org/whl/torch-2.7.1+cu128.html \
 || pip install --no-cache-dir --no-build-isolation torch-scatter

# =============================================================================
# STEP 5 — NVIDIA apex (fused layer norm)
# =============================================================================
RUN git clone https://github.com/NVIDIA/apex.git /tmp/apex \
 && cd /tmp/apex \
 && pip install --no-cache-dir --no-build-isolation \
    -v --global-option="--cpp_ext" --global-option="--cuda_ext" . \
 || pip install --no-cache-dir -v --no-build-isolation . \
 && rm -rf /tmp/apex

# =============================================================================
# STEP 6 — pointops (from Pointcept)
# =============================================================================
RUN git clone https://github.com/Pointcept/Pointcept.git /tmp/pointcept \
 && cd /tmp/pointcept/libs/pointops \
 && python setup.py install \
 && rm -rf /tmp/pointcept

# =============================================================================
# STEP 7 — torkit3d (from Point-SAM dependency)
# =============================================================================
RUN git clone https://github.com/vivym/torkit3d.git /tmp/torkit3d \
 && cd /tmp/torkit3d \
 && pip install --no-cache-dir --no-build-isolation . \
 && rm -rf /tmp/torkit3d

# =============================================================================
# STEP 8 — mesh2sdf (CUDA extension)
# =============================================================================
RUN pip install --no-cache-dir pybind11 \
 && pip install --no-cache-dir --no-build-isolation mesh2sdf==1.1.0

# =============================================================================
# STEP 9 — Download model weights from HuggingFace
# =============================================================================
RUN mkdir -p /app/pretrained \
 && python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='Czvvd/PartSAM', local_dir='/app/pretrained')" \
 && chown -R partsam:partsam /app/pretrained

# =============================================================================
# STEP 10 — Copy application source
# =============================================================================
COPY --chown=partsam:partsam PartSAM/    /app/PartSAM/
COPY --chown=partsam:partsam partfield/  /app/partfield/
COPY --chown=partsam:partsam utils/      /app/utils/
COPY --chown=partsam:partsam configs/    /app/configs/
COPY --chown=partsam:partsam service/    /app/service/

# -- Writable directories for non-root user ------------------------------------
RUN mkdir -p /app/results /app/logs \
 && chown -R partsam:partsam /app /home/partsam

# -- Switch to non-root user ---------------------------------------------------
USER partsam

# -- Port ----------------------------------------------------------------------
EXPOSE 8000

# -- Health check --------------------------------------------------------------
HEALTHCHECK \
    --interval=30s \
    --timeout=15s \
    --start-period=120s \
    --retries=5 \
    CMD curl -sf http://localhost:8000/health || exit 1

# -- Entrypoint ----------------------------------------------------------------
CMD ["python", "-m", "uvicorn", "service.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
