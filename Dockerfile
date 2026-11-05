# RunPod serverless worker: NVIDIA RE-USE speech enhancement for /voice-enhancer «Максимальная очистка».
#
# Engine chosen by measurement (05.11.2026, docs/voice-enhancer-meter.md): RE-USE beat the in-browser chain
# (DeepFilterNet 3) on every degradation class of our corpora at 48 kHz and keeps the voice's highs up to 20 kHz
# within ±1.4 dB (VCTK). It needs CUDA: mamba_ssm's fused kernels; on CPU it runs 8–36 s per audio second.
#
# Base = the RunPod PyTorch image the measurement ran on: the mamba-ssm / causal-conv1d wheels below resolve to
# prebuilt CUDA 12.4 / torch 2.4 / cp311 wheels there (no compile), and RunPod hosts often have it cached, which
# shortens a cold start.
#
# Licence: model weights and code — OpenMDW-1.1 (NVIDIA); the LICENSE file ships in the image (/models/reuse).
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/models/reuse \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-build-isolation -r requirements.txt \
    && python -c "from mamba_ssm import Mamba; import causal_conv1d; print('mamba_ssm ok')"

# Weights + NVIDIA's model code (models/, utils/, config.json, LICENSE) pinned to one revision and baked in:
# a cold start pulls the image, it never downloads a checkpoint.
COPY download_model.py .
RUN python download_model.py

COPY reuse_io.py handler.py io_selftest.py ./

CMD ["python", "-u", "handler.py"]
