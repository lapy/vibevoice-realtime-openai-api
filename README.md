# VibeVoice Realtime 0.5B OpenAI-Compatible TTS Server

OpenAI-compatible TTS API wrapping [VibeVoice-Realtime-0.5B](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B) for Open WebUI.

![image](assets/openwebui_settings.png)

> **Note**: If both this wrapper and Open WebUI runs in a container, use `host.docker.internal:8880` instead of `localhost`.

[![Demo: VibeVoice-Realtime OpenAI API-compatible Text-to-Speech Server for Open WebUI](https://i3.ytimg.com/vi/12VwN-AM1os/maxresdefault.jpg)](https://youtu.be/12VwN-AM1os)

> 👆🏻 📹 YouTube video demonstration of "Mike" vocal used on Open WebUI. 📹 👆🏻

## Features

- ✅ **OpenAI API Compatible**
  - `/v1/audio/speech`
  - `/v1/audio/voices`
  - `/v1/audio/models`
  - Can be used on many OpenAI API-compatible drop-in endpoint.
- ⚡ **Real-time Performance** - **~0.5x RTF** (Real-Time Factor) on an **RTX 3060**.
- 🚀 **GPU Accelerated** - Requiring **only ~2GB of VRAM**, CUDA with Flash Attention (Docker) or SDPA
- 🔊 **7 Voices** - With OpenAI voice name aliases (alloy, nova, etc.)
- 🎵 **Multiple Formats** - MP3, WAV, OPUS, FLAC, AAC, PCM
- 📦 **Self-contained** - Models download to `./models/` on first run

## Requirements

- Python 3.13 (via uv) / Docker with NVIDIA GPU support
- NVIDIA GPU with a driver that supports **CUDA 12.8** (see [CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/index.html))
- ffmpeg

---

## Option 1: Docker (Recommended)

Best performance with Flash Attention pre-installed.

- **CUDA 12.8.0** runtime (NVIDIA [`12.8.0-cudnn-runtime-ubuntu24.04`](https://hub.docker.com/r/nvidia/cuda/tags))
- **Python 3.13** via uv
- **Prebuilt wheel**: flash-attn (downloaded during build; matches PyTorch 2.11 + cu128)

```bash
git clone https://github.com/marhensa/vibevoice-realtime-openai-api.git
cd vibevoice-realtime-openai-api

# Using docker-compose (recommended)
docker compose up -d --build

# Or manual build/run
docker build -t vibevoice-realtime-openai-api .
docker run --gpus all -p 8880:8880 \
  -v ./models:/home/ubuntu/app/models \
  -e CFG_SCALE=1.25 \
  vibevoice-realtime-openai-api
```

> ⚠️ **Please be patient** and check your network monitor, because on first run it downloads models 📦 (~2GB) and voice presets 🎤 (~22MB) from huggingface and Microsoft VibeVoice repositories to `./models/`. It's not stuck, it's just downloading.

---

## Option 2: Python venv

Requires Python 3.13 and an NVIDIA driver compatible with CUDA 12.8.

### Windows

```powershell
winget install --id Gyan.FFmpeg

git clone https://github.com/marhensa/vibevoice-realtime-openai-api.git
cd vibevoice-realtime-openai-api

# Install uv
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Create venv
uv venv .venv --python 3.13 --seed
.venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt

# Run (optional: set CFG_SCALE for expressiveness, 0.0-3.0)
$env:CFG_SCALE="1.25"; python vibevoice_realtime_openai_api.py --port 8880
```

### Linux

```bash
sudo apt install ffmpeg

git clone https://github.com/marhensa/vibevoice-realtime-openai-api.git
cd vibevoice-realtime-openai-api

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv
uv venv .venv --python 3.13 --seed
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Download and install prebuilt Flash Attention (Linux x86_64; matches torch 2.11 + CUDA 12.8)
mkdir -p prebuilt-wheels
curl -L -o ./prebuilt-wheels/flash_attn-2.8.3+cu128torch2.11-cp313-cp313-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl \
  "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.4/flash_attn-2.8.3%2Bcu128torch2.11-cp313-cp313-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl"
uv pip install ./prebuilt-wheels/flash_attn-*.whl

# Run (optional: set CFG_SCALE for expressiveness, 0.0-3.0)
CFG_SCALE=1.25 python vibevoice_realtime_openai_api.py --port 8880
```

First run downloads models (~2GB) and voice presets (~22MB) to `./models/`.

---

## Open WebUI Configuration

| Setting | Value |
|---------|-------|
| TTS Engine | OpenAI |
| API Base URL | `http://localhost:8880/v1` |
| API Key | `sk-unused` |
| TTS Model | `tts-1-hd` |
| TTS Voice | `Carter`, `Emma`, `alloy`, `nova`, etc. |
| Response splitting | `Paragraph` (recommended for low-end GPU) |

> **Note**: If both this wrapper and Open WebUI runs in a container, use `host.docker.internal:8880` instead of `localhost`.

## Available Voices

| OpenAI Name | VibeVoice Name | Gender |
|-------------|----------------|--------|
| alloy | Carter | Male |
| echo | Davis | Male |
| fable | Emma | Female |
| onyx | Frank | Male |
| nova | Grace | Female |
| shimmer | Mike | Male |
| - | Samuel | Male |

You can use either OpenAI names or VibeVoice names in the API.

### Custom Voices / Additional Voices

If there's any updated voices, you can download them from [here](https://github.com/microsoft/VibeVoice/tree/main/demo/voices/streaming_model).

You can add custom / additional voices by placing `.pt` files in `./models/voices/`. The server scans this directory on startup.

> **Note**: The Realtime 0.5B model does not provide public voice cloning tools. For custom voice creation, [contact Microsoft](https://github.com/microsoft/VibeVoice). Microsoft plans to expand available speakers in future updates.

## API

```bash
# Health check
curl http://localhost:8880/health

# List voices
curl http://localhost:8880/v1/audio/voices
```

```powershell
# Generate speech (PowerShell)
Invoke-RestMethod -Uri "http://localhost:8880/v1/audio/speech" `
  -Method Post -ContentType "application/json" `
  -Body '{"input": "Welcome to VibeVoice! This is real-time text to speech, powered by Microsoft research.", "voice": "Emma"}' `
  -OutFile "speech.mp3"
```

```bash
# Generate speech (bash/Linux)
curl -X POST http://localhost:8880/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Welcome to VibeVoice! This is real-time text to speech, powered by Microsoft research.", "voice": "Emma"}' \
  --output speech.mp3
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODELS_DIR` | `./models` | Path to models directory |
| `VIBEVOICE_DEVICE` | `cuda` | Device: `cuda` (NVIDIA GPUs), `cpu`, or `mps` (Apple Silicon GPUs) |
| `CFG_SCALE` | `1.25` | CFG guidance scale (0.0-3.0, higher = more expressive) |
## Container registry (GitHub Actions)

Pushes to the repository default branch and version tags `v*` build and publish a **linux/amd64** image to GitHub Container Registry:

`ghcr.io/<owner>/<repo>:latest` (default branch) and `ghcr.io/<owner>/<repo>:<git-sha>`.

Forks and pull requests run the same Docker build without pushing. Make the package public or grant pull access under **Packages** in the repo settings if needed.

## License

- [VibeVoice](https://github.com/microsoft/VibeVoice) (code + model): MIT License (Microsoft)
- [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) (base LLM): Apache 2.0 (Alibaba)
- This wrapper: MIT License
