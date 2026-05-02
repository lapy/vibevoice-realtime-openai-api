# VibeVoice Realtime 0.5B OpenAI-Compatible TTS Server

OpenAI-compatible TTS API wrapping [VibeVoice-Realtime-0.5B](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B) for Open WebUI.

![image](assets/openwebui_settings.png)

> **Note**: If both this wrapper and Open WebUI runs in a container, use `host.docker.internal:8880` instead of `localhost`.

[![Demo: VibeVoice-Realtime OpenAI API-compatible Text-to-Speech Server for Open WebUI](https://i3.ytimg.com/vi/12VwN-AM1os/maxresdefault.jpg)](https://youtu.be/12VwN-AM1os)

> 👆🏻 📹 YouTube video demonstration of "Mike" vocal used on Open WebUI. 📹 👆🏻

## Features

- ✅ **OpenAI API Compatible**
  - `/v1/models` (model discovery; some clients call this instead of `/v1/audio/models`)
  - `/v1/audio/speech`
  - `/v1/audio/voices` (see note below — OpenAI’s docs do not define a standard “list voices” HTTP endpoint)
  - `/v1/audio/models`
  - Usable as a drop-in base URL for many OpenAI-compatible TTS clients.
  - OpenAI documents fixed TTS voice names for `tts-1` / `tts-1-hd`, but **no official REST route to list them**. **`GET /v1/audio/voices`** here returns installed Microsoft preset IDs for discovery.
- ⚡ **Real-time Performance** — About **~0.5× RTF** on an **RTX 3060** (varies by GPU).
- 🚀 **GPU Accelerated** — About **2 GB VRAM** typical; CUDA with Flash Attention (Docker) or SDPA.
- 🔊 **Voices** — Same stems as [Microsoft `streaming_model`](https://github.com/microsoft/VibeVoice/tree/main/demo/voices/streaming_model), plus short aliases (e.g. `Emma`, `de-spk0`) as in `GET /v1/audio/voices`. **No** OpenAI names like `alloy` / `nova`.
- 🎵 **Multiple Formats** — MP3, WAV, OPUS, FLAC, AAC, PCM.
- 🖥 **Browser UI** — `http://localhost:8880/` for text → speech with voice/format selection; **CFG scale** (0–3) can be saved to the server (`models/ui_settings.json`) and applies to all `/v1/audio/speech` requests that omit `cfg_scale`.

## Requirements

- Python 3.13 (via uv) / Docker with NVIDIA GPU support
- NVIDIA GPU: drivers compatible with **CUDA 12.6+** for the default Docker image and local `cu126` installs. See [CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/index.html).
- ffmpeg

---

## Option 1: Docker (Recommended)

Best performance with Flash Attention pre-installed.

- **Default image:** CUDA **12.6.3** runtime + PyTorch **`cu126`** ([`12.6.3-cudnn-runtime-ubuntu24.04`](https://hub.docker.com/r/nvidia/cuda/tags)) — broad **NVIDIA GPU** coverage on a sufficient driver. `flash-attn` wheel matches `cu126`.
- **Python 3.13** via uv
- **Optional:** rebuild with `CUDA_IMAGE_TAG=12.8.0-…`, `PYTORCH_CUDA=cu128`, and the `cu128` flash-attn URL for CUDA **12.8** wheels — **confirm** PyTorch’s `cu128` build targets your GPU (some lines drop older architectures). See comment block at top of `Dockerfile`.

```bash
git clone https://github.com/marhensa/vibevoice-realtime-openai-api.git
cd vibevoice-realtime-openai-api

# Using docker-compose (recommended)
docker compose up -d --build

# Or manual build/run (default: CUDA 12.6 + cu126)
docker build -t vibevoice-realtime-openai-api .
docker run --gpus all -p 8880:8880 \
  -v ./models:/home/ubuntu/app/models \
  -e CFG_SCALE=1.25 \
  vibevoice-realtime-openai-api
```

> ⚠️ **Please be patient** and check your network monitor, because on first run it downloads models 📦 (~2GB) and voice presets 🎤 (~22MB) from huggingface and Microsoft VibeVoice repositories to `./models/`. It's not stuck, it's just downloading.
>
> **Browser UI:** open `http://localhost:8880/` (compose bind-mounts `./static` for quick UI tweaks; rebuild the image for Python or dependency changes).

---

## Option 2: Python venv

Requires Python 3.13 and a recent NVIDIA driver. Default local install uses PyTorch **`cu126`** (same as Docker). You can use **`cu128`** instead if you need CUDA 12.8 wheels — **verify** `torch.cuda.get_arch_list()` (or PyTorch docs) includes your GPU.

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

# Install dependencies (default: cu126, same as Docker)
$env:UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu126"; uv pip install -r requirements.txt

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

# Install dependencies (default: cu126, same as Docker)
export UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu126
uv pip install -r requirements.txt

# Download and install prebuilt Flash Attention (must match CUDA wheel index: cu126 by default)
mkdir -p prebuilt-wheels
curl -L -o ./prebuilt-wheels/flash_attn-local.whl \
  "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.4/flash_attn-2.8.3%2Bcu126torch2.11-cp313-cp313-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl"
uv pip install ./prebuilt-wheels/flash_attn-local.whl

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
| TTS Voice | Canonical stem (e.g. `en-Emma_woman`) or short alias (e.g. `Emma`, `de-spk0`); see [Available Voices](#available-voices) |
| Response splitting | `Paragraph` (recommended for low-end GPU) |

> **Note**: If both this wrapper and Open WebUI runs in a container, use `host.docker.internal:8880` instead of `localhost`.

## Available Voices

**Canonical voice ID** is the **`.pt` filename without the extension** from Microsoft’s repo: [demo/voices/streaming_model](https://github.com/microsoft/VibeVoice/tree/main/demo/voices/streaming_model). The server downloads these presets on first run. **`GET /v1/audio/voices`** returns each voice’s canonical `voice_id`, optional **`aliases`** (short names), and metadata.

Use either the **canonical stem** or any **alias** (case-insensitive) in the JSON `voice` field of `POST /v1/audio/speech`. There is **no** mapping from OpenAI names (`alloy`, `nova`, …).

| Voice ID (canonical) | Aliases | Locale | Gender |
|------------------------|---------|--------|--------|
| `de-Spk0_man` | `de-spk0` | German (`de`) | male |
| `de-Spk1_woman` | `de-spk1` | German (`de`) | female |
| `en-Carter_man` | `carter` | English (`en`) | male |
| `en-Davis_man` | `davis` | English (`en`) | male |
| `en-Emma_woman` | `emma` | English (`en`) | female |
| `en-Frank_man` | `frank` | English (`en`) | male |
| `en-Grace_woman` | `grace` | English (`en`) | female |
| `en-Mike_man` | `mike` | English (`en`) | male |
| `fr-Spk0_man` | `fr-spk0` | French (`fr`) | male |
| `fr-Spk1_woman` | `fr-spk1` | French (`fr`) | female |
| `in-Samuel_man` | `samuel` | Indian English (`in`) | male |
| `it-Spk0_woman` | `it-spk0` | Italian (`it`) | female |
| `it-Spk1_man` | `it-spk1` | Italian (`it`) | male |
| `jp-Spk0_man` | `jp-spk0` | Japanese (`jp`) | male |
| `jp-Spk1_woman` | `jp-spk1` | Japanese (`jp`) | female |
| `kr-Spk0_woman` | `kr-spk0` | Korean (`kr`) | female |
| `kr-Spk1_man` | `kr-spk1` | Korean (`kr`) | male |
| `nl-Spk0_man` | `nl-spk0` | Dutch (`nl`) | male |
| `nl-Spk1_woman` | `nl-spk1` | Dutch (`nl`) | female |
| `pl-Spk0_man` | `pl-spk0` | Polish (`pl`) | male |
| `pl-Spk1_woman` | `pl-spk1` | Polish (`pl`) | female |
| `pt-Spk0_woman` | `pt-spk0` | Portuguese (`pt`) | female |
| `pt-Spk1_man` | `pt-spk1` | Portuguese (`pt`) | male |
| `sp-Spk0_woman` | `sp-spk0` | Spanish (`sp`) | female |
| `sp-Spk1_man` | `sp-spk1` | Spanish (`sp`) | male |

### Custom Voices / Additional Voices

If there's any updated voices, you can download them from [here](https://github.com/microsoft/VibeVoice/tree/main/demo/voices/streaming_model).

You can add custom / additional voices by placing `.pt` files in `./models/voices/`. The server scans this directory on startup.

> **Note**: The Realtime 0.5B model does not provide public voice cloning tools. For custom voice creation, [contact Microsoft](https://github.com/microsoft/VibeVoice). Microsoft plans to expand available speakers in future updates.

## TTS quality

What you hear depends on **diffusion steps**, **CFG**, whether you **stream**, **output format**, and **GPU** headroom.

| Control | Where | Notes |
|--------|--------|--------|
| **DDPM steps** | `ddpm_steps` in `models/ui_settings.json` (default for all requests), or optional **`ddpm_steps`** in `POST /v1/audio/speech` | Higher values (e.g. 10–20) often improve clarity at the cost of slower synthesis. **Per-request** `ddpm_steps` is applied under a **global lock** (other clients wait; use for A/B or rare high-quality runs, not high QPS). |
| **CFG scale** | `cfg_scale` in settings, or optional **`cfg_scale`** in the JSON body | Typical range **~1.2–1.6**; very high values can destabilize. Omitted field → saved server default. |
| **Streaming** | `stream: true` or `stream_format: "audio"` | **Lower first-byte latency**; path uses light **causal** filtering only. **Non-streaming** applies full **`enhance_tts_audio`** after generation (band shaping, optional light DNS, peak, fades). For **final assets**, prefer non-stream when latency allows. |
| **Full enhance on stream** | **`vibevoice_post_enhance_stream`: true** (streaming only) | Server buffers the full utterance, runs the same **offline-style** enhance as non-stream, then sends PCM. **Not low-latency**—use when you need streaming API shape but want closer-to-offline cleanup. |
| **Format** | `response_format` | **WAV/FLAC** are lossless; **MP3/AAC** add encoder flavor (often negligible at high bitrate). |

---

## API

```bash
# Health check
curl http://localhost:8880/health

# List voices
curl http://localhost:8880/v1/audio/voices

# Optional: CFG in JSON (per request); omit to use value from models/ui_settings.json or CFG_SCALE env
curl -X POST http://localhost:8880/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello", "voice": "Emma", "response_format": "mp3", "cfg_scale": 1.25}' \
  --output speech.mp3

# Optional: per-request DDPM steps (serialized; other in-flight overrides wait)
curl -X POST http://localhost:8880/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello", "voice": "Emma", "response_format": "wav", "ddpm_steps": 18}' \
  --output speech.wav

# Streaming with full offline-style enhancement after the utterance completes (high latency)
curl -X POST http://localhost:8880/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello", "voice": "Emma", "response_format": "pcm", "stream": true, "vibevoice_post_enhance_stream": true}' \
  --output speech.pcm
```

```powershell
# Generate speech (PowerShell)
Invoke-RestMethod -Uri "http://localhost:8880/v1/audio/speech" `
  -Method Post -ContentType "application/json" `
  -Body '{"input": "Welcome to VibeVoice! This is real-time text to speech, powered by Microsoft research.", "voice": "Emma"}' `
  -OutFile "speech.mp3"
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODELS_DIR` | `./models` | Path to models directory |
| `VIBEVOICE_DEVICE` | `cuda` | Torch device: `cuda` (first NVIDIA GPU), `cuda:N` (e.g. `cuda:2` for the third GPU), `cpu`, or `mps` |
| `VIBEVOICE_USE_FLASH_ATTN` | (unset) | Set to `1` to try **Flash Attention 2** first on CUDA. On some **older NVIDIA architectures** (e.g. compute capability **7.0**) the default is **SDPA** for reliability. |
| `CFG_SCALE` | `1.25` | Initial CFG (0–3) if `models/ui_settings.json` is absent; overridden by that file after first save from the UI or `PUT /api/ui/settings` |

`ddpm_steps` in `ui_settings.json` is the default DDPM count at startup and after **Save** in the UI. Optional JSON field **`ddpm_steps`** on `/v1/audio/speech` overrides for **one** request (global mutex + restore); concurrent throughput drops because overrides cannot overlap safely on a single model instance.

## Troubleshooting (common log lines)

- **APEX FusedRMSNorm not available** — VibeVoice uses native PyTorch RMSNorm when [NVIDIA Apex](https://github.com/NVIDIA/apex) is not installed; behavior is correct, only a possible small speed difference.
- **Tokenizer class … Qwen2Tokenizer … VibeVoiceTextTokenizerFast** — Hugging Face warns when the checkpoint’s tokenizer type differs from the class used to load it; usually harmless for this model.
- **Error in cpuinfo / /proc/cpuinfo** — Often seen in containers with restricted CPU visibility; unrelated to GPU inference.
- **Wrong PyTorch CUDA wheel / “no kernel for this GPU”** — Prefer **`cu126`** installs (`UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu126`); the default **Dockerfile** uses **CUDA 12.6 + cu126**. Some **`cu128`** PyTorch builds omit older NVIDIA architectures. On startup we **default to SDPA** on capability **7.0** unless `VIBEVOICE_USE_FLASH_ATTN=1`, and we **fail fast** if `torch.cuda.get_arch_list()` omits your GPU’s `sm_XX`.
- **`RuntimeError: cutlassF: no kernel found to launch!`** — Common on **pre-Ampere** NVIDIA GPUs with **bfloat16** and PyTorch’s **memory-efficient SDPA**. This server **disables flash/mem-efficient SDPA** and loads in **float32** on those GPUs automatically (Ampere+ still uses **bfloat16**).

## Container registry (GitHub Actions)

Pushes to the repository default branch and version tags `v*` build and publish a **linux/amd64** image to GitHub Container Registry:

`ghcr.io/<owner>/<repo>:latest` (default branch) and `ghcr.io/<owner>/<repo>:<git-sha>`.

Forks and pull requests run the same Docker build without pushing. Make the package public or grant pull access under **Packages** in the repo settings if needed.

## License

- [VibeVoice](https://github.com/microsoft/VibeVoice) (code + model): MIT License (Microsoft)
- [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) (base LLM): Apache 2.0 (Alibaba)
- This wrapper: MIT License
