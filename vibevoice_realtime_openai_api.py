#!/usr/bin/env python3
"""
VibeVoice OpenAI-Compatible TTS Server

A FastAPI server that wraps VibeVoice-Realtime-0.5B with an OpenAI-compatible API,
enabling integration with Open WebUI and other OpenAI TTS-compatible applications.

Usage:
    python vibevoice_realtime_openai_api.py --port 8880
"""

import argparse
import copy
import io
import json
import os
import re
import subprocess
import threading
import time
import traceback
import unicodedata
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

# Set HuggingFace cache BEFORE importing any HF libraries
# Only use HF_HOME (TRANSFORMERS_CACHE is deprecated in v5)
# MODELS_DIR can be overridden via env var for Docker volume mounts
MODELS_DIR = Path(os.environ.get("MODELS_DIR", Path(__file__).parent / "models"))
os.environ["HF_HOME"] = str(MODELS_DIR / "huggingface")

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
import uvicorn
import scipy.io.wavfile as wavfile

# VibeVoice imports (after setting HF_HOME)
from vibevoice.modular.modeling_vibevoice_streaming_inference import (
    VibeVoiceStreamingForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_streaming_processor import (
    VibeVoiceStreamingProcessor,
)

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

SAMPLE_RATE = 24000
DEFAULT_MODEL_PATH = "microsoft/VibeVoice-Realtime-0.5B"

# CFG scale for generation (configurable via env var; overridable by persisted UI settings)
_CFG_SCALE_ENV = float(os.environ.get("CFG_SCALE", "1.25"))
CFG_SCALE = max(0.0, min(3.0, _CFG_SCALE_ENV))

_CFG_PERSIST_LOCK = threading.Lock()
_runtime_cfg_scale: float = CFG_SCALE

STATIC_DIR = Path(__file__).resolve().parent / "static"
UI_SETTINGS_FILENAME = "ui_settings.json"


def _ui_settings_path() -> Path:
    return MODELS_DIR / UI_SETTINGS_FILENAME


def load_persisted_cfg_scale() -> None:
    """Apply `CFG_SCALE` from env, then override from `MODELS_DIR/ui_settings.json` if valid."""
    global _runtime_cfg_scale
    base = max(0.0, min(3.0, float(os.environ.get("CFG_SCALE", "1.25"))))
    path = _ui_settings_path()
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            v = float(data["cfg_scale"])
            _runtime_cfg_scale = max(0.0, min(3.0, v))
            return
        except (KeyError, ValueError, TypeError, json.JSONDecodeError, OSError):
            pass
    _runtime_cfg_scale = base


def get_runtime_cfg_scale() -> float:
    return _runtime_cfg_scale


def set_runtime_cfg_scale(value: float) -> float:
    """Persist CFG scale to disk and use it for requests that omit ``cfg_scale``."""
    global _runtime_cfg_scale
    v = max(0.0, min(3.0, float(value)))
    with _CFG_PERSIST_LOCK:
        _runtime_cfg_scale = v
        path = _ui_settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"cfg_scale": v}, indent=2) + "\n", encoding="utf-8")
    return v


def sanitize_tts_input(text: str) -> str:
    """Drop everything that is not letters, combining marks, numbers, punctuation, or whitespace.

    Removes emoji, symbols, control/format characters, private-use code points, etc.
    Line/paragraph breaks and tabs collapse to a single ASCII space; runs of spaces merge.
    """
    out: List[str] = []
    prev_space = False
    for ch in text:
        if ch in "\n\r\t\v\f":
            ch = " "
        cat = unicodedata.category(ch)
        major = cat[0]
        if major == "Z":
            if not prev_space:
                out.append(" ")
                prev_space = True
            continue
        if major in ("L", "M", "N", "P"):
            out.append(ch)
            prev_space = False
    return "".join(out).strip()


# Voices directory
VOICES_DIR = MODELS_DIR / "voices"

# Voice preset files from:
# https://github.com/microsoft/VibeVoice/tree/main/demo/voices/streaming_model
STREAMING_MODEL_VOICE_FILES: List[str] = [
    "de-Spk0_man.pt",
    "de-Spk1_woman.pt",
    "en-Carter_man.pt",
    "en-Davis_man.pt",
    "en-Emma_woman.pt",
    "en-Frank_man.pt",
    "en-Grace_woman.pt",
    "en-Mike_man.pt",
    "fr-Spk0_man.pt",
    "fr-Spk1_woman.pt",
    "in-Samuel_man.pt",
    "it-Spk0_woman.pt",
    "it-Spk1_man.pt",
    "jp-Spk0_man.pt",
    "jp-Spk1_woman.pt",
    "kr-Spk0_woman.pt",
    "kr-Spk1_man.pt",
    "nl-Spk0_man.pt",
    "nl-Spk1_woman.pt",
    "pl-Spk0_man.pt",
    "pl-Spk1_woman.pt",
    "pt-Spk0_woman.pt",
    "pt-Spk1_man.pt",
    "sp-Spk0_woman.pt",
    "sp-Spk1_man.pt",
]

DEFAULT_VOICE_STEM = "en-Carter_man"
DEFAULT_VOICE_ALIAS = "Carter"

# GitHub raw URL for voice presets
VOICE_BASE_URL = "https://github.com/microsoft/VibeVoice/raw/main/demo/voices/streaming_model"


def _build_voice_aliases() -> Dict[str, str]:
    """Map short lowercase alias -> canonical Microsoft streaming_model stem.

    English speakers use given names (as in the legacy VOICE_PRESETS). Other locales use
    ``{lang}-spk0`` / ``{lang}-spk1`` matching Microsoft's Spk0/Spk1 filenames.
    """
    out: Dict[str, str] = {}
    named: List[tuple[str, str]] = [
        ("carter", "en-Carter_man"),
        ("davis", "en-Davis_man"),
        ("emma", "en-Emma_woman"),
        ("frank", "en-Frank_man"),
        ("grace", "en-Grace_woman"),
        ("mike", "en-Mike_man"),
        ("samuel", "in-Samuel_man"),
    ]
    for alias, stem in named:
        out[alias] = stem
    for fn in STREAMING_MODEL_VOICE_FILES:
        stem = Path(fn).stem
        m = re.match(r"^([a-z]{2})-(Spk[01])_(man|woman)$", stem, re.IGNORECASE)
        if m:
            lang = m.group(1).lower()
            spk = m.group(2).lower()
            out[f"{lang}-{spk}"] = stem
    return out


VOICE_ALIASES: Dict[str, str] = _build_voice_aliases()


def _aliases_for_stem(stem: str) -> List[str]:
    """Human-friendly aliases that resolve to this stem (sorted)."""
    return sorted(a for a, s in VOICE_ALIASES.items() if s == stem)

# Supported audio formats
SUPPORTED_FORMATS = ["mp3", "wav", "opus", "flac", "aac", "pcm"]

# ------------------------------------------------------------------------------
# Model Download Utilities
# ------------------------------------------------------------------------------

def ensure_voices_downloaded() -> None:
    """Download voice presets if not present"""
    VOICES_DIR.mkdir(parents=True, exist_ok=True)

    for filename in STREAMING_MODEL_VOICE_FILES:
        voice_path = VOICES_DIR / filename
        if not voice_path.exists():
            url = f"{VOICE_BASE_URL}/{filename}"
            print(f"[download] Downloading voice preset: {filename}...")
            try:
                urllib.request.urlretrieve(url, voice_path)
                print(f"[download] Downloaded {filename}")
            except Exception as e:
                print(f"[error] Failed to download {filename}: {e}")


def get_model_cache_dir() -> str:
    """Get model cache directory"""
    model_cache = MODELS_DIR / "huggingface"
    model_cache.mkdir(parents=True, exist_ok=True)
    return str(model_cache)


# ------------------------------------------------------------------------------
# Pydantic Models
# ------------------------------------------------------------------------------

class TTSRequest(BaseModel):
    """OpenAI-compatible TTS request"""
    model_config = ConfigDict(extra="ignore")

    input: str = Field(..., description="Text to synthesize", max_length=4096)
    voice: str = Field(
        default=DEFAULT_VOICE_ALIAS,
        description="Voice: canonical stem (e.g. en-Emma_woman) or short alias (e.g. Emma, de-spk0)",
    )
    model: str = Field(default="tts-1", description="Model ID (ignored, for compatibility)")
    response_format: str = Field(default="mp3", description="Audio format")
    speed: float = Field(default=1.0, description="Speed (not yet supported)")
    stream: bool = Field(default=False, description="Enable streaming response")
    cfg_scale: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=3.0,
        description="CFG guidance for this request only; omit to use persisted server default",
    )


class VoiceInfo(BaseModel):
    """Voice information"""
    voice_id: str
    name: str
    type: str
    gender: Optional[str] = None
    aliases: List[str] = Field(default_factory=list)


class VoicesResponse(BaseModel):
    """Response for /v1/audio/voices endpoint"""
    voices: List[VoiceInfo]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str
    model_loaded: bool
    device: str
    features: Dict[str, Any]


class UISettings(BaseModel):
    """Persisted web UI / default API settings"""
    cfg_scale: float = Field(ge=0.0, le=3.0, description="CFG guidance (higher = more expressive)")


# ------------------------------------------------------------------------------
# TTS Service
# ------------------------------------------------------------------------------

class VibeVoiceTTSService:
    """Service for managing VibeVoice model and generating speech"""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self.processor: Optional[VibeVoiceStreamingProcessor] = None
        self.model: Optional[VibeVoiceStreamingForConditionalGenerationInference] = None
        self.voice_presets: Dict[str, Path] = {}
        self._voice_cache: Dict[str, Any] = {}
        self._torch_device = torch.device(device)

    def load(self) -> None:
        """Load model and voice presets"""
        # Set HuggingFace cache to models folder
        os.environ["HF_HOME"] = get_model_cache_dir()

        # Download voice presets
        ensure_voices_downloaded()

        print(f"[startup] Loading processor from {self.model_path}")
        self.processor = VibeVoiceStreamingProcessor.from_pretrained(self.model_path)

        # Determine dtype and attention implementation based on device
        if self.device == "cuda":
            load_dtype = torch.bfloat16
            device_map = "cuda"
            attn_impl = "flash_attention_2"
        elif self.device == "mps":
            load_dtype = torch.float32
            device_map = None
            attn_impl = "sdpa"
        else:  # cpu
            load_dtype = torch.float32
            device_map = "cpu"
            attn_impl = "sdpa"

        print(f"[startup] Loading model with dtype={load_dtype}, attn={attn_impl}")

        try:
            self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                self.model_path,
                torch_dtype=load_dtype,
                device_map=device_map,
                attn_implementation=attn_impl,
            )
            if self.device == "mps":
                self.model.to("mps")
        except Exception as e:
            if attn_impl == "flash_attention_2":
                print(f"[startup] Flash Attention failed, falling back to SDPA: {e}")
                self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    device_map=device_map,
                    attn_implementation="sdpa",
                )
            else:
                raise

        self.model.eval()
        self.model.set_ddpm_inference_steps(num_steps=5)

        # Load voice presets
        self._load_voice_presets()
        print(f"[startup] Model ready on {self.device}")

    def _load_voice_presets(self) -> None:
        """Scan and load available voice presets"""
        if not VOICES_DIR.exists():
            print(f"[warning] Voices directory not found: {VOICES_DIR}")
            return

        for pt_file in sorted(VOICES_DIR.glob("*.pt")):
            stem = pt_file.stem
            self.voice_presets[stem] = pt_file

        print(f"[startup] Found {len(self.voice_presets)} voice preset files under {VOICES_DIR}")

    def get_available_voices(self) -> List[VoiceInfo]:
        """Microsoft streaming_model voices (stems match GitHub filenames without .pt)."""
        voices: List[VoiceInfo] = []
        for stem in sorted(self.voice_presets.keys()):
            path_stem = self.voice_presets[stem].stem
            gender = None
            if "_woman" in path_stem:
                gender = "female"
            elif "_man" in path_stem:
                gender = "male"
            voices.append(
                VoiceInfo(
                    voice_id=stem,
                    name=stem,
                    type="microsoft-streaming_model",
                    gender=gender,
                    aliases=_aliases_for_stem(stem),
                )
            )
        return voices

    def _resolve_voice(self, voice: str) -> str:
        """Resolve requested voice to a preset stem (Microsoft streaming_model ID)."""
        v = voice.strip()
        if v.lower().endswith(".pt"):
            v = v[:-3]
        vl = v.lower()
        if vl in VOICE_ALIASES:
            stem = VOICE_ALIASES[vl]
            if stem in self.voice_presets:
                return stem
        if v in self.voice_presets:
            return v
        for stem in self.voice_presets:
            if stem.lower() == vl:
                return stem
        available = sorted(self.voice_presets.keys())
        fallback = DEFAULT_VOICE_STEM if DEFAULT_VOICE_STEM in self.voice_presets else (available[0] if available else v)
        print(f"[warning] Voice '{voice}' not found, using '{fallback}'. Available stems: {available}")
        return fallback

    def _get_voice_prompt(self, voice: str) -> Any:
        """Load or get cached voice prompt"""
        if voice not in self._voice_cache:
            voice_path = self.voice_presets[voice]
            print(f"[tts] Loading voice prompt from {voice_path}")
            self._voice_cache[voice] = torch.load(
                voice_path,
                map_location=self._torch_device,
                weights_only=False
            )
        return self._voice_cache[voice]

    def generate_speech(self, text: str, voice: str, cfg_scale: float = 1.5) -> np.ndarray:
        """Generate speech from text

        Args:
            text: Text to synthesize
            voice: Voice name
            cfg_scale: CFG scale for generation

        Returns:
            Audio samples as numpy array (float32, 24kHz)
        """
        if not self.model or not self.processor:
            raise RuntimeError("Model not loaded")

        voice = self._resolve_voice(voice)
        prefilled_outputs = self._get_voice_prompt(voice)

        # Keep only text + punctuation; normalize smart quotes and whitespace
        text = sanitize_tts_input(text).replace("'", "'")

        # Prepare inputs
        inputs = self.processor.process_input_with_cached_prompt(
            text=text,
            cached_prompt=prefilled_outputs,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        # Move to device
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(self._torch_device)

        print(f"[tts] Generating speech for {len(text)} chars with voice '{voice}'")
        start_time = time.time()

        # Generate
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=None,
            cfg_scale=cfg_scale,
            tokenizer=self.processor.tokenizer,
            generation_config={"do_sample": False},
            verbose=False,
            all_prefilled_outputs=copy.deepcopy(prefilled_outputs),
        )

        elapsed = time.time() - start_time

        # Extract audio
        if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
            audio = outputs.speech_outputs[0]
            if torch.is_tensor(audio):
                audio = audio.detach().cpu().to(torch.float32).numpy()
            else:
                audio = np.asarray(audio, dtype=np.float32)

            if audio.ndim > 1:
                audio = audio.reshape(-1)

            # Normalize
            peak = np.max(np.abs(audio))
            if peak > 1.0:
                audio = audio / peak

            duration = len(audio) / SAMPLE_RATE
            rtf = elapsed / duration if duration > 0 else float("inf")
            print(f"[tts] Generated {duration:.2f}s audio in {elapsed:.2f}s (RTF: {rtf:.2f}x)")

            return audio
        else:
            raise RuntimeError("No audio output generated")


# ------------------------------------------------------------------------------
# Audio Format Conversion
# ------------------------------------------------------------------------------

def convert_audio(audio: np.ndarray, format: str, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Convert audio to specified format using ffmpeg

    Args:
        audio: Audio samples (float32, mono)
        format: Output format (mp3, wav, opus, flac, aac, pcm)
        sample_rate: Sample rate

    Returns:
        Audio bytes in specified format
    """
    format = format.lower()

    if format == "pcm":
        # Raw PCM16 little-endian
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        return pcm.tobytes()

    if format == "wav":
        # Use scipy for WAV
        buffer = io.BytesIO()
        wavfile.write(buffer, sample_rate, (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16))
        return buffer.getvalue()

    # Use ffmpeg for other formats
    # Prepare input WAV
    wav_buffer = io.BytesIO()
    wavfile.write(wav_buffer, sample_rate, (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16))
    wav_data = wav_buffer.getvalue()

    # ffmpeg format mappings
    format_args = {
        "mp3": ["-f", "mp3", "-codec:a", "libmp3lame", "-q:a", "2"],
        "opus": ["-f", "opus", "-codec:a", "libopus"],
        "flac": ["-f", "flac", "-codec:a", "flac"],
        "aac": ["-f", "adts", "-codec:a", "aac"],
    }

    if format not in format_args:
        raise ValueError(f"Unsupported format: {format}")

    # Run ffmpeg
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-f", "wav",
        "-i", "pipe:0",
        *format_args[format],
        "pipe:1"
    ]

    try:
        result = subprocess.run(
            cmd,
            input=wav_data,
            capture_output=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"[error] ffmpeg failed: {e.stderr.decode()}")
        raise RuntimeError(f"Audio conversion failed: {e}")
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found. Please install ffmpeg.")


def get_content_type(format: str) -> str:
    """Get MIME content type for audio format"""
    types = {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "opus": "audio/opus",
        "flac": "audio/flac",
        "aac": "audio/aac",
        "pcm": "audio/pcm",
    }
    return types.get(format.lower(), "application/octet-stream")


# ------------------------------------------------------------------------------
# FastAPI Application
# ------------------------------------------------------------------------------

# Global service instance
tts_service: Optional[VibeVoiceTTSService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager for startup and shutdown"""
    global tts_service

    # --- Startup ---
    model_path = os.environ.get("VIBEVOICE_MODEL_PATH", DEFAULT_MODEL_PATH)
    device = os.environ.get("VIBEVOICE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

    load_persisted_cfg_scale()

    tts_service = VibeVoiceTTSService(model_path=model_path, device=device)
    try:
        tts_service.load()
    except Exception as e:
        print(f"[FATAL] Model loading failed: {e}")
        traceback.print_exc()

    yield

    # --- Shutdown ---
    if tts_service and tts_service.model:
        del tts_service.model
        torch.cuda.empty_cache()


app = FastAPI(
    title="VibeVoice TTS Server",
    description="OpenAI-compatible TTS API powered by VibeVoice-Realtime-0.5B",
    version="1.0.0",
    lifespan=lifespan
)

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def ui_index():
    """Simple browser UI for speech generation and CFG scale."""
    path = STATIC_DIR / "index.html"
    if not path.is_file():
        return HTMLResponse(
            "<!DOCTYPE html><html><body><p>static/index.html is missing from the deployment.</p></body></html>",
            status_code=500,
        )
    return HTMLResponse(path.read_text(encoding="utf-8"))


@app.get("/api/ui/settings")
async def get_ui_settings():
    """Current persisted CFG scale (used when API requests omit ``cfg_scale``)."""
    return {"cfg_scale": get_runtime_cfg_scale()}


@app.put("/api/ui/settings")
async def put_ui_settings(settings: UISettings):
    """Persist CFG scale to disk and apply to all future API calls that omit ``cfg_scale``."""
    return {"cfg_scale": set_runtime_cfg_scale(settings.cfg_scale)}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="ok",
        service="vibevoice-realtime-openai-api",
        model_loaded=tts_service is not None and tts_service.model is not None,
        device=tts_service.device if tts_service else "unknown",
        features={
            "streaming": False,
            "formats": SUPPORTED_FORMATS,
            "sample_rate": SAMPLE_RATE,
        }
    )


@app.get("/v1/audio/voices", response_model=VoicesResponse)
async def list_voices():
    """List installed Microsoft streaming_model voices.

    OpenAI documents fixed TTS voice names but does not define a standard HTTP
    endpoint to list them; this route is a small extension for clients that need discovery.
    Each voice includes ``voice_id`` (canonical Microsoft stem) and ``aliases`` (short names
    such as ``Carter`` or ``de-spk0``).
    """
    if not tts_service:
        raise HTTPException(status_code=503, detail="Service not ready")

    return VoicesResponse(voices=tts_service.get_available_voices())


def _list_tts_models_payload() -> Dict[str, Any]:
    """Shared OpenAI-style model list for /v1/models and /v1/audio/models."""
    return {
        "object": "list",
        "data": [
            {
                "id": "tts-1",
                "object": "model",
                "created": 1699000000,
                "owned_by": "vibevoice",
                "name": "VibeVoice-Realtime-0.5B"
            },
            {
                "id": "tts-1-hd",
                "object": "model",
                "created": 1699000000,
                "owned_by": "vibevoice",
                "name": "VibeVoice-Realtime-0.5B"
            }
        ]
    }


@app.get("/v1/models")
async def list_openai_models():
    """List models (OpenAI-compatible; some clients call GET/LIST /v1/models)."""
    return _list_tts_models_payload()


@app.get("/v1/audio/models")
async def list_models():
    """List available TTS models (OpenAI-compatible)"""
    return _list_tts_models_payload()


@app.post("/v1/audio/speech")
async def create_speech(request: TTSRequest):
    """Generate speech from text (OpenAI-compatible)"""
    if not tts_service:
        raise HTTPException(status_code=503, detail="Service not ready")

    text_in = sanitize_tts_input(request.input)
    if not text_in:
        raise HTTPException(status_code=400, detail="Input text is required")

    if len(text_in) > 4096:
        raise HTTPException(status_code=400, detail="Input text exceeds 4096 characters")

    if request.response_format.lower() not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Supported: {SUPPORTED_FORMATS}"
        )

    try:
        effective_cfg = (
            request.cfg_scale if request.cfg_scale is not None else get_runtime_cfg_scale()
        )
        audio = tts_service.generate_speech(
            text=text_in,
            voice=request.voice,
            cfg_scale=effective_cfg,
        )

        # Convert to requested format
        audio_bytes = convert_audio(audio, request.response_format)
        content_type = get_content_type(request.response_format)

        return Response(
            content=audio_bytes,
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=speech.{request.response_format}"
            }
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="VibeVoice OpenAI-Compatible TTS Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8880, help="Port to bind")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH, help="Model path")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "mps"], help="Device")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    # Set environment variables for startup
    os.environ["VIBEVOICE_MODEL_PATH"] = args.model_path
    os.environ["VIBEVOICE_DEVICE"] = args.device

    print(f"Starting VibeVoice TTS Server on http://{args.host}:{args.port}")
    print(f"Browser UI: http://{args.host}:{args.port}/")
    print(f"OpenAI TTS endpoint: http://{args.host}:{args.port}/v1/audio/speech")

    uvicorn.run(
        "vibevoice_realtime_openai_api:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    # To suppress warnings, run with: python -W ignore vibevoice_realtime_openai_api.py
    main()
