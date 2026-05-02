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
import struct
import subprocess
import threading
import time
import traceback
import unicodedata
import urllib.request
from pathlib import Path
from typing import Dict, List, Literal, Optional, Any, Iterator
from contextlib import asynccontextmanager, contextmanager

# Set HuggingFace cache BEFORE importing any HF libraries
# Only use HF_HOME (TRANSFORMERS_CACHE is deprecated in v5)
# MODELS_DIR can be overridden via env var for Docker volume mounts
MODELS_DIR = Path(os.environ.get("MODELS_DIR", Path(__file__).parent / "models"))
os.environ["HF_HOME"] = str(MODELS_DIR / "huggingface")

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator
import uvicorn
import scipy.io.wavfile as wavfile
from scipy import signal as scipy_signal

# VibeVoice imports (after setting HF_HOME)
from vibevoice.modular.modeling_vibevoice_streaming_inference import (
    VibeVoiceStreamingForConditionalGenerationInference,
)
from vibevoice.modular.streamer import AudioStreamer
from vibevoice.processor.vibevoice_streaming_processor import (
    VibeVoiceStreamingProcessor,
)

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

SAMPLE_RATE = 24000
DEFAULT_MODEL_PATH = "microsoft/VibeVoice-Realtime-0.5B"

# --- TTS audio enhancement (env-overridable, numpy/scipy only) ---
def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    if v in ("1", "true", "yes", "on"):
        return True
    return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


_CFG_PERSIST_LOCK = threading.Lock()
_DDPM_OVERRIDE_LOCK = threading.Lock()

STATIC_DIR = Path(__file__).resolve().parent / "static"
UI_SETTINGS_FILENAME = "ui_settings.json"


def _ui_settings_path() -> Path:
    return MODELS_DIR / UI_SETTINGS_FILENAME


class UISettings(BaseModel):
    """Persisted UI + server defaults (``models/ui_settings.json``). Env vars seed defaults on first load."""

    model_config = ConfigDict(extra="ignore")

    cfg_scale: float = Field(ge=0.0, le=3.0, description="CFG guidance when API omits cfg_scale")
    ddpm_steps: int = Field(ge=1, le=50, description="Diffusion denoising steps (quality vs speed)")

    tts_enhance: bool = Field(description="Master switch for post-processing (non-streaming + streaming band-limit)")
    tts_highpass_hz: float = Field(ge=20.0, le=500.0, description="High-pass cutoff (Hz)")
    tts_lowpass_hz: float = Field(
        ge=0.0,
        le=12000.0,
        description="Low-pass cutoff (Hz); 0 disables",
    )
    tts_light_dns: bool = Field(description="Light STFT noise suppression (offline path only)")
    tts_fade_ms: float = Field(ge=0.0, le=50.0, description="Fade in/out (ms)")
    tts_target_peak: float = Field(ge=0.5, le=1.0, description="Peak ceiling when samples exceed full scale")

    ui_playback_stream_pcm: bool = Field(
        default=False,
        description="If true, web UI calls /v1/audio/speech with stream=true and pcm",
    )


def default_ui_settings() -> UISettings:
    """Baseline from environment (used when no JSON or missing keys)."""
    return UISettings(
        cfg_scale=max(0.0, min(3.0, float(os.environ.get("CFG_SCALE", "1.25")))),
        ddpm_steps=max(1, min(50, _env_int("VIBEVOICE_DDPM_STEPS", 5))),
        tts_enhance=_env_bool("TTS_ENHANCE", True),
        tts_highpass_hz=max(20.0, min(500.0, _env_float("TTS_HIGHPASS_HZ", 60.0))),
        tts_lowpass_hz=max(0.0, min(12000.0, _env_float("TTS_LOWPASS_HZ", 0.0))),
        tts_light_dns=_env_bool("TTS_LIGHT_DNS", False),
        tts_fade_ms=max(0.0, min(50.0, _env_float("TTS_FADE_MS", 3.5))),
        tts_target_peak=max(0.5, min(1.0, _env_float("TTS_TARGET_PEAK", 0.98))),
        ui_playback_stream_pcm=False,
    )


_runtime_ui: UISettings = default_ui_settings()


def load_persisted_ui_settings() -> None:
    """Merge ``models/ui_settings.json`` over env defaults (per-field)."""
    global _runtime_ui
    base = default_ui_settings().model_dump()
    path = _ui_settings_path()
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for k in UISettings.model_fields:
                if k in data:
                    base[k] = data[k]
            _runtime_ui = UISettings.model_validate(base)
            return
        except (ValueError, TypeError, json.JSONDecodeError, OSError) as e:
            print(f"[warning] Could not load {path}: {e}")
    _runtime_ui = UISettings.model_validate(base)


def get_runtime_ui_settings() -> UISettings:
    with _CFG_PERSIST_LOCK:
        return _runtime_ui.model_copy()


def get_runtime_cfg_scale() -> float:
    with _CFG_PERSIST_LOCK:
        return _runtime_ui.cfg_scale


def save_ui_settings(settings: UISettings) -> UISettings:
    """Validate, persist, update runtime, and apply DDPM steps if model is loaded."""
    global _runtime_ui

    validated = UISettings.model_validate(settings.model_dump())
    path = _ui_settings_path()
    blob = json.dumps(validated.model_dump(), indent=2) + "\n"
    with _CFG_PERSIST_LOCK:
        _runtime_ui = validated
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(blob, encoding="utf-8")

    g = globals()
    svc = g.get("tts_service")
    if svc is not None and getattr(svc, "model", None) is not None:
        try:
            svc.model.set_ddpm_inference_steps(num_steps=validated.ddpm_steps)
            print(f"[ui] DDPM inference steps set to {validated.ddpm_steps}")
        except Exception as e:
            print(f"[warning] Could not apply ddpm_steps: {e}")

    return get_runtime_ui_settings()


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


def enhance_tts_audio(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    *,
    settings: Optional[UISettings] = None,
) -> np.ndarray:
    """Lightweight cleanup for synthesized speech: DC removal, band-pass, optional STFT gate, peak, fades.

    Designed to stay cheap (scipy.signal only). Streaming paths use a simpler causal variant separately.
    """
    s = settings if settings is not None else get_runtime_ui_settings()
    if not s.tts_enhance or audio.size == 0:
        return audio.astype(np.float32, copy=False)

    x = np.asarray(audio, dtype=np.float64)
    x -= np.mean(x)

    nyq = sample_rate / 2.0
    hp = min(s.tts_highpass_hz, nyq * 0.05)
    if hp > 15.0:
        sos_hp = scipy_signal.butter(2, hp, btype="high", fs=sample_rate, output="sos")
        try:
            x = scipy_signal.sosfiltfilt(sos_hp, x)
        except ValueError:
            x = scipy_signal.sosfilt(sos_hp, x)

    if s.tts_lowpass_hz and s.tts_lowpass_hz < nyq * 0.95:
        lp = min(s.tts_lowpass_hz, nyq * 0.99)
        sos_lp = scipy_signal.butter(2, lp, btype="low", fs=sample_rate, output="sos")
        try:
            x = scipy_signal.sosfiltfilt(sos_lp, x)
        except ValueError:
            x = scipy_signal.sosfilt(sos_lp, x)

    if s.tts_light_dns and x.size >= 1024:
        x = _light_spectral_gate(x.astype(np.float64), sample_rate)

    x = x.astype(np.float32)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 1.0:
        x = (x / peak * s.tts_target_peak).astype(np.float32)

    if s.tts_fade_ms > 0 and x.size > 2:
        n = int(sample_rate * s.tts_fade_ms * 1e-3)
        n = min(n, x.size // 2)
        if n > 0:
            ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
            x[:n] *= ramp
            x[-n:] *= ramp[::-1]

    return x.astype(np.float32)


@contextmanager
def ddpm_steps_override(
    service: Optional["VibeVoiceTTSService"],
    steps: Optional[int],
):
    """Temporarily set DDPM inference steps on the shared model (serialized with other overrides)."""
    if steps is None or service is None or service.model is None:
        yield
        return
    with _DDPM_OVERRIDE_LOCK:
        prev = service.model.ddpm_inference_steps
        try:
            service.model.set_ddpm_inference_steps(num_steps=steps)
            yield
        finally:
            service.model.set_ddpm_inference_steps(num_steps=prev)


def iter_pcm_s16_chunks_full_enhance(
    pcm_chunks: Iterator[bytes],
    chunk_bytes: int = 8192,
) -> Iterator[bytes]:
    """Concatenate streamed int16 PCM, run ``enhance_tts_audio``, re-chunk for the HTTP body."""
    buf = bytearray()
    for chunk in pcm_chunks:
        buf.extend(chunk)
    if not buf:
        return
    x = np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32768.0
    x = enhance_tts_audio(x, SAMPLE_RATE)
    pcm = (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
    for i in range(0, len(pcm), chunk_bytes):
        yield pcm[i : i + chunk_bytes]


def _light_spectral_gate(x: np.ndarray, sr: int) -> np.ndarray:
    """Conservative magnitude flooring using a per-bin noise floor (median over time)."""
    n_fft = 512
    hop = 128
    if x.size < n_fft:
        return x.astype(np.float64, copy=False)
    nover = n_fft - hop
    f, t, Z = scipy_signal.stft(x, sr, nperseg=n_fft, noverlap=nover, boundary="zeros")
    mag = np.abs(Z)
    noise = np.median(mag, axis=1, keepdims=True)
    mask = np.maximum(1.0 - 1.15 * noise / (mag + 1e-7), 0.2)
    Zr = Z * mask
    _, xr = scipy_signal.istft(Zr, sr, nperseg=n_fft, noverlap=nover)
    if xr.size >= x.size:
        xr = xr[: x.size]
    else:
        xr = np.pad(xr, (0, x.size - xr.size))
    return xr.astype(np.float64)


class _StreamingBandlimit:
    """Minimal per-chunk filtering for streamed PCM (causal SOS, keeps latency low)."""

    def __init__(self, sample_rate: int, settings: Optional[UISettings] = None):
        s = settings if settings is not None else get_runtime_ui_settings()
        self.sample_rate = sample_rate
        self._enhance = s.tts_enhance
        hp = min(s.tts_highpass_hz, sample_rate * 0.05)
        self.sos_hp = scipy_signal.butter(2, hp, btype="high", fs=sample_rate, output="sos")
        self.zi_hp = scipy_signal.sosfilt_zi(self.sos_hp)
        self.sos_lp = None
        self.zi_lp = None
        nyq = sample_rate / 2.0
        if s.tts_lowpass_hz and s.tts_lowpass_hz < nyq * 0.95:
            lp = min(s.tts_lowpass_hz, nyq * 0.99)
            self.sos_lp = scipy_signal.butter(2, lp, btype="low", fs=sample_rate, output="sos")
            self.zi_lp = scipy_signal.sosfilt_zi(self.sos_lp)

    def process(self, chunk: np.ndarray) -> np.ndarray:
        if not self._enhance or chunk.size == 0:
            return chunk.astype(np.float32, copy=False)
        x = chunk.astype(np.float64)
        x -= np.mean(x)
        x, self.zi_hp = scipy_signal.sosfilt(self.sos_hp, x, zi=self.zi_hp)
        if self.sos_lp is not None and self.zi_lp is not None:
            x, self.zi_lp = scipy_signal.sosfilt(self.sos_lp, x, zi=self.zi_lp)
        return np.clip(x, -1.0, 1.0).astype(np.float32)


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
    """OpenAI-compatible TTS request (POST /v1/audio/speech)."""

    model_config = ConfigDict(extra="ignore")

    input: str = Field(..., description="Text to synthesize", max_length=4096)
    voice: str = Field(
        default=DEFAULT_VOICE_ALIAS,
        description='Voice: canonical stem, alias, or OpenAI-style {"id": "..."}',
    )
    model: str = Field(default="tts-1", description="Model ID (ignored, for compatibility)")
    instructions: Optional[str] = Field(
        default=None,
        description="OpenAPI field for gpt-4o-mini-tts style prompts (ignored here)",
    )
    response_format: str = Field(default="mp3", description="Audio format")
    speed: float = Field(default=1.0, description="Speed (not yet supported)")
    stream: bool = Field(
        default=False,
        description="Chunked audio body (legacy; prefer stream_format=audio)",
    )
    stream_format: Optional[Literal["sse", "audio"]] = Field(
        default=None,
        description='OpenAI: "audio" for raw/chunked audio, "sse" for Server-Sent Events',
    )
    cfg_scale: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=3.0,
        description="CFG guidance for this request only; omit to use persisted server default",
    )
    ddpm_steps: Optional[int] = Field(
        default=None,
        ge=1,
        le=50,
        description="Override DDPM denoising steps for this request (serialized server-wide; restores after)",
    )
    vibevoice_post_enhance_stream: bool = Field(
        default=False,
        description="When streaming: buffer the utterance, apply full offline enhance, then send (high latency)",
    )

    @field_validator("voice", mode="before")
    @classmethod
    def coerce_voice(cls, v: Any) -> str:
        if isinstance(v, str):
            return v
        if isinstance(v, dict):
            vid = v.get("id")
            if isinstance(vid, str) and vid.strip():
                return vid.strip()
        raise ValueError('voice must be a string or an object with id, e.g. {"id": "voice_123"}')


def _speech_streaming_mode(req: TTSRequest) -> Optional[Literal["sse", "audio"]]:
    """How the client requested streaming (OpenAI: stream_format; we also honor stream=true)."""
    if req.stream_format == "sse":
        return "sse"
    if req.stream_format == "audio":
        return "audio"
    if req.stream:
        return "audio"
    return None


def _streaming_wav_header_pcm_s16le(sample_rate: int = SAMPLE_RATE, channels: int = 1) -> bytes:
    """Minimal WAV header for unknown-length PCM streams (chunk sizes 0xFFFFFFFF)."""
    bits_per_sample = 16
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        0xFFFFFFFF,
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        0xFFFFFFFF,
    )


def iter_pcm_chunks_through_ffmpeg(
    pcm_chunks: Iterator[bytes], fmt: str
) -> Iterator[bytes]:
    """Stream s16le mono PCM chunks through ffmpeg and yield encoded output chunks."""
    format_args = {
        "mp3": ["-f", "mp3", "-codec:a", "libmp3lame", "-q:a", "2"],
        "opus": ["-f", "opus", "-codec:a", "libopus"],
        "flac": ["-f", "flac", "-codec:a", "flac"],
        "aac": ["-f", "adts", "-codec:a", "aac"],
    }
    fmt = fmt.lower()
    if fmt not in format_args:
        raise ValueError(f"No ffmpeg streaming pipeline for format: {fmt}")

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "s16le",
        "-ar",
        str(SAMPLE_RATE),
        "-ac",
        "1",
        "-i",
        "pipe:0",
        *format_args[fmt],
        "pipe:1",
    ]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=0,
    )
    errors: List[BaseException] = []

    def writer() -> None:
        try:
            if proc.stdin is None:
                return
            for chunk in pcm_chunks:
                proc.stdin.write(chunk)
            proc.stdin.close()
        except BaseException as e:
            errors.append(e)
            try:
                if proc.stdin:
                    proc.stdin.close()
            except BrokenPipeError:
                pass
            try:
                proc.terminate()
            except ProcessLookupError:
                pass

    thread = threading.Thread(target=writer, daemon=True)
    thread.start()
    try:
        if proc.stdout is None:
            raise RuntimeError("ffmpeg stdout not available")
        while True:
            data = proc.stdout.read(8192)
            if not data:
                break
            yield data
    finally:
        thread.join(timeout=7200.0)
        proc.wait()
        if errors:
            raise RuntimeError("Streaming encode failed") from errors[0]
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg streaming encode failed (exit {proc.returncode})")


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
        cuda_idx = _parse_cuda_device_index(self.device)
        if cuda_idx is not None:
            torch.cuda.set_device(cuda_idx)
            _apply_cuda_sdp_backends_for_capability(cuda_idx)
            load_dtype = _cuda_model_load_dtype(cuda_idx)
            device_map = "cuda"
            attn_impl = preferred_cuda_attn_implementation(self.device)
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
        ddpm = get_runtime_ui_settings().ddpm_steps
        self.model.set_ddpm_inference_steps(num_steps=ddpm)
        print(f"[startup] DDPM inference steps = {ddpm}")

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

    def _tts_inputs_for_text(self, text: str, voice: str) -> tuple[str, str, Any, Dict[str, Any]]:
        """Resolve voice, sanitize text, build model inputs on device."""
        voice_stem = self._resolve_voice(voice)
        prefilled_outputs = self._get_voice_prompt(voice_stem)
        clean = sanitize_tts_input(text).replace("'", "'")
        inputs = self.processor.process_input_with_cached_prompt(
            text=clean,
            cached_prompt=prefilled_outputs,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        inputs_dict = dict(inputs)
        for k, v in inputs_dict.items():
            if torch.is_tensor(v):
                inputs_dict[k] = v.to(self._torch_device)
        return clean, voice_stem, prefilled_outputs, inputs_dict

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

        clean, voice_stem, prefilled_outputs, inputs = self._tts_inputs_for_text(text, voice)

        print(f"[tts] Generating speech for {len(clean)} chars with voice '{voice_stem}'")
        start_time = time.time()

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

        if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
            audio = outputs.speech_outputs[0]
            if torch.is_tensor(audio):
                audio = audio.detach().cpu().to(torch.float32).numpy()
            else:
                audio = np.asarray(audio, dtype=np.float32)

            if audio.ndim > 1:
                audio = audio.reshape(-1)

            audio = enhance_tts_audio(audio, SAMPLE_RATE)

            duration = len(audio) / SAMPLE_RATE
            rtf = elapsed / duration if duration > 0 else float("inf")
            print(f"[tts] Generated {duration:.2f}s audio in {elapsed:.2f}s (RTF: {rtf:.2f}x)")

            return audio
        else:
            raise RuntimeError("No audio output generated")

    def iter_speech_pcm_chunks(
        self, text: str, voice: str, cfg_scale: float = 1.5
    ) -> Iterator[bytes]:
        """Stream int16 little-endian PCM (mono, 24 kHz) using the model's AudioStreamer.

        Chunks are emitted as they are decoded; light causal band-limiting follows persisted ``tts_enhance`` / filter settings.
        """
        if not self.model or not self.processor:
            raise RuntimeError("Model not loaded")

        clean, voice_stem, prefilled_outputs, inputs = self._tts_inputs_for_text(text, voice)
        print(f"[tts] Streaming speech for {len(clean)} chars with voice '{voice_stem}'")

        streamer = AudioStreamer(batch_size=1, stop_signal=None)
        errors: List[BaseException] = []

        def run_generate() -> None:
            try:
                self.model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=cfg_scale,
                    tokenizer=self.processor.tokenizer,
                    generation_config={"do_sample": False},
                    verbose=False,
                    all_prefilled_outputs=copy.deepcopy(prefilled_outputs),
                    audio_streamer=streamer,
                )
            except BaseException as e:
                errors.append(e)
                try:
                    streamer.end()
                except Exception:
                    pass

        worker = threading.Thread(target=run_generate, daemon=True)
        worker.start()

        causal = _StreamingBandlimit(SAMPLE_RATE, get_runtime_ui_settings())
        try:
            sample_iter = streamer.get_stream(0)
            for tensor_chunk in sample_iter:
                arr = tensor_chunk
                if torch.is_tensor(arr):
                    arr = arr.detach().cpu().to(torch.float32).numpy()
                arr = np.asarray(arr, dtype=np.float32).reshape(-1)
                if arr.size == 0:
                    continue
                arr = causal.process(arr)
                pcm = (np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
                yield pcm
        finally:
            worker.join(timeout=7200.0)

        if errors:
            raise RuntimeError("Streaming TTS failed") from errors[0]


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


def _parse_cuda_device_index(device: str) -> Optional[int]:
    """Return integer CUDA ordinal for ``cuda`` / ``cuda:N``, else ``None``."""
    d = device.strip().lower()
    if d == "cuda":
        return 0
    if d.startswith("cuda:"):
        tail = d[5:].strip()
        if tail.isdigit():
            return int(tail)
    return None


def _ampere_or_newer_cuda_capability(cuda_idx: int) -> bool:
    """True if compute capability >= 8.0 (Ampere and newer)."""
    major, _ = torch.cuda.get_device_capability(cuda_idx)
    return major >= 8


def _apply_cuda_sdp_backends_for_capability(cuda_idx: int) -> None:
    """Pre-Ampere GPUs often hit ``cutlassF: no kernel found`` with bf16 + mem-efficient SDPA; use math kernel."""
    if _ampere_or_newer_cuda_capability(cuda_idx):
        return
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    print(
        "[startup] Pre-Ampere GPU: disabled flash/mem-efficient SDPA (math kernel only) to avoid CUTLASS dispatch issues"
    )


def _cuda_model_load_dtype(cuda_idx: int) -> torch.dtype:
    """bf16 is a poor default on many pre-Ampere chips with SDPA; float32 is slower but stable."""
    if _ampere_or_newer_cuda_capability(cuda_idx):
        return torch.bfloat16
    print("[startup] Pre-Ampere GPU: using float32 weights (not bfloat16) for CUDA stability")
    return torch.float32


def assert_pytorch_supports_visible_gpu(device: str) -> None:
    """Fail fast if this PyTorch build omits the selected GPU (some newer CUDA wheel lines drop older architectures)."""
    if not torch.cuda.is_available():
        return
    idx = _parse_cuda_device_index(device)
    if idx is None:
        return
    n = torch.cuda.device_count()
    if idx >= n:
        raise RuntimeError(
            f"CUDA device cuda:{idx} not available (only {n} device(s) visible)."
        )
    cap = torch.cuda.get_device_capability(idx)
    arch = f"sm_{cap[0]}{cap[1]}"
    getter = getattr(torch.cuda, "get_arch_list", None)
    if not callable(getter):
        return
    arch_list = getter()
    if not arch_list or arch in arch_list:
        return
    raise RuntimeError(
        f"This PyTorch wheel has no CUDA kernels for {arch} (device capability {cap}). "
        f"Built arches: {arch_list}. "
        f"For broad NVIDIA GPU coverage (including older architectures), install PyTorch from the "
        f"cu126 wheel index instead of cu128: UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu126 "
        f"(the default Docker image uses cu126)."
    )


def preferred_cuda_attn_implementation(device: str) -> str:
    """Prefer SDPA on older NVIDIA GPUs where Flash Attention 2 is a weak default; allow override via env."""
    if os.environ.get("VIBEVOICE_USE_FLASH_ATTN", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return "flash_attention_2"
    if not torch.cuda.is_available():
        return "sdpa"
    idx = _parse_cuda_device_index(device)
    if idx is None:
        return "sdpa"
    if torch.cuda.get_device_capability(idx) == (7, 0):
        return "sdpa"
    return "flash_attention_2"


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

    load_persisted_ui_settings()

    if _parse_cuda_device_index(device) is not None and torch.cuda.is_available():
        assert_pytorch_supports_visible_gpu(device)

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


@app.get("/api/ui/settings", response_model=UISettings)
async def get_ui_settings():
    """Full persisted UI + server defaults (``models/ui_settings.json``)."""
    return get_runtime_ui_settings()


@app.put("/api/ui/settings", response_model=UISettings)
async def put_ui_settings(settings: UISettings):
    """Persist all UI settings and apply ``ddpm_steps`` to the loaded model immediately."""
    return save_ui_settings(settings)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    ru = get_runtime_ui_settings()
    return HealthResponse(
        status="ok",
        service="vibevoice-realtime-openai-api",
        model_loaded=tts_service is not None and tts_service.model is not None,
        device=tts_service.device if tts_service else "unknown",
        features={
            "streaming": True,
            "stream_format_sse": False,
            "stream_format_audio": True,
            "streaming_response_formats": SUPPORTED_FORMATS,
            "formats": SUPPORTED_FORMATS,
            "sample_rate": SAMPLE_RATE,
            "ddpm_per_request": True,
            "vibevoice_post_enhance_stream": True,
            "ui_settings": ru.model_dump(),
        },
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

    stream_mode = _speech_streaming_mode(request)
    if stream_mode == "sse":
        raise HTTPException(
            status_code=501,
            detail="stream_format=sse is not implemented. Use stream_format=audio (or stream=true) for chunked audio.",
        )

    rf = request.response_format.lower()

    try:
        effective_cfg = (
            request.cfg_scale if request.cfg_scale is not None else get_runtime_cfg_scale()
        )

        if stream_mode == "audio":

            def streaming_pcm_body() -> Iterator[bytes]:
                with ddpm_steps_override(tts_service, request.ddpm_steps):
                    inner = tts_service.iter_speech_pcm_chunks(
                        text=text_in,
                        voice=request.voice,
                        cfg_scale=effective_cfg,
                    )
                    if request.vibevoice_post_enhance_stream:
                        yield from iter_pcm_s16_chunks_full_enhance(inner)
                    else:
                        yield from inner

            if rf == "pcm":
                return StreamingResponse(
                    streaming_pcm_body(),
                    media_type="audio/pcm",
                    headers={
                        "Content-Disposition": 'attachment; filename="speech.pcm"',
                        "X-Sample-Rate": str(SAMPLE_RATE),
                        "X-Channels": "1",
                        "X-Bits-Per-Sample": "16",
                    },
                )

            if rf == "wav":

                def wav_stream() -> Iterator[bytes]:
                    yield _streaming_wav_header_pcm_s16le()
                    yield from streaming_pcm_body()

                return StreamingResponse(
                    wav_stream(),
                    media_type="audio/wav",
                    headers={
                        "Content-Disposition": 'attachment; filename="speech.wav"',
                        "X-Sample-Rate": str(SAMPLE_RATE),
                        "X-Channels": "1",
                        "X-Bits-Per-Sample": "16",
                    },
                )

            def encoded_stream() -> Iterator[bytes]:
                yield from iter_pcm_chunks_through_ffmpeg(streaming_pcm_body(), rf)

            return StreamingResponse(
                encoded_stream(),
                media_type=get_content_type(rf),
                headers={"Content-Disposition": f'attachment; filename="speech.{rf}"'},
            )

        with ddpm_steps_override(tts_service, request.ddpm_steps):
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

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def _parse_cli_device(s: str) -> str:
    """``cpu``, ``mps``, ``cuda`` (device 0), or ``cuda:N`` for a specific NVIDIA GPU index."""
    raw = s.strip()
    sl = raw.lower()
    if sl in ("cpu", "mps", "cuda"):
        return sl
    if sl.startswith("cuda:"):
        tail = sl[5:].strip()
        if tail.isdigit() and int(tail) >= 0:
            return f"cuda:{int(tail)}"
    raise argparse.ArgumentTypeError(
        "device must be cpu, mps, cuda, or cuda:N (e.g. cuda:2 for the third GPU)"
    )


def main():
    parser = argparse.ArgumentParser(description="VibeVoice OpenAI-Compatible TTS Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8880, help="Port to bind")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH, help="Model path")
    parser.add_argument(
        "--device",
        type=_parse_cli_device,
        default=None,
        help="torch device: cpu | mps | cuda | cuda:N — overrides VIBEVOICE_DEVICE when set",
    )
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    # Set environment variables for startup
    os.environ["VIBEVOICE_MODEL_PATH"] = args.model_path
    if args.device is not None:
        os.environ["VIBEVOICE_DEVICE"] = args.device
    elif "VIBEVOICE_DEVICE" not in os.environ:
        os.environ["VIBEVOICE_DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"

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
