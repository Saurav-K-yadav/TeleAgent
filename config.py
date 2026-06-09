"""
Central configuration for the telecalling agent.
Edit the values in this file to match your local setup.
"""

import os
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent
DATA_DIR   = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ── Hugging Face ──────────────────────────────────────────────────────────────
# Required: Cohere Transcribe is gated — you must accept the license at
# https://huggingface.co/CohereLabs/cohere-transcribe-03-2026
# then set your HF token here or in the environment variable HF_TOKEN.
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# ── ASR: Cohere Transcribe ────────────────────────────────────────────────────
TRANSCRIBE_MODEL_ID = "CohereLabs/cohere-transcribe-03-2026"
TRANSCRIBE_LANGUAGE = "en"          # ISO 639-1 — English only for now
TRANSCRIBE_DEVICE   = "cuda:0"      # falls back to "cpu" automatically in code

# ── Intent Parser: Qwen2.5-7B-Instruct (GGUF via llama-cpp-python) ────────────
# Download command (run once):
#   huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF \
#       --include "qwen2.5-7b-instruct-q4_k_m.gguf" \
#       --local-dir ./models
QWEN_GGUF_PATH    = MODELS_DIR / "qwen2.5-7b-instruct-q4_k_m.gguf"
QWEN_N_GPU_LAYERS = 20     # offload 20 transformer layers to GPU (~0.8 GB VRAM)
                           # remaining ~15 layers run on CPU RAM
QWEN_N_CTX        = 4096   # context window — sufficient for a call transcript
QWEN_MAX_TOKENS   = 512    # max tokens for the structured JSON response
QWEN_TEMPERATURE  = 0.1    # near-deterministic for structured output

# ── Evaluator: MiniCPM3-4B (CPU, bitsandbytes 4-bit) ─────────────────────────
MINICPM_MODEL_ID   = "openbmb/MiniCPM3-4B"
MINICPM_DEVICE     = "cpu"   # runs after Qwen is done — no VRAM conflict
MINICPM_MAX_TOKENS = 256

# ── VAD: Silero VAD (ONNX, CPU) ───────────────────────────────────────────────
VAD_SAMPLE_RATE        = 16000   # Hz — Silero and Cohere both want 16kHz
VAD_CHUNK_MS           = 250     # ms per audio chunk fed to VAD
VAD_CHUNK_SAMPLES      = int(VAD_SAMPLE_RATE * VAD_CHUNK_MS / 1000)  # 4000
VAD_SILENCE_THRESHOLD  = 0.5     # Silero confidence below this → silence
VAD_SILENCE_DURATION_S = 0.8     # seconds of silence before utterance is flushed
VAD_MIN_SPEECH_S       = 0.5     # ignore utterances shorter than this (noise)

# ── SQLite database ───────────────────────────────────────────────────────────
DB_PATH = DATA_DIR / "calls.db"

# ── Scheduling rules (evaluated by MiniCPM) ──────────────────────────────────
# These are injected into MiniCPM's system prompt so it can reason about them.
SCHEDULING_RULES = """
1. Meetings can only be booked Monday–Friday, 09:00–18:00.
2. Minimum meeting duration is 15 minutes; maximum is 120 minutes.
3. Back-to-back meetings are not allowed — require a 15-minute gap between slots.
4. If the caller does not provide a date or time, ask for one before confirming.
5. If the requested slot is already booked, suggest the next available slot.
6. Always confirm the caller's name before booking.
"""

# ── Gradio UI ─────────────────────────────────────────────────────────────────
APP_TITLE       = "📞 AI Telecalling Agent"
APP_DESCRIPTION = "Speak naturally — the agent will schedule your meeting automatically."
SERVER_PORT     = 7860
SERVER_NAME     = "0.0.0.0"   # bind to all interfaces for HF Spaces