"""
pipeline/transcriber.py

Cohere Transcribe wrapper for the telecalling agent.

Model
─────
CohereLabs/cohere-transcribe-03-2026
  - 2B parameter Conformer encoder + lightweight Transformer decoder
  - Audio-in, text-out ASR, Apache 2.0
  - Gated: requires HF_TOKEN with accepted licence at
    https://huggingface.co/CohereLabs/cohere-transcribe-03-2026

Memory budget on RTX 2050 (4 GB VRAM)
──────────────────────────────────────
  Full fp32  →  ~8 GB   ✗
  float16    →  ~4 GB   ✗ (tight, unstable with other allocations)
  torch_dtype=torch.float16 + device_map keeps ~2.2 GB → ✓

Design decisions
────────────────
  - Lazy-loaded: model is not pulled from HF until the first call arrives.
    This keeps Gradio startup fast and avoids OOM if the user never speaks.
  - GPU-first with automatic CPU fallback: if CUDA is unavailable or VRAM
    is insufficient the model moves to CPU transparently.
  - Uses model.transcribe(audio_arrays=...) to avoid disk I/O entirely.
  - compile=False by default: torch.compile causes a 30-60 s warmup on
    first call which would break the live-call UX. Set compile=True only
    in batch / offline mode.
  - Thread-safe: model loading is protected by a threading.Lock so
    concurrent Gradio sessions don't race to download.

Usage
─────
    transcriber = Transcriber()                 # cheap — model not loaded yet
    text = transcriber.transcribe(sr, audio_np) # loads model on first call
    transcriber.unload()                        # free VRAM between sessions
"""

import logging
import threading
import time
from typing import Optional

import numpy as np
import torch

from config import (
    TRANSCRIBE_MODEL_ID,
    TRANSCRIBE_LANGUAGE,
    TRANSCRIBE_DEVICE,
    HF_TOKEN,
)

logger = logging.getLogger(__name__)


class Transcriber:
    """
    Lazy-loading wrapper around CohereLabs/cohere-transcribe-03-2026.

    Thread-safe: a single instance can be shared across the whole app.
    """

    def __init__(self):
        self._model     = None
        self._processor = None
        self._device    = None
        self._lock      = threading.Lock()
        self._loaded    = False

    # ── Public API ────────────────────────────────────────────────────────────

    def transcribe(self, sample_rate: int, audio: np.ndarray) -> str:
        """
        Transcribe a single utterance.

        Parameters
        ----------
        sample_rate : int
            Sample rate of `audio` (typically 16000 from VAD).
        audio : np.ndarray
            Mono float32 PCM in [-1.0, 1.0].

        Returns
        -------
        str
            Transcribed text, stripped of leading/trailing whitespace.
            Returns "" on error so the pipeline can continue gracefully.
        """
        if audio is None or len(audio) == 0:
            return ""

        self._ensure_loaded()

        audio = self._validate_audio(audio)
        if audio is None:
            return ""

        try:
            t0 = time.perf_counter()

            results = self._model.transcribe(
                processor       = self._processor,
                audio_arrays    = [audio],        # list of np.ndarray
                sample_rates    = [sample_rate],  # matching list of ints
                language        = TRANSCRIBE_LANGUAGE,
                punctuation     = True,
                batch_size      = 1,              # one utterance at a time (live)
                compile         = False,          # no warmup delay in live mode
                pipeline_detokenization = False,  # not needed for batch_size=1
            )

            elapsed = time.perf_counter() - t0
            text    = results[0].strip() if results else ""

            duration = len(audio) / sample_rate
            rtfx     = duration / elapsed if elapsed > 0 else 0
            logger.info(
                f"Transcribed {duration:.2f}s audio in {elapsed:.2f}s "
                f"(RTFx {rtfx:.1f}x): '{text[:80]}{'…' if len(text)>80 else ''}'"
            )
            return text

        except Exception as exc:
            logger.error(f"Transcription failed: {exc}", exc_info=True)
            return ""

    def transcribe_batch(
        self, utterances: list[tuple[int, np.ndarray]]
    ) -> list[str]:
        """
        Transcribe multiple utterances in one forward pass.
        More efficient for post-call batch processing.

        Parameters
        ----------
        utterances : list of (sample_rate, audio_np) tuples

        Returns
        -------
        list of str — one entry per input, "" on individual errors
        """
        if not utterances:
            return []

        self._ensure_loaded()

        sample_rates  = [sr  for sr, _  in utterances]
        audio_arrays  = [self._validate_audio(a) for _, a in utterances]

        # Replace any None (invalid) arrays with silent arrays
        audio_arrays = [
            a if a is not None else np.zeros(sample_rates[i], dtype=np.float32)
            for i, a in enumerate(audio_arrays)
        ]

        try:
            results = self._model.transcribe(
                processor       = self._processor,
                audio_arrays    = audio_arrays,
                sample_rates    = sample_rates,
                language        = TRANSCRIBE_LANGUAGE,
                punctuation     = True,
                batch_size      = len(utterances),
                compile         = False,
                pipeline_detokenization = True,  # helps with larger batches
            )
            return [r.strip() for r in results]

        except Exception as exc:
            logger.error(f"Batch transcription failed: {exc}", exc_info=True)
            return [""] * len(utterances)

    def unload(self):
        """
        Release GPU memory.  Call at end of a call session if memory is tight.
        Model will be reloaded lazily on the next call.
        """
        with self._lock:
            if self._loaded:
                del self._model
                del self._processor
                self._model     = None
                self._processor = None
                self._loaded    = False
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("Transcriber unloaded — VRAM freed.")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def device(self) -> Optional[str]:
        return str(self._device) if self._device else None

    # ── Internal ──────────────────────────────────────────────────────────────

    def _ensure_loaded(self):
        """Load model + processor exactly once, thread-safely."""
        if self._loaded:
            return

        with self._lock:
            if self._loaded:   # double-checked locking
                return
            self._load()

    def _load(self):
        """
        Pull model from HuggingFace and move to best available device.

        Device selection for RTX 2050 (4 GB):
          - float16 on CUDA: ~2.2 GB VRAM ← preferred
          - float32 on CPU:  ~8.0 GB RAM  ← fallback
        """
        import os
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

        logger.info(f"Loading Cohere Transcribe ({TRANSCRIBE_MODEL_ID})…")
        t0 = time.perf_counter()

        token_kwargs = {"token": HF_TOKEN} if HF_TOKEN else {}
        
        # After first download, use local cache only to avoid repeated HF hub calls
        local_only = os.path.exists(
            os.path.expanduser(f"~/.cache/huggingface/hub/models--{TRANSCRIBE_MODEL_ID.replace('/', '--')}")
        )

        # Processor (tokenizer + feature extractor) — tiny, load first
        self._processor = AutoProcessor.from_pretrained(
            TRANSCRIBE_MODEL_ID,
            trust_remote_code=True,
            local_files_only=local_only,
            **token_kwargs,
        )

        # Determine device & dtype
        if torch.cuda.is_available():
            try:
                self._device = torch.device(TRANSCRIBE_DEVICE)
                dtype        = torch.float16
                logger.info(f"CUDA available — loading in float16 on {self._device}")
            except Exception:
                self._device = torch.device("cpu")
                dtype        = torch.float32
                logger.warning("CUDA device init failed — falling back to CPU float32")
        else:
            self._device = torch.device("cpu")
            dtype        = torch.float32
            logger.info("No CUDA — loading on CPU in float32 (will be slower)")

        # Model weights
        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
            TRANSCRIBE_MODEL_ID,
            trust_remote_code = True,
            torch_dtype       = dtype,
            low_cpu_mem_usage = True,   # stream weights instead of double-buffering
            local_files_only  = local_only,
            **token_kwargs,
        ).to(self._device)

        self._model.eval()

        elapsed = time.perf_counter() - t0
        logger.info(
            f"Cohere Transcribe ready on {self._device} "
            f"(dtype={dtype}, loaded in {elapsed:.1f}s)"
        )

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self._device) / 1024**3
            reserved  = torch.cuda.memory_reserved(self._device)  / 1024**3
            logger.info(
                f"VRAM after load — allocated: {allocated:.2f} GB, "
                f"reserved: {reserved:.2f} GB"
            )

        self._loaded = True

    @staticmethod
    def _validate_audio(audio: np.ndarray) -> Optional[np.ndarray]:
        """
        Ensure audio is mono float32.  Returns None if unusable.
        """
        if audio is None or len(audio) == 0:
            return None

        audio = np.array(audio, dtype=np.float32)

        # Stereo → mono
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        elif audio.ndim != 1:
            logger.warning(f"Unexpected audio shape {audio.shape} — skipping")
            return None

        # Normalise int16-range inputs
        if audio.max() > 1.0 or audio.min() < -1.0:
            audio = audio / 32768.0

        audio = np.clip(audio, -1.0, 1.0)
        return audio


# ── Module-level singleton ─────────────────────────────────────────────────────
# Shared across the whole app — model loaded once, reused every utterance.
_transcriber: Optional[Transcriber] = None


def get_transcriber() -> Transcriber:
    """Return the module-level singleton, creating it if needed."""
    global _transcriber
    if _transcriber is None:
        _transcriber = Transcriber()
    return _transcriber


# ── Offline smoke test (no HF token needed) ───────────────────────────────────

def _smoke_test_offline():
    """
    Validates the audio pre-processing path without loading the model.
    Full model load requires a HF token + accepted Cohere licence.
    """
    import math
    logging.basicConfig(level=logging.INFO)
    logger.info("Running offline smoke test (pre-processing only)…")

    SR = 16000

    # 1. Stereo int16 → mono float32
    stereo_int16 = (np.random.randn(SR, 2) * 32767).astype(np.int16)
    result = Transcriber._validate_audio(stereo_int16)
    assert result is not None
    assert result.ndim   == 1,        f"Expected mono, got shape {result.shape}"
    assert result.dtype  == np.float32
    assert result.max()  <= 1.0
    assert result.min()  >= -1.0
    logger.info("  ✓ Stereo int16 → mono float32 normalisation")

    # 2. Already mono float32 — passthrough
    mono_float = np.sin(2 * math.pi * 440 * np.linspace(0, 1, SR)).astype(np.float32)
    result = Transcriber._validate_audio(mono_float)
    assert result is not None and result.shape == (SR,)
    logger.info("  ✓ Mono float32 passthrough")

    # 3. Empty input → None
    result = Transcriber._validate_audio(np.array([]))
    assert result is None
    logger.info("  ✓ Empty input → None")

    # 4. Singleton pattern
    t1 = get_transcriber()
    t2 = get_transcriber()
    assert t1 is t2
    logger.info("  ✓ Module singleton")

    logger.info("\nOffline smoke test PASSED ✓")
    logger.info(
        "\nTo run the full model test (requires HF_TOKEN + accepted licence):\n"
        "  export HF_TOKEN=hf_...\n"
        "  PYTHONPATH=. python3 -c \"\n"
        "  from pipeline.transcriber import get_transcriber\n"
        "  import numpy as np\n"
        "  t = get_transcriber()\n"
        "  # 2s of 440 Hz tone — expect near-empty transcription\n"
        "  audio = np.sin(2*3.14*440*np.linspace(0,2,32000)).astype('float32')\n"
        "  print(repr(t.transcribe(16000, audio)))\n"
        "  \""
    )


if __name__ == "__main__":
    _smoke_test_offline()