"""
pipeline/transcriber.py

Hugging Face Moonshine ASR wrapper for the telecalling agent.

The public API intentionally matches the previous ASR wrapper:

    transcriber = Transcriber()
    text = transcriber.transcribe(sample_rate, audio_np)

Internally this uses UsefulSensors/moonshine-tiny through Transformers:
AutoFeatureExtractor prepares audio features, AutoTokenizer decodes generated
tokens, and MoonshineForConditionalGeneration generates transcripts in memory.
"""

import io
import logging
import threading
import time
from typing import Optional, Sequence

import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None
    _TORCH_AVAILABLE = False

import soundfile as sf
from huggingface_hub import InferenceApi

from config import (
    TRANSCRIBE_MODEL_ID,
    TRANSCRIBE_DEVICE,
    HF_TOKEN,
)

logger = logging.getLogger(__name__)


class Transcriber:
    """
    Lazy-loading wrapper around UsefulSensors/moonshine-tiny.

    Thread-safe: a single instance can be shared across the whole app.
    """

    # Recommended by the Moonshine model card to reduce hallucination loops.
    _MAX_TOKENS_PER_SECOND = 6.5

    def __init__(self):
        self._model = None
        self._feature_extractor = None
        self._tokenizer = None
        self._device = None
        self._dtype = None
        self._sample_rate = None
        self._lock = threading.Lock()
        self._loaded = False
        self._use_remote = False

    def transcribe(self, sample_rate: int, audio: np.ndarray) -> str:
        """
        Transcribe a single utterance.

        Parameters
        ----------
        sample_rate : int
            Sample rate of `audio`.
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
            text = self._generate_text([audio], [sample_rate])[0]
            elapsed = time.perf_counter() - t0

            duration = len(audio) / sample_rate
            rtfx = duration / elapsed if elapsed > 0 else 0
            logger.info(
                f"Transcribed {duration:.2f}s audio in {elapsed:.2f}s "
                f"(RTFx {rtfx:.1f}x): '{text[:80]}{'...' if len(text) > 80 else ''}'"
            )
            return text

        except Exception as exc:
            logger.error(f"Transcription failed: {exc}", exc_info=True)
            return ""

    def transcribe_batch(
        self, utterances: list[tuple[int, np.ndarray]]
    ) -> list[str]:
        """
        Transcribe multiple utterances in one generation call.

        Parameters
        ----------
        utterances : list of (sample_rate, audio_np) tuples

        Returns
        -------
        list of str, one entry per input, "" on individual errors
        """
        if not utterances:
            return []

        self._ensure_loaded()

        sample_rates = [sr for sr, _ in utterances]
        audio_arrays = [self._validate_audio(a) for _, a in utterances]

        audio_arrays = [
            a if a is not None else np.zeros(sample_rates[i], dtype=np.float32)
            for i, a in enumerate(audio_arrays)
        ]

        try:
            return self._generate_text(audio_arrays, sample_rates)
        except Exception as exc:
            logger.error(f"Batch transcription failed: {exc}", exc_info=True)
            return [""] * len(utterances)

    def unload(self):
        """
        Release GPU memory. Call at end of a call session if memory is tight.
        Model will be reloaded lazily on the next call.
        """
        with self._lock:
            if self._loaded:
                self._model = None
                self._feature_extractor = None
                self._tokenizer = None
                self._device = None
                self._dtype = None
                self._sample_rate = None
                self._loaded = False
                if _TORCH_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("Moonshine transcriber unloaded; VRAM freed.")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def device(self) -> Optional[str]:
        return str(self._device) if self._device else None

    def _ensure_loaded(self):
        """Load model + processor exactly once, thread-safely."""
        if self._loaded:
            return

        with self._lock:
            if self._loaded:
                return
            if not _TORCH_AVAILABLE:
                logger.warning(
                    "PyTorch is unavailable in this environment; "
                    "using remote ASR fallback."
                )
                self._use_remote = True
                self._loaded = True
                return
            self._load()

    def _load(self):
        """Pull Moonshine from Hugging Face and move it to the best device."""
        import os

        from transformers import AutoFeatureExtractor, AutoTokenizer

        self._install_torchvision_import_stub_if_needed()
        from transformers import MoonshineForConditionalGeneration

        logger.info(f"Loading Moonshine ASR ({TRANSCRIBE_MODEL_ID})...")
        t0 = time.perf_counter()

        token_kwargs = {"token": HF_TOKEN} if HF_TOKEN else {}
        local_only = os.path.exists(
            os.path.expanduser(
                f"~/.cache/huggingface/hub/models--{TRANSCRIBE_MODEL_ID.replace('/', '--')}"
            )
        )

        self._feature_extractor = AutoFeatureExtractor.from_pretrained(
            TRANSCRIBE_MODEL_ID,
            local_files_only=local_only,
            **token_kwargs,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            TRANSCRIBE_MODEL_ID,
            local_files_only=local_only,
            **token_kwargs,
        )
        self._sample_rate = int(self._feature_extractor.sampling_rate)

        if torch.cuda.is_available():
            try:
                self._device = torch.device(TRANSCRIBE_DEVICE)
                self._dtype = torch.float16
                logger.info(f"CUDA available; loading Moonshine in float16 on {self._device}")
            except Exception:
                self._device = torch.device("cpu")
                self._dtype = torch.float32
                logger.warning("CUDA device init failed; falling back to CPU float32")
        else:
            self._device = torch.device("cpu")
            self._dtype = torch.float32
            logger.info("No CUDA; loading Moonshine on CPU in float32")

        self._model = MoonshineForConditionalGeneration.from_pretrained(
            TRANSCRIBE_MODEL_ID,
            torch_dtype=self._dtype,
            low_cpu_mem_usage=True,
            local_files_only=local_only,
            **token_kwargs,
        ).to(self._device)
        self._model.eval()

        elapsed = time.perf_counter() - t0
        logger.info(
            f"Moonshine ASR ready on {self._device} "
            f"(dtype={self._dtype}, sample_rate={self._sample_rate}, loaded in {elapsed:.1f}s)"
        )

        if torch.cuda.is_available() and self._device.type == "cuda":
            allocated = torch.cuda.memory_allocated(self._device) / 1024**3
            reserved = torch.cuda.memory_reserved(self._device) / 1024**3
            logger.info(
                f"VRAM after load: allocated={allocated:.2f} GB, reserved={reserved:.2f} GB"
            )

        self._loaded = True

    def _generate_text(
        self,
        audio_arrays: Sequence[np.ndarray],
        sample_rates: Sequence[int],
    ) -> list[str]:
        """Run Moonshine generation and decode transcripts."""
        if self._use_remote:
            return [self._remote_transcribe(audio, sr) for audio, sr in zip(audio_arrays, sample_rates)]

        prepared = [
            self._resample(audio, sr, self._sample_rate)
            for audio, sr in zip(audio_arrays, sample_rates)
        ]

        inputs = self._feature_extractor(
            prepared,
            return_tensors="pt",
            sampling_rate=self._sample_rate,
            padding=True,
        )
        inputs = inputs.to(self._device, self._dtype)

        with torch.inference_mode():
            seq_lens = inputs.attention_mask.sum(dim=-1)
            token_limit_factor = self._MAX_TOKENS_PER_SECOND / self._sample_rate
            max_length = max(1, int((seq_lens * token_limit_factor).max().item()))
            generated_ids = self._model.generate(**inputs, max_length=max_length)

        return [
            self._tokenizer.decode(ids, skip_special_tokens=True).strip()
            for ids in generated_ids
        ]

    def _remote_transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """Use the Hugging Face Inference API to transcribe audio if local Torch is unavailable."""
        if not HF_TOKEN:
            logger.warning("HF_TOKEN is not set; falling back to mock transcription.")
            return ""

        try:
            client = InferenceApi(repo_id="openai/whisper-small", token=HF_TOKEN)
            with io.BytesIO() as buffer:
                sf.write(buffer, audio, samplerate=sample_rate, format="WAV")
                buffer.seek(0)
                response = client(inputs=buffer)
            if isinstance(response, dict) and "text" in response:
                return response["text"].strip()
            return str(response)
        except Exception as exc:
            logger.error(f"Remote transcription failed: {exc}", exc_info=True)
            return ""

    @staticmethod
    def _install_torchvision_import_stub_if_needed() -> None:
        """
        Keep audio-only Moonshine usable when torchvision is installed but broken.

        Transformers 5.x imports generic image/video utilities while importing
        modeling classes. A mismatched torchvision wheel can raise before any ASR
        code runs, even though Moonshine does not use torchvision at inference.
        """
        try:
            import torchvision  # noqa: F401
            return
        except Exception:
            pass

        import importlib.machinery
        import sys
        import types

        for name in (
            "torchvision",
            "torchvision.transforms",
            "torchvision.transforms.v2",
            "torchvision.transforms.v2.functional",
            "torchvision.io",
        ):
            sys.modules.pop(name, None)

        torchvision = types.ModuleType("torchvision")
        torchvision.__spec__ = importlib.machinery.ModuleSpec("torchvision", None)
        transforms = types.ModuleType("torchvision.transforms")
        transforms.__spec__ = importlib.machinery.ModuleSpec("torchvision.transforms", None)
        transforms_v2 = types.ModuleType("torchvision.transforms.v2")
        transforms_v2.__spec__ = importlib.machinery.ModuleSpec(
            "torchvision.transforms.v2", None
        )
        transforms_v2_functional = types.ModuleType("torchvision.transforms.v2.functional")
        transforms_v2_functional.__spec__ = importlib.machinery.ModuleSpec(
            "torchvision.transforms.v2.functional", None
        )
        torchvision_io = types.ModuleType("torchvision.io")
        torchvision_io.__spec__ = importlib.machinery.ModuleSpec("torchvision.io", None)

        class InterpolationMode:
            NEAREST = 0
            NEAREST_EXACT = 0
            BILINEAR = 2
            BICUBIC = 3
            BOX = 4
            HAMMING = 5
            LANCZOS = 1

        transforms.InterpolationMode = InterpolationMode
        transforms.v2 = transforms_v2
        transforms_v2.functional = transforms_v2_functional
        torchvision.transforms = transforms
        torchvision.io = torchvision_io

        sys.modules["torchvision"] = torchvision
        sys.modules["torchvision.transforms"] = transforms
        sys.modules["torchvision.transforms.v2"] = transforms_v2
        sys.modules["torchvision.transforms.v2.functional"] = transforms_v2_functional
        sys.modules["torchvision.io"] = torchvision_io

    @staticmethod
    def _validate_audio(audio: np.ndarray) -> Optional[np.ndarray]:
        """Ensure audio is mono float32 in [-1.0, 1.0]."""
        if audio is None or len(audio) == 0:
            return None

        audio = np.array(audio, dtype=np.float32)

        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        elif audio.ndim != 1:
            logger.warning(f"Unexpected audio shape {audio.shape}; skipping")
            return None

        if audio.max() > 1.0 or audio.min() < -1.0:
            audio = audio / 32768.0

        return np.clip(audio, -1.0, 1.0)

    @staticmethod
    def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Simple linear interpolation resample for occasional sample-rate mismatch."""
        if orig_sr == target_sr:
            return audio.astype(np.float32, copy=False)
        if len(audio) == 0:
            return audio.astype(np.float32)

        ratio = target_sr / orig_sr
        new_length = max(1, int(len(audio) * ratio))
        return np.interp(
            np.linspace(0, len(audio) - 1, new_length),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)


_transcriber: Optional[Transcriber] = None


def get_transcriber() -> Transcriber:
    """Return the module-level singleton, creating it if needed."""
    global _transcriber
    if _transcriber is None:
        _transcriber = Transcriber()
    return _transcriber


def _smoke_test_offline():
    """Validate audio preprocessing and singleton behavior without loading the model."""
    import math

    logging.basicConfig(level=logging.INFO)
    logger.info("Running offline smoke test (pre-processing only)...")

    sr = 16000

    stereo_int16 = (np.random.randn(sr, 2) * 32767).astype(np.int16)
    result = Transcriber._validate_audio(stereo_int16)
    assert result is not None
    assert result.ndim == 1, f"Expected mono, got shape {result.shape}"
    assert result.dtype == np.float32
    assert result.max() <= 1.0
    assert result.min() >= -1.0
    logger.info("Stereo int16 to mono float32 normalization")

    mono_float = np.sin(2 * math.pi * 440 * np.linspace(0, 1, sr)).astype(np.float32)
    result = Transcriber._validate_audio(mono_float)
    assert result is not None and result.shape == (sr,)
    logger.info("Mono float32 passthrough")

    result = Transcriber._validate_audio(np.array([]))
    assert result is None
    logger.info("Empty input returns None")

    t1 = get_transcriber()
    t2 = get_transcriber()
    assert t1 is t2
    logger.info("Module singleton")

    logger.info("Offline smoke test PASSED")
    logger.info(
        "\nTo run the full model test:\n"
        "  python -c \"from pipeline.transcriber import get_transcriber; "
        "import numpy as np; t=get_transcriber(); "
        "audio=np.zeros(16000,dtype='float32'); print(repr(t.transcribe(16000,audio)))\""
    )


if __name__ == "__main__":
    _smoke_test_offline()
