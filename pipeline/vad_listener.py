"""
pipeline/vad_listener.py

Silero VAD-based speech listener.

Architecture
────────────
Gradio streams raw PCM audio from the browser microphone in small numpy
chunks.  This module wraps that stream in a stateful VADListener that:

  1. Feeds each chunk to Silero VADIterator (runs on CPU via ONNX).
  2. Accumulates PCM frames while speech is detected.
  3. On silence-end event, flushes the complete utterance into a
     thread-safe queue as a (sample_rate, np.ndarray) tuple.
  4. The orchestrator reads from that queue and passes each utterance
     to Cohere Transcribe.

Because Gradio's streaming callback fires on the main thread, all VAD
processing is synchronous and cheap (< 1 ms per 250 ms chunk on CPU).
No background threads are needed at this stage.

Usage
─────
    listener = VADListener()

    # Called by Gradio's streaming audio component on each chunk:
    for utterance in listener.process_chunk(sample_rate, audio_np):
        # utterance is (sample_rate, np.ndarray[float32]) — ready for ASR
        transcript = transcriber.transcribe(*utterance)

    # Reset between calls:
    listener.reset()
"""

import logging
import queue
from typing import Generator, Optional, Tuple

import numpy as np
import torch

from silero_vad import VADIterator, load_silero_vad

from config import (
    VAD_SAMPLE_RATE,
    VAD_CHUNK_SAMPLES,
    VAD_SILENCE_THRESHOLD,
    VAD_SILENCE_DURATION_S,
    VAD_MIN_SPEECH_S,
)

# Silero v6 requires exactly 512-sample windows at 16kHz (32 ms).
# We still accept larger Gradio chunks and stride through them internally.
VAD_WINDOW_SAMPLES = 512

logger = logging.getLogger(__name__)

# ── Types ─────────────────────────────────────────────────────────────────────
Utterance = Tuple[int, np.ndarray]   # (sample_rate, float32 PCM)


class VADListener:
    """
    Stateful VAD processor.  One instance per Gradio user session.

    The VADIterator state machine emits:
      {'start': sample_idx}  — speech onset detected
      {'end':   sample_idx}  — silence detected after speech (utterance done)
      None                   — mid-speech or mid-silence, keep accumulating

    We buffer raw PCM frames during the speech window and yield a complete
    utterance when 'end' fires.
    """

    def __init__(self):
        logger.info("Loading Silero VAD model (ONNX, CPU)…")
        self._model = load_silero_vad(onnx=True)     # ONNX = no PyTorch GPU mem

        # min_silence_duration_ms drives the 'end' event timing.
        # We use VAD_SILENCE_DURATION_S from config (default 0.8 s = 800 ms).
        self._vad = VADIterator(
            model=self._model,
            threshold=VAD_SILENCE_THRESHOLD,
            sampling_rate=VAD_SAMPLE_RATE,
            min_silence_duration_ms=int(VAD_SILENCE_DURATION_S * 1000),
            speech_pad_ms=60,       # 60 ms padding on each side — natural feel
        )

        self._speech_buffer: list[np.ndarray] = []  # accumulates speech frames
        self._speaking: bool = False

        # Internal queue — orchestrator can also pull from here if preferred
        self._utterance_queue: queue.SimpleQueue = queue.SimpleQueue()

        logger.info("Silero VAD ready.")

    # ── Public interface ──────────────────────────────────────────────────────

    def process_chunk(
        self, sample_rate: int, audio: np.ndarray
    ) -> Generator[Utterance, None, None]:
        """
        Accept one audio chunk from Gradio and yield zero or one utterance.

        Parameters
        ----------
        sample_rate : int
            Sample rate reported by Gradio (may differ from VAD_SAMPLE_RATE).
        audio : np.ndarray
            Raw PCM, any shape.  Will be normalised to mono float32 16 kHz.

        Yields
        ------
        (VAD_SAMPLE_RATE, np.ndarray[float32])
            Complete utterance, ready to be passed to Cohere Transcribe.
        """
        if audio is None or len(audio) == 0:
            return

        # ── 1. Normalise to mono float32 ──────────────────────────────────────
        audio = self._to_mono_float32(audio)

        # ── 2. Resample if Gradio gives a different rate ──────────────────────
        if sample_rate != VAD_SAMPLE_RATE:
            audio = self._resample(audio, sample_rate, VAD_SAMPLE_RATE)

        # ── 3. Pad to chunk boundary (Silero needs fixed-size windows) ────────
        audio = self._pad_to_chunk(audio)

        # ── 4. Walk through 512-sample windows (Silero v6 requirement) ─────────
        for i in range(0, len(audio), VAD_WINDOW_SAMPLES):
            window = audio[i : i + VAD_WINDOW_SAMPLES]
            if len(window) < VAD_WINDOW_SAMPLES:
                break   # incomplete trailing window — skip

            tensor = torch.from_numpy(window)
            event  = self._vad(tensor)

            if event is None:
                # Mid-speech: keep buffering.  Mid-silence: do nothing.
                if self._speaking:
                    self._speech_buffer.append(window)

            elif "start" in event:
                # Speech onset — start accumulating
                if not self._speaking:
                    logger.debug(f"Speech START at sample {event['start']}")
                    self._speaking = True
                    self._speech_buffer = [window]  # include onset window
                
            elif "end" in event:
                # Speech ended — flush if long enough
                if self._speaking:
                    self._speaking = False
                    self._speech_buffer.append(window)  # include trailing window
                    utterance = self._flush()
                    if utterance is not None:
                        logger.debug(
                            f"Utterance flushed: {len(utterance[1])/VAD_SAMPLE_RATE:.2f}s"
                        )
                        yield utterance

    def reset(self):
        """
        Call between calls / sessions to wipe VAD state and audio buffer.
        """
        self._vad.reset_states()
        self._speech_buffer = []
        self._speaking = False
        # drain the queue
        while not self._utterance_queue.empty():
            try:
                self._utterance_queue.get_nowait()
            except queue.Empty:
                break
        logger.debug("VADListener reset.")

    @property
    def is_speaking(self) -> bool:
        return self._speaking

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _flush(self) -> Optional[Utterance]:
        """
        Concatenate buffer frames into a single utterance array.
        Drops utterances shorter than VAD_MIN_SPEECH_S (noise / clicks).
        """
        if not self._speech_buffer:
            return None

        utterance = np.concatenate(self._speech_buffer, axis=0)
        duration  = len(utterance) / VAD_SAMPLE_RATE

        from config import VAD_MIN_SPEECH_S
        if duration < VAD_MIN_SPEECH_S:
            logger.debug(f"Utterance too short ({duration:.2f}s) — discarded.")
            self._speech_buffer = []
            return None

        self._speech_buffer = []
        return (VAD_SAMPLE_RATE, utterance)

    @staticmethod
    def _to_mono_float32(audio: np.ndarray) -> np.ndarray:
        """
        Convert any numpy audio array to mono float32 in [-1.0, 1.0].
        Gradio typically gives int16 or float32; handles both.
        """
        audio = np.array(audio, dtype=np.float32)

        # Stereo → mono
        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        # int16 range → float32 [-1, 1]
        if audio.max() > 1.0 or audio.min() < -1.0:
            audio = audio / 32768.0

        # Clip to valid range
        audio = np.clip(audio, -1.0, 1.0)
        return audio

    @staticmethod
    def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """
        Simple linear interpolation resample.
        Avoids importing torchaudio here (saves CUDA context startup cost).
        Good enough for pre-processing before VAD.
        """
        if orig_sr == target_sr:
            return audio
        ratio      = target_sr / orig_sr
        new_length = int(len(audio) * ratio)
        return np.interp(
            np.linspace(0, len(audio) - 1, new_length),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)

    @staticmethod
    def _pad_to_chunk(audio: np.ndarray) -> np.ndarray:
        """
        Zero-pad audio so its length is a multiple of VAD_CHUNK_SAMPLES.
        Silero requires fixed-size windows.
        """
        remainder = len(audio) % VAD_WINDOW_SAMPLES
        if remainder:
            padding = VAD_WINDOW_SAMPLES - remainder
            audio   = np.concatenate([audio, np.zeros(padding, dtype=np.float32)])
        return audio


# ── Standalone smoke test ─────────────────────────────────────────────────────

def _smoke_test():
    """
    Simulates a real mic stream:
      - 0.5 s silence
      - 1.5 s synthetic speech-like tone (500 Hz sine)
      - 0.9 s silence  ← should trigger 'end' and yield one utterance
      - 0.5 s silence
    """
    import math
    logging.basicConfig(level=logging.DEBUG)

    SR   = VAD_SAMPLE_RATE
    FREQ = 500            # Hz — sine at 500 Hz is reliably detected as speech

    def sine_chunk(duration_s: float, frequency: float = 0) -> np.ndarray:
        n = int(SR * duration_s)
        if frequency == 0:
            return np.zeros(n, dtype=np.float32)
        t = np.linspace(0, duration_s, n, endpoint=False)
        return (0.6 * np.sin(2 * math.pi * frequency * t)).astype(np.float32)

    timeline = [
        sine_chunk(0.5),          # silence
        sine_chunk(1.5, FREQ),    # speech
        sine_chunk(0.9),          # silence — should flush
        sine_chunk(0.5),          # trailing silence
    ]

    listener = VADListener()
    utterances_received = 0

    for chunk in timeline:
        # Feed in 250 ms sub-chunks as Gradio would
        for start in range(0, len(chunk), VAD_CHUNK_SAMPLES):
            sub = chunk[start : start + VAD_CHUNK_SAMPLES]
            for utt in listener.process_chunk(SR, sub):
                utterances_received += 1
                duration = len(utt[1]) / utt[0]
                print(f"  ✓ Utterance received: {duration:.2f}s at {utt[0]} Hz")

    print(f"\nSmoke test complete — {utterances_received} utterance(s) flushed.")
    assert utterances_received >= 1, "Expected at least one utterance!"
    print("PASSED ✓")


if __name__ == "__main__":
    _smoke_test()