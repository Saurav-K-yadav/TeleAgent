"""
pipeline/orchestrator.py

Orchestrator for the telecalling agent pipeline.

Coordinates the end-to-end flow:
  1. VAD listener        → detects speech boundaries
  2. Transcriber         → audio-to-text (Moonshine)
  3. Intent parser       → structured intent extraction (Qwen2.5)
  4. Evaluator           → scheduling decision + spoken response (MiniCPM3)
  5. Database updates    → persist call state

This module is the glue between Gradio's audio stream and the ML models.
It maintains call state across multiple audio chunks and orchestrates
lazy-loaded model lifecycle.

Usage
─────
    session = CallSession()
    session.start_call()

    # Called by Gradio on each audio chunk:
    update = session.process_audio_chunk(sample_rate, audio_chunk)
    # use update to refresh UI

    session.end_call()
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from config import DB_PATH
from db import (
    create_call,
    append_transcript,
    update_call_intent,
    update_call_decision,
    is_slot_available,
    get_booked_slots,
    get_recent_calls,
)
from pipeline.transcriber import get_transcriber
from pipeline.intent_parser import get_intent_parser
from pipeline.evaluater import get_evaluator

logger = logging.getLogger(__name__)


# ── Mock Transcriber — fallback when PyTorch/dependencies missing ──────────────

class MockTranscriber:
    """Fallback mock transcriber for testing without PyTorch."""

    def __init__(self):
        self.is_loaded = True
        self.device = "mock"
        self._call_count = 0

    def transcribe(self, sample_rate: int, audio: np.ndarray) -> str:
        """Return placeholder transcriptions for testing."""
        self._call_count += 1
        responses = [
            "I'd like to book a meeting",
            "Can we schedule for next Monday at 2 PM",
            "Yes, that works for me",
            "Please confirm my booking",
            "Thank you, goodbye",
        ]
        return responses[self._call_count % len(responses)]

    def transcribe_batch(self, utterances: list) -> list[str]:
        return [self.transcribe(sr, audio) for sr, audio in utterances]

    def unload(self):
        pass


def get_mock_transcriber() -> MockTranscriber:
    """Get mock transcriber instance."""
    return MockTranscriber()


# ── MockVADListener — always available as fallback ────────────────────────────

class MockVADListener:
    """Fallback mock VAD for testing without onnxruntime or torch/numpy issues."""

    def __init__(self):
        self.is_speaking = False
        self._buffer = []
        self._chunk_count = 0

    def process_chunk(self, sample_rate: int, audio):
        """Simulate speech detection using simple chunk counting."""
        if audio is None or len(audio) == 0:
            return

        try:
            audio_np = np.asarray(audio, dtype=np.float32)
        except Exception:
            return

        has_energy = float(np.max(np.abs(audio_np))) > 0.01

        if has_energy:
            self.is_speaking = True
            self._buffer.append(audio_np)
            self._chunk_count += 1

            if self._chunk_count >= 8:
                try:
                    utterance = np.concatenate(self._buffer, axis=0)
                    self._buffer = []
                    self._chunk_count = 0
                    self.is_speaking = False
                    yield (sample_rate, utterance)
                except Exception as e:
                    logger.debug(f"MockVAD concatenate failed: {e}")
                    self._buffer = []
                    self._chunk_count = 0
        else:
            if self._buffer:
                try:
                    utterance = np.concatenate(self._buffer, axis=0)
                    self._buffer = []
                    self._chunk_count = 0
                    self.is_speaking = False
                    yield (sample_rate, utterance)
                except Exception as e:
                    logger.debug(f"MockVAD flush failed: {e}")
                    self._buffer = []
                    self._chunk_count = 0

    def reset(self):
        self._buffer = []
        self._chunk_count = 0
        self.is_speaking = False


# ── Try to import real VADListener; fallback to mock if needed ─────────────────

try:
    from pipeline.vad_listener import VADListener
except Exception as e:
    logger.warning(f"VADListener import failed ({type(e).__name__}: {e}) — will use MockVADListener")
    VADListener = MockVADListener


# ── Output types ──────────────────────────────────────────────────────────────

@dataclass
class PipelineUpdate:
    """Update to display in the Gradio UI after each processing step."""
    status: str                              # status badge text
    vad_speaking: bool                       # True if mic active
    transcript_lines: list[str]              # accumulated utterances
    intent_md: str                           # markdown table of intent
    agent_response: str                      # what the agent said
    booking_confirmed: Optional[dict]        # booking details if confirmed
    call_log: list[dict]                     # recent call records


@dataclass
class _OrchestrationResponse:
    """Internal response from pipeline processing."""
    spoken_text: str                         # text to say back to caller
    call_id: int                             # database row id
    is_terminal: bool                        # True if call should end
    interim_intent: Optional[str] = None     # intent extracted so far
    booking_info: Optional[dict] = None      # booking details if confirmed


# ── Session orchestrator ──────────────────────────────────────────────────────

class CallSession:
    """
    Stateful session for one telecalling interaction in Gradio.

    Lifecycle
    ─────────
    1. __init__() — create empty, not yet active
    2. start_call() → activates VAD listener, creates database record
    3. process_audio_chunk() — repeatedly as Gradio streams audio
    4. end_call() — cleanup and reset for next call
    5. reset() — full session reset

    All methods return PipelineUpdate to refresh the UI.
    """

    def __init__(self):
        """Initialize an inactive session."""
        self._call_active = False
        self._call_id = None
        self._vad = None

        # Try to load real transcriber; fall back to mock on import error
        try:
            self._transcriber = get_transcriber()
        except (ImportError, ModuleNotFoundError) as e:
            logger.warning(f"Transcriber import failed ({e}) — using MockTranscriber")
            self._transcriber = get_mock_transcriber()

        try:
            self._parser = get_intent_parser()
        except (ImportError, ModuleNotFoundError) as e:
            logger.warning(f"Parser import failed ({e})")
            self._parser = None

        try:
            self._evaluator = get_evaluator()
        except (ImportError, ModuleNotFoundError) as e:
            logger.warning(f"Evaluator import failed ({e})")
            self._evaluator = None

        self._utterances = []
        self._current_intent = None
        self._last_response = ""
        self._booking_info = None

        logger.info("CallSession created (inactive)")

    @property
    def call_active(self) -> bool:
        """Whether a call is currently in progress."""
        return self._call_active

    # ── Lifecycle methods ──────────────────────────────────────────────────────

    def start_call(self) -> PipelineUpdate:
        """Start a new call session."""
        self._call_active = True

        # Try to initialize real VADListener; fall back to mock on error
        try:
            self._vad = VADListener()
        except (RuntimeError, ValueError) as e:
            logger.warning(f"VADListener init failed at runtime ({e}) — using MockVADListener")
            self._vad = MockVADListener()

        self._utterances = []
        self._current_intent = None
        self._last_response = ""
        self._booking_info = None

        self._call_id = create_call("gradio_session")
        logger.info(f"Call started — id={self._call_id}")

        return self._build_update()

    def end_call(self) -> PipelineUpdate:
        """End the current call."""
        if not self._call_active:
            return self._build_update()

        self._call_active = False
        if self._vad:
            self._vad.reset()

        logger.info(f"Call ended — id={self._call_id}")
        return self._build_update()

    def reset(self) -> PipelineUpdate:
        """Full session reset."""
        self.end_call()
        self._call_id = None
        self._utterances = []
        self._current_intent = None
        self._last_response = ""
        self._booking_info = None
        return self._build_update()

    # ── Audio processing ──────────────────────────────────────────────────────

    def process_audio_chunk(
        self, sample_rate: int, audio: np.ndarray
    ) -> PipelineUpdate:
        """
        Process one audio chunk from Gradio microphone.

        If a complete utterance is detected (speech → silence), runs the
        full pipeline: transcribe → parse intent → evaluate → update DB.
        """
        if not self._call_active or not self._vad or not self._call_id:
            return self._build_update()

        # ── Step 1: VAD — detect utterance boundaries ──────────────────────────
        try:
            utterances_detected = list(self._vad.process_chunk(sample_rate, audio))
        except RuntimeError as e:
            # VAD failed at runtime (e.g., torch/numpy issue) — fall back to mock
            if not isinstance(self._vad, MockVADListener):
                logger.warning(
                    f"VAD process_chunk failed ({e}) — switching to MockVADListener"
                )
                self._vad = MockVADListener()
                try:
                    utterances_detected = list(self._vad.process_chunk(sample_rate, audio))
                except Exception as e2:
                    logger.error(f"Mock VAD also failed: {e2}")
                    return self._build_update()
            else:
                logger.error(f"Mock VAD failed: {e}")
                return self._build_update()

        if not utterances_detected:
            return self._build_update()

        # ── Steps 2–5: process each utterance ──────────────────────────────────
        for sample_rate_utt, audio_np in utterances_detected:
            self._process_utterance(sample_rate_utt, audio_np)

        return self._build_update()

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _process_utterance(self, sample_rate: int, audio: np.ndarray) -> None:
        """Process one complete utterance through the full pipeline."""
        # Transcribe
        try:
            transcript_chunk = self._transcriber.transcribe(sample_rate, audio)
        except (RuntimeError, ImportError) as e:
            # Model failed to load at runtime — fall back to mock
            if not isinstance(self._transcriber, MockTranscriber):
                logger.warning(f"Transcriber failed ({e}) — switching to MockTranscriber")
                self._transcriber = get_mock_transcriber()
                transcript_chunk = self._transcriber.transcribe(sample_rate, audio)
            else:
                logger.error(f"MockTranscriber also failed: {e}")
                return

        if not transcript_chunk or not transcript_chunk.strip():
            logger.debug("Transcription empty — skipping")
            return

        append_transcript(self._call_id, transcript_chunk)
        self._utterances.append(transcript_chunk)
        logger.info(f"Transcribed: '{transcript_chunk[:80]}…'")

        # Parse intent
        if self._parser:
            try:
                self._current_intent = self._parser.parse_accumulated(self._utterances)
                update_call_intent(self._call_id, self._current_intent.model_dump())
            except Exception as e:
                logger.warning(f"Intent parsing failed: {e}")
                return
        else:
            logger.debug("Parser not available — skipping intent parse")
            return

        # Evaluate
        if not self._evaluator:
            logger.debug("Evaluator not available — skipping evaluation")
            return

        slot_available = True
        slot_conflicts = []

        if (self._current_intent.preferred_date and
            self._current_intent.preferred_time):
            slot_available = is_slot_available(
                self._current_intent.preferred_date,
                self._current_intent.preferred_time,
                self._current_intent.duration_minutes or 30,
            )
            if not slot_available:
                slot_conflicts = get_booked_slots(self._current_intent.preferred_date)

        try:
            result = self._evaluator.evaluate(
                self._current_intent,
                self._utterances,
                slot_available=slot_available,
                slot_conflicts=slot_conflicts,
            )
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            result = None

        if result:
            update_call_decision(self._call_id, result.decision, result.reasoning)
            self._last_response = result.spoken_response

            # Handle booking confirmation
            if result.decision == "schedule":
                self._booking_info = {
                    "booking_id": self._call_id,
                    "caller": self._current_intent.caller_name or "Unknown",
                    "date": self._current_intent.preferred_date,
                    "time": self._current_intent.preferred_time,
                    "duration": self._current_intent.duration_minutes or 30,
                    "type": self._current_intent.meeting_type or "phone",
                }

            # Terminal decision — end call
            if result.decision == "end_call":
                self._call_active = False

    def _build_update(self) -> PipelineUpdate:
        """Build UI update from current state."""
        status = "🟢 Active" if self._call_active else "⚫ Ready"
        vad_speaking = self._vad.is_speaking if self._vad else False

        # Format intent as markdown table
        intent_md = self._format_intent_md()

        # Call log
        call_log = get_recent_calls(limit=10)

        return PipelineUpdate(
            status=status,
            vad_speaking=vad_speaking,
            transcript_lines=self._utterances,
            intent_md=intent_md,
            agent_response=self._last_response,
            booking_confirmed=self._booking_info,
            call_log=call_log,
        )

    def _format_intent_md(self) -> str:
        """Format current intent as markdown table."""
        if not self._current_intent:
            return "_No data yet — waiting for first utterance…_"

        intent = self._current_intent
        rows = [
            f"| Field | Value |",
            f"|-------|-------|",
            f"| Intent | `{intent.intent}` |",
            f"| Caller | {intent.caller_name or '—'} |",
            f"| Date | {intent.preferred_date or '—'} |",
            f"| Time | {intent.preferred_time or '—'} |",
            f"| Duration | {intent.duration_minutes or '—'} min |",
            f"| Type | {intent.meeting_type or '—'} |",
            f"| Confidence | {intent.confidence:.1%} |",
            f"| Missing | {', '.join(intent.missing_fields) or '—'} |",
        ]
        return "\n".join(rows)
