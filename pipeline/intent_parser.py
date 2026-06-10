"""
pipeline/intent_parser.py

Qwen2.5-7B-Instruct (Q4_K_M GGUF) intent & entity extractor.

Responsibilities
────────────────
Takes a raw transcript string from Moonshine ASR and returns a
validated SchedulingIntent object — structured data the evaluator and
DB layer can act on directly.

Why GGUF + llama-cpp-python
────────────────────────────
  - Qwen2.5-7B-Instruct in Q4_K_M needs ~4.5 GB total:
      20 layers on RTX 2050 GPU  → ~0.8 GB VRAM
      remaining ~15 layers on CPU RAM → ~3.7 GB RAM
  - llama-cpp-python's grammar feature forces output to be valid JSON
    with no post-processing hacks — zero hallucinated keys.

GBNF Grammar
────────────
llama.cpp supports GBNF (Generative BNF) grammars that constrain the
token sampling so the model physically cannot produce invalid JSON.
We define a tight grammar matching exactly our SchedulingIntent schema.

Download the GGUF before running (one-time, ~4.5 GB):
  huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF \\
      --include "qwen2.5-7b-instruct-q4_k_m.gguf" \\
      --local-dir ./models
"""

import json
import logging
import threading
import time
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from config import (
    QWEN_GGUF_PATH,
    QWEN_N_GPU_LAYERS,
    QWEN_N_CTX,
    QWEN_MAX_TOKENS,
    QWEN_TEMPERATURE,
)

logger = logging.getLogger(__name__)


# ── Output schema ─────────────────────────────────────────────────────────────

class SchedulingIntent(BaseModel):
    """
    Structured output from Qwen2.5.  Every field is Optional so partial
    information (e.g. caller gave name but no date yet) is represented
    cleanly rather than hallucinated.
    """
    intent: str = Field(
        description=(
            "One of: book_meeting | reschedule | cancel | "
            "check_availability | provide_info | end_call | unclear"
        )
    )
    caller_name:      Optional[str] = Field(None, description="Full name of the caller")
    preferred_date:   Optional[str] = Field(None, description="YYYY-MM-DD or null")
    preferred_time:   Optional[str] = Field(None, description="HH:MM 24-hour or null")
    duration_minutes: Optional[int] = Field(None, description="Meeting length in minutes")
    participants:     list[str]     = Field(default_factory=list)
    meeting_type:     Optional[str] = Field(None, description="phone | video | in_person")
    notes:            Optional[str] = Field(None, description="Any extra context")
    confidence:       float         = Field(0.0, description="0.0–1.0 extraction confidence")
    missing_fields:   list[str]     = Field(
        default_factory=list,
        description="Fields still needed to complete booking"
    )

    @field_validator("intent")
    @classmethod
    def validate_intent(cls, v: str) -> str:
        valid = {
            "book_meeting", "reschedule", "cancel",
            "check_availability", "provide_info", "end_call", "unclear"
        }
        return v if v in valid else "unclear"

    @field_validator("preferred_date")
    @classmethod
    def validate_date(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        import re
        if re.match(r"^\d{4}-\d{2}-\d{2}$", v):
            return v
        return None   # reject malformed dates

    @field_validator("preferred_time")
    @classmethod
    def validate_time(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        import re
        if re.match(r"^\d{2}:\d{2}$", v):
            return v
        return None

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    def compute_missing(self) -> "SchedulingIntent":
        """
        Populate missing_fields based on what a booking actually needs.
        Returns self for chaining.
        """
        needed = []
        if self.intent == "book_meeting":
            if not self.caller_name:    needed.append("caller_name")
            if not self.preferred_date: needed.append("preferred_date")
            if not self.preferred_time: needed.append("preferred_time")
        self.missing_fields = needed
        return self


# ── GBNF grammar ──────────────────────────────────────────────────────────────
# Constrains Qwen to produce ONLY valid JSON matching our schema.
# llama.cpp samples tokens that satisfy this grammar at every step.

_SCHEDULING_GRAMMAR = r"""
root   ::= ws "{" ws "\"intent\"" ws ":" ws intent-val ws "," ws "\"caller_name\"" ws ":" ws str-or-null ws "," ws "\"preferred_date\"" ws ":" ws str-or-null ws "," ws "\"preferred_time\"" ws ":" ws str-or-null ws "," ws "\"duration_minutes\"" ws ":" ws int-or-null ws "," ws "\"participants\"" ws ":" ws str-array ws "," ws "\"meeting_type\"" ws ":" ws meeting-type-val ws "," ws "\"notes\"" ws ":" ws str-or-null ws "," ws "\"confidence\"" ws ":" ws confidence-val ws "," ws "\"missing_fields\"" ws ":" ws str-array ws "}" ws

intent-val ::= "\"book_meeting\"" | "\"reschedule\"" | "\"cancel\"" | "\"check_availability\"" | "\"provide_info\"" | "\"end_call\"" | "\"unclear\""

meeting-type-val ::= "\"phone\"" | "\"video\"" | "\"in_person\"" | "null"

str-or-null ::= string | "null"
int-or-null ::= integer | "null"

confidence-val ::= ("0" | "1") | ("0" "." [0-9]+) | ("1" "." [0]*)

str-array ::= "[" ws "]" | "[" ws string (ws "," ws string)* ws "]"

string ::= "\"" ([^"\\] | "\\\\" ["\\\\/bfnrt] | "\\u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])* "\""
integer ::= ("-"?) [0-9]+
number  ::= integer ("." [0-9]+)?
bool    ::= "true" | "false"
null    ::= "null"
array   ::= "[" ws (number (ws "," ws number)*)? ws "]"
ws      ::= [ \t\n\r]*
"""


# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a scheduling assistant that extracts structured information from call transcripts.

Extract ONLY what the caller explicitly said. Never invent information.
Return a single JSON object with these exact fields:
- intent: what the caller wants (book_meeting/reschedule/cancel/check_availability/provide_info/end_call/unclear)
- caller_name: their name if given, else null
- preferred_date: YYYY-MM-DD if mentioned, else null. Resolve relative dates (e.g. "tomorrow", "next Monday") to absolute dates based on today being {today}.
- preferred_time: HH:MM 24-hour if mentioned, else null. Convert "3pm" → "15:00", "9:30 in the morning" → "09:30".
- duration_minutes: integer if mentioned, else null. Default 30 if they say "quick meeting".
- participants: list of names mentioned besides the caller
- meeting_type: "phone", "video", "in_person", or null
- notes: any other relevant context, else null
- confidence: 0.0–1.0 reflecting how complete the information is
- missing_fields: list of field names still needed for a complete booking
"""

_USER_PROMPT_TEMPLATE = """Transcript:
\"\"\"{transcript}\"\"\"

JSON:"""


# ── Parser class ──────────────────────────────────────────────────────────────

class IntentParser:
    """
    Lazy-loading Qwen2.5-7B-Instruct GGUF intent extractor.
    Thread-safe singleton pattern — one model instance for the whole app.
    """

    def __init__(self):
        self._llm     = None
        self._grammar = None
        self._lock    = threading.Lock()
        self._loaded  = False

    # ── Public ────────────────────────────────────────────────────────────────

    def parse(self, transcript: str) -> SchedulingIntent:
        """
        Extract scheduling intent from a transcript string.

        Parameters
        ----------
        transcript : str
            Raw text from Moonshine ASR (one or more utterances joined).

        Returns
        -------
        SchedulingIntent
            Validated Pydantic model.  Falls back to intent="unclear" on
            any parse failure so the pipeline never crashes.
        """
        if not transcript or not transcript.strip():
            return self._fallback("Empty transcript")

        self._ensure_loaded()

        prompt = self._build_prompt(transcript)

        try:
            t0 = time.perf_counter()

            response = self._llm(
                prompt,
                max_tokens  = QWEN_MAX_TOKENS,
                temperature = QWEN_TEMPERATURE,
                stop        = ["\n}\n", "```"],   # belt-and-suspenders stop
                grammar     = self._grammar,
                echo        = False,
            )

            elapsed  = time.perf_counter() - t0
            raw_text = response["choices"][0]["text"].strip()

            logger.info(f"Qwen inference in {elapsed:.2f}s — raw: {raw_text[:120]}…")

            return self._parse_response(raw_text, transcript)

        except Exception as exc:
            logger.error(f"IntentParser.parse failed: {exc}", exc_info=True)
            return self._fallback(str(exc))

    def parse_accumulated(self, utterances: list[str]) -> SchedulingIntent:
        """
        Parse the full conversation so far (list of utterance strings).
        Joins them with newlines and runs a single inference pass.
        Use this after each new utterance to get an updated intent state.
        """
        full_transcript = "\n".join(u for u in utterances if u.strip())
        return self.parse(full_transcript)

    def unload(self):
        """Free memory — model reloads lazily on next call."""
        with self._lock:
            if self._loaded:
                del self._llm
                self._llm    = None
                self._loaded = False
                logger.info("IntentParser unloaded.")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ── Internal ──────────────────────────────────────────────────────────────

    def _ensure_loaded(self):
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            self._load()

    def _load(self):
        try:
            from llama_cpp import Llama, LlamaGrammar
        except ImportError:
            raise RuntimeError(
                "llama-cpp-python is not installed.\n"
                "Install with CUDA support:\n"
                "  Windows: set CMAKE_ARGS=-DGGML_CUDA=on && "
                "pip install llama-cpp-python --no-cache-dir\n"
                "  Linux:   CMAKE_ARGS='-DGGML_CUDA=on' "
                "pip install llama-cpp-python --no-cache-dir"
            )

        if not QWEN_GGUF_PATH.exists():
            raise FileNotFoundError(
                f"GGUF not found at {QWEN_GGUF_PATH}\n"
                "Download with:\n"
                "  huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF \\\n"
                "      --include 'qwen2.5-7b-instruct-q4_k_m.gguf' \\\n"
                "      --local-dir ./models"
            )

        logger.info(
            f"Loading Qwen2.5-7B Q4_K_M — "
            f"{QWEN_N_GPU_LAYERS} layers on GPU, rest on CPU…"
        )
        t0 = time.perf_counter()

        self._llm = Llama(
            model_path    = str(QWEN_GGUF_PATH),
            n_gpu_layers  = QWEN_N_GPU_LAYERS,   # 20 → ~0.8 GB VRAM
            n_ctx         = QWEN_N_CTX,           # 4096 tokens
            n_threads     = 6,                    # leave 2 cores for Gradio
            n_batch       = 512,
            verbose       = False,
        )

        self._grammar = LlamaGrammar.from_string(_SCHEDULING_GRAMMAR)

        elapsed = time.perf_counter() - t0
        logger.info(f"Qwen2.5 ready in {elapsed:.1f}s")
        self._loaded = True

    def _build_prompt(self, transcript: str) -> str:
        """
        Qwen2.5-Instruct uses ChatML format:
          <|im_start|>system\n…<|im_end|>\n
          <|im_start|>user\n…<|im_end|>\n
          <|im_start|>assistant\n
        The grammar then forces the assistant turn to be valid JSON.
        """
        from datetime import date
        today = date.today().strftime("%Y-%m-%d")

        system = _SYSTEM_PROMPT.format(today=today)
        user   = _USER_PROMPT_TEMPLATE.format(transcript=transcript.strip())

        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def _parse_response(self, raw: str, original_transcript: str) -> SchedulingIntent:
        """
        Parse and validate Qwen's JSON output.
        The grammar guarantees structural validity; Pydantic validates values.
        """
        # Strip any accidental markdown fences
        raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()

        try:
            data   = json.loads(raw)
            intent = SchedulingIntent(**data).compute_missing()
            logger.info(
                f"Parsed intent={intent.intent} "
                f"name={intent.caller_name} "
                f"date={intent.preferred_date} "
                f"time={intent.preferred_time} "
                f"confidence={intent.confidence:.2f} "
                f"missing={intent.missing_fields}"
            )
            return intent

        except Exception as exc:
            logger.warning(f"JSON parse/validation failed: {exc} — raw was: {raw[:200]}")
            return self._fallback(str(exc))

    @staticmethod
    def _fallback(reason: str) -> SchedulingIntent:
        logger.warning(f"Returning fallback intent — reason: {reason}")
        return SchedulingIntent(
            intent     = "unclear",
            confidence = 0.0,
        ).compute_missing()


# ── Module singleton ──────────────────────────────────────────────────────────

_parser: Optional[IntentParser] = None


def get_intent_parser() -> IntentParser:
    global _parser
    if _parser is None:
        _parser = IntentParser()
    return _parser


# ── Offline smoke test ────────────────────────────────────────────────────────

def _smoke_test_offline():
    """Tests schema, validators, and prompt building without loading the model."""
    logging.basicConfig(level=logging.INFO)
    logger.info("Running IntentParser offline smoke test…")

    # 1. Valid full intent
    intent = SchedulingIntent(
        intent           = "book_meeting",
        caller_name      = "Priya Sharma",
        preferred_date   = "2026-06-10",
        preferred_time   = "14:00",
        duration_minutes = 30,
        participants     = ["Priya Sharma"],
        meeting_type     = "video",
        notes            = None,
        confidence       = 0.95,
    ).compute_missing()
    assert intent.missing_fields == [], f"Expected no missing fields, got {intent.missing_fields}"
    logger.info("  ✓ Full booking intent — no missing fields")

    # 2. Partial intent — date and time missing
    partial = SchedulingIntent(
        intent      = "book_meeting",
        caller_name = "Raj",
        confidence  = 0.4,
    ).compute_missing()
    assert "preferred_date" in partial.missing_fields
    assert "preferred_time" in partial.missing_fields
    logger.info(f"  ✓ Partial intent missing fields: {partial.missing_fields}")

    # 3. Invalid intent string → coerced to "unclear"
    coerced = SchedulingIntent(intent="nonsense", confidence=0.1)
    assert coerced.intent == "unclear"
    logger.info("  ✓ Invalid intent string coerced to 'unclear'")

    # 4. Malformed date → None
    bad_date = SchedulingIntent(intent="book_meeting", preferred_date="June 10th")
    assert bad_date.preferred_date is None
    logger.info("  ✓ Malformed date rejected → None")

    # 5. Confidence clamping
    clamped = SchedulingIntent(intent="unclear", confidence=999.0)
    assert clamped.confidence == 1.0
    logger.info("  ✓ Confidence clamped to 1.0")

    # 6. Prompt build
    parser = IntentParser()
    prompt = parser._build_prompt("Hi I want to book a meeting tomorrow at 3pm")
    assert "<|im_start|>system" in prompt
    assert "2026-" in prompt   # today injected
    assert "tomorrow" in prompt
    logger.info("  ✓ Prompt format correct (ChatML)")

    # 7. Fallback
    fallback = IntentParser._fallback("test")
    assert fallback.intent == "unclear"
    logger.info("  ✓ Fallback returns 'unclear' intent")

    logger.info("\nOffline smoke test PASSED ✓")


if __name__ == "__main__":
    _smoke_test_offline()
