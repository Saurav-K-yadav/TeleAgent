"""
pipeline/evaluator.py

MiniCPM3-4B scheduling evaluator.

Responsibilities
────────────────
Takes a validated SchedulingIntent from the intent parser and decides:
  - "schedule"      → all fields present, slot is free → confirm booking
  - "ask_followup"  → fields missing OR slot taken → ask caller a question
  - "reject"        → invalid request (weekend, out of hours, cancel confirmed)
  - "end_call"      → caller said goodbye

Also generates a natural spoken response the agent will display
(and optionally TTS in future) back to the caller.

Memory strategy for RTX 2050
─────────────────────────────
MiniCPM3-4B runs AFTER the transcriber finishes each utterance.
They never overlap. MiniCPM3 loads in 4-bit (INT4) via bitsandbytes
on CPU, freeing the full 4 GB budget for Moonshine ASR.

If bitsandbytes is unavailable (e.g. first-time setup) we fall back to
plain float32 on CPU. Slower but always works.

Why MiniCPM3-4B for this role
───────────────────────────────
  - Optimised for deep reasoning with structured JSON output
  - 4B params → ~2.5 GB RAM in INT4; ~8 GB in float32
  - Strong at rule-following (our scheduling rules inject cleanly)
  - Fast enough on CPU for the non-latency-critical decision step
"""

import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

import torch

from config import (
    MINICPM_MODEL_ID,
    MINICPM_DEVICE,
    MINICPM_MAX_TOKENS,
    SCHEDULING_RULES,
)
from pipeline.intent_parser import SchedulingIntent

logger = logging.getLogger(__name__)


# ── Output type ───────────────────────────────────────────────────────────────

@dataclass
class EvaluationResult:
    decision:         str            # schedule | ask_followup | reject | end_call
    spoken_response:  str            # what the agent says back to the caller
    reasoning:        str            # internal chain-of-thought (for DB + debug)
    suggested_date:   Optional[str]  # YYYY-MM-DD if rescheduling to next free slot
    suggested_time:   Optional[str]  # HH:MM if rescheduling


# ── Prompts ───────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an AI scheduling assistant on a live phone call.

SCHEDULING RULES:
{rules}

Your job: given the caller's current intent data and the conversation so far,
decide what to do next and generate a SHORT, natural spoken response.

Respond ONLY with a JSON object — no prose outside the JSON:
{{
  "decision": "schedule" | "ask_followup" | "reject" | "end_call",
  "spoken_response": "what you say to the caller (1-2 sentences, conversational)",
  "reasoning": "brief internal note explaining your decision",
  "suggested_date": "YYYY-MM-DD or null",
  "suggested_time": "HH:MM or null"
}}

Guidelines:
- spoken_response must sound natural on a phone call. No bullet points.
- If asking a follow-up, ask for ONE missing field at a time.
- If the slot is taken, suggest the next available slot from slot_conflicts.
- If intent is "end_call" or caller says goodbye, set decision to "end_call".
- Keep spoken_response under 30 words.
"""

_USER_PROMPT = """Intent data:
{intent_json}

Slot available: {slot_available}
Slot conflicts (booked slots on that date): {slot_conflicts}
Conversation history (last 6 utterances):
{history}

JSON decision:"""


# ── Evaluator ─────────────────────────────────────────────────────────────────

class Evaluator:
    """
    Lazy-loading MiniCPM3-4B evaluator.  CPU-only, INT4 quantised.
    """

    def __init__(self):
        self._model     = None
        self._tokenizer = None
        self._lock      = threading.Lock()
        self._loaded    = False

    # ── Public ────────────────────────────────────────────────────────────────

    def evaluate(
        self,
        intent:        SchedulingIntent,
        utterances:    list[str],
        slot_available: bool             = True,
        slot_conflicts: list[dict]       = None,
    ) -> EvaluationResult:
        """
        Evaluate a scheduling intent and generate a spoken response.

        Parameters
        ----------
        intent         : validated SchedulingIntent from intent_parser
        utterances     : full list of transcript strings so far
        slot_available : result of db.is_slot_available()
        slot_conflicts : result of db.get_booked_slots() for that date

        Returns
        -------
        EvaluationResult — never raises, falls back gracefully on error
        """
        # Fast-path: handle terminal intents without model inference
        fast = self._fast_path(intent)
        if fast:
            return fast

        self._ensure_loaded()
        
        # If model failed to load, use fallback result
        if self._model is None:
            logger.debug("MiniCPM3 disabled — using fallback result")
            return self._fallback_result(intent)

        prompt = self._build_prompt(intent, utterances, slot_available, slot_conflicts or [])

        try:
            t0  = time.perf_counter()
            raw = self._generate(prompt)
            elapsed = time.perf_counter() - t0
            logger.info(f"MiniCPM3 inference in {elapsed:.2f}s — raw: {raw[:120]}")
            return self._parse_response(raw, intent)

        except Exception as exc:
            logger.error(f"Evaluator.evaluate failed: {exc}", exc_info=True)
            return self._fallback_result(intent)

    def unload(self):
        with self._lock:
            if self._loaded:
                del self._model, self._tokenizer
                self._model     = None
                self._tokenizer = None
                self._loaded    = False
                logger.info("Evaluator unloaded.")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ── Internal ──────────────────────────────────────────────────────────────

    def _fast_path(self, intent: SchedulingIntent) -> Optional[EvaluationResult]:
        """
        Handle clear-cut cases without burning inference time.
        """
        if intent.intent == "end_call":
            return EvaluationResult(
                decision        = "end_call",
                spoken_response = "Thank you for calling. Goodbye!",
                reasoning       = "Caller indicated end of call.",
                suggested_date  = None,
                suggested_time  = None,
            )

        if intent.intent == "unclear" or intent.confidence < 0.15:
            return EvaluationResult(
                decision        = "ask_followup",
                spoken_response = (
                    "I'm sorry, I didn't quite catch that. "
                    "Could you tell me what you'd like to schedule?"
                ),
                reasoning       = f"Intent unclear or confidence too low ({intent.confidence:.2f}).",
                suggested_date  = None,
                suggested_time  = None,
            )

        return None   # fall through to model inference

    def _ensure_loaded(self):
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            self._load()

    def _load(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        logger.info(f"Loading MiniCPM3-4B ({MINICPM_MODEL_ID}) on CPU…")
        t0 = time.perf_counter()

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                MINICPM_MODEL_ID,
                trust_remote_code=True,
            )

            # Try INT4 via bitsandbytes first; fall back to float32
            try:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit              = True,
                    bnb_4bit_compute_dtype    = torch.float32,
                    bnb_4bit_use_double_quant = True,
                    bnb_4bit_quant_type       = "nf4",
                )
                self._model = AutoModelForCausalLM.from_pretrained(
                    MINICPM_MODEL_ID,
                    quantization_config = bnb_config,
                    trust_remote_code   = True,
                    device_map          = "cpu",
                    low_cpu_mem_usage   = True,
                )
                logger.info("MiniCPM3 loaded in INT4 (bitsandbytes NF4)")

            except (ImportError, Exception) as e:
                logger.warning(f"bitsandbytes INT4 failed ({e}) — loading float32 on CPU")
                self._model = AutoModelForCausalLM.from_pretrained(
                    MINICPM_MODEL_ID,
                    torch_dtype       = torch.float32,
                    trust_remote_code = True,
                    low_cpu_mem_usage = True,
                )
                self._model.to("cpu")

            self._model.eval()

            elapsed = time.perf_counter() - t0
            logger.info(f"MiniCPM3 ready in {elapsed:.1f}s")
            self._loaded = True

        except Exception as e:
            logger.error(
                f"MiniCPM3 failed to load: {e}\n"
                f"MiniCPM3 will be DISABLED. The pipeline will still work for transcription and intent parsing."
            )
            # Mark as loaded but with model=None so evaluate() knows to skip
            self._loaded = True
            self._model = None
            self._tokenizer = None

    def _build_prompt(
        self,
        intent:         SchedulingIntent,
        utterances:     list[str],
        slot_available: bool,
        slot_conflicts: list[dict],
    ) -> str:
        """
        MiniCPM3 uses ChatML format identical to Qwen:
          <|im_start|>system … <|im_end|>
          <|im_start|>user   … <|im_end|>
          <|im_start|>assistant
        """
        # Serialise intent — exclude heavy fields for prompt brevity
        intent_dict = intent.model_dump(
            exclude={"participants", "notes", "missing_fields"}
        )
        intent_dict["missing_fields"] = intent.missing_fields  # keep for context

        history = "\n".join(
            f"  [{i+1}] {u}" for i, u in enumerate(utterances[-6:])
        ) or "  (none yet)"

        conflicts_str = (
            json.dumps(slot_conflicts, indent=2)
            if slot_conflicts
            else "none"
        )

        system = _SYSTEM_PROMPT.format(rules=SCHEDULING_RULES.strip())
        user   = _USER_PROMPT.format(
            intent_json    = json.dumps(intent_dict, indent=2),
            slot_available = str(slot_available),
            slot_conflicts = conflicts_str,
            history        = history,
        )

        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def _generate(self, prompt: str) -> str:
        inputs = self._tokenizer(
            prompt,
            return_tensors    = "pt",
            truncation        = True,
            max_length        = 3072,
        ).to("cpu")

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens  = MINICPM_MAX_TOKENS,
                temperature     = 0.2,
                do_sample       = True,
                top_p           = 0.9,
                repetition_penalty = 1.1,
                pad_token_id    = self._tokenizer.eos_token_id,
            )

        # Decode only the new tokens (skip the prompt)
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    def _parse_response(
        self, raw: str, intent: SchedulingIntent
    ) -> EvaluationResult:
        raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()

        # Extract JSON block if model added preamble despite instructions
        if "{" in raw:
            raw = raw[raw.index("{") : raw.rindex("}") + 1]

        try:
            data = json.loads(raw)
            decision = data.get("decision", "ask_followup")

            # Sanity-check decision against intent state
            if decision == "schedule" and intent.missing_fields:
                logger.warning(
                    "Model said 'schedule' but fields are missing "
                    f"({intent.missing_fields}) — overriding to ask_followup"
                )
                decision = "ask_followup"

            return EvaluationResult(
                decision        = decision,
                spoken_response = data.get("spoken_response", self._default_response(intent)),
                reasoning       = data.get("reasoning", ""),
                suggested_date  = data.get("suggested_date"),
                suggested_time  = data.get("suggested_time"),
            )

        except Exception as exc:
            logger.warning(f"Evaluator JSON parse failed: {exc} — raw: {raw[:200]}")
            return self._fallback_result(intent)

    @staticmethod
    def _default_response(intent: SchedulingIntent) -> str:
        if intent.missing_fields:
            field_map = {
                "caller_name":    "Could I get your name please?",
                "preferred_date": "What date would you like to schedule this for?",
                "preferred_time": "And what time works best for you?",
            }
            return field_map.get(
                intent.missing_fields[0],
                "Could you give me a bit more detail?"
            )
        return "Let me check availability for you."

    @staticmethod
    def _fallback_result(intent: SchedulingIntent) -> EvaluationResult:
        return EvaluationResult(
            decision        = "ask_followup",
            spoken_response = "I'm sorry, could you repeat that?",
            reasoning       = "Evaluator fallback — parse error.",
            suggested_date  = None,
            suggested_time  = None,
        )


# ── Module singleton ──────────────────────────────────────────────────────────

_evaluator: Optional[Evaluator] = None


def get_evaluator() -> Evaluator:
    global _evaluator
    if _evaluator is None:
        _evaluator = Evaluator()
    return _evaluator


# ── Offline smoke test ────────────────────────────────────────────────────────

def _smoke_test_offline():
    logging.basicConfig(level=logging.INFO)
    logger.info("Running Evaluator offline smoke test…")

    from pipeline.intent_parser import SchedulingIntent

    # 1. end_call fast-path
    intent_end = SchedulingIntent(intent="end_call", confidence=0.9)
    ev = Evaluator()
    result = ev._fast_path(intent_end)
    assert result is not None
    assert result.decision == "end_call"
    logger.info("  ✓ end_call fast-path")

    # 2. unclear fast-path
    intent_unclear = SchedulingIntent(intent="unclear", confidence=0.05)
    result = ev._fast_path(intent_unclear)
    assert result is not None and result.decision == "ask_followup"
    logger.info("  ✓ unclear fast-path")

    # 3. Normal intent — no fast-path
    intent_book = SchedulingIntent(
        intent="book_meeting", caller_name="Priya",
        preferred_date="2026-06-10", preferred_time="14:00",
        confidence=0.9,
    ).compute_missing()
    result = ev._fast_path(intent_book)
    assert result is None   # should fall through to model
    logger.info("  ✓ valid booking intent passes through to model")

    # 4. Safety override — model says 'schedule' but fields missing
    intent_partial = SchedulingIntent(
        intent="book_meeting", caller_name="Raj", confidence=0.5
    ).compute_missing()
    raw_json = json.dumps({
        "decision": "schedule",   # model hallucinated this
        "spoken_response": "Booked!",
        "reasoning": "test",
        "suggested_date": None,
        "suggested_time": None,
    })
    result = ev._parse_response(raw_json, intent_partial)
    assert result.decision == "ask_followup", (
        f"Safety override failed — got {result.decision}"
    )
    logger.info("  ✓ safety override: 'schedule' with missing fields → ask_followup")

    # 5. Prompt build sanity
    prompt = ev._build_prompt(intent_book, ["Hi I want to book a call"], True, [])
    assert "<|im_start|>system" in prompt
    assert "SCHEDULING RULES" in prompt
    assert "slot_available" in prompt or "Slot available" in prompt
    logger.info("  ✓ prompt structure correct")

    # 6. Singleton
    e1 = get_evaluator()
    e2 = get_evaluator()
    assert e1 is e2
    logger.info("  ✓ module singleton")

    logger.info("\nOffline smoke test PASSED ✓")


if __name__ == "__main__":
    _smoke_test_offline()
