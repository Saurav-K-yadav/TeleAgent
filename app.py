"""
app.py — Telecalling Agent — Gradio 6 UI

Layout
──────
┌─────────────────────────────────────────────────────┐
│  📞 AI Telecalling Agent          [status badge]    │
├──────────────────────┬──────────────────────────────┤
│  🎤 LIVE CALL        │  📋 EXTRACTED DATA           │
│  ┌────────────────┐  │  [intent markdown table]     │
│  │ Audio stream   │  │                              │
│  └────────────────┘  ├──────────────────────────────┤
│  [Start] [End]       │  🤖 AGENT RESPONSE           │
│  ┌────────────────┐  │  [spoken response box]       │
│  │ Transcript     │  │                              │
│  └────────────────┘  │  ✅ BOOKING CONFIRMED        │
│                      │  [booking details box]       │
├──────────────────────┴──────────────────────────────┤
│  📁 CALL LOG                                        │
│  [dataframe — recent calls]                         │
└─────────────────────────────────────────────────────┘
"""

import logging
import os
import json

import gradio as gr
import numpy as np
from pipeline.transcriber import get_transcriber
from pipeline.intent_parser import get_intent_parser
from pipeline.evaluater import get_evaluator

from config import APP_TITLE, APP_DESCRIPTION, SERVER_PORT, SERVER_NAME
from pipeline.orchestrator import CallSession, PipelineUpdate
from db import init_db

# Load HuggingFace config and set token early
try:
    with open("hf_config.json", "r") as f:
        hf_cfg = json.load(f)
        hf_token = hf_cfg.get("huggingface", {}).get("hub", {}).get("token", "")
        if hf_token and hf_token != "${HF_TOKEN}":
            os.environ["HF_TOKEN"] = hf_token
except (FileNotFoundError, json.JSONDecodeError) as e:
    pass  # hf_config.json not found or invalid, use env var if set

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress verbose logs from HuggingFace hub
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("transformers.modeling_utils").setLevel(logging.WARNING)

# Initialize database on startup
init_db()


# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
/* Global */
.gradio-container { font-family: 'Inter', sans-serif; max-width: 1200px; }

/* Status badge */
#status-badge textarea {
    font-size: 0.9rem;
    font-weight: 600;
    text-align: center;
    border-radius: 20px;
    padding: 4px 12px;
    background: #f0fdf4;
    border: 1px solid #86efac;
    color: #166534;
}

/* Agent response */
#agent-box textarea {
    font-size: 1.05rem;
    font-style: italic;
    background: #eff6ff;
    border: 1px solid #93c5fd;
    border-radius: 8px;
    color: #1e3a5f;
    min-height: 80px;
}

/* Booking confirmed */
#booking-box textarea {
    background: #f0fdf4;
    border: 1px solid #4ade80;
    border-radius: 8px;
    color: #14532d;
    font-weight: 500;
}

/* Transcript */
#transcript-box textarea {
    font-family: monospace;
    font-size: 0.85rem;
    background: #1e1e2e;
    color: #cdd6f4;
    border-radius: 8px;
    min-height: 180px;
}

/* VAD indicator dot */
#vad-dot {
    text-align: center;
    font-size: 1.2rem;
}

/* Call buttons */
.call-btn-start { background: #16a34a !important; color: white !important; }
.call-btn-end   { background: #dc2626 !important; color: white !important; }
.call-btn-reset { background: #6b7280 !important; color: white !important; }

/* Intent table inside Markdown */
#intent-panel table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
#intent-panel th, #intent-panel td {
    padding: 5px 10px;
    border: 1px solid #e2e8f0;
    text-align: left;
}
#intent-panel tr:nth-child(even) { background: #f8fafc; }

/* Call log */
#call-log { font-size: 0.82rem; }
"""


# ── UI helpers ────────────────────────────────────────────────────────────────

def _format_transcript(lines: list[str]) -> str:
    if not lines:
        return "(waiting for speech…)"
    return "\n".join(f"[{i+1}] {l}" for i, l in enumerate(lines))


def _format_booking(info: dict | None) -> str:
    if not info:
        return ""
    return (
        f"✅  Booking #{info['booking_id']} confirmed!\n"
        f"    📅  {info['date']}  🕐  {info['time']}  "
        f"({info['duration']} min)\n"
        f"    👤  {info['caller']}   📞  {info['type'].replace('_', ' ').title()}"
    )


def _call_log_rows(records: list[dict]) -> list[list]:
    rows = []
    for r in records:
        ts = r.get("timestamp", "")[:16].replace("T", " ")
        rows.append([
            r.get("id", ""),
            ts,
            r.get("caller_name") or "—",
            r.get("intent")      or "—",
            r.get("decision")    or "—",
            r.get("status")      or "—",
        ])
    return rows


# ── Gradio App ────────────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:

    with gr.Blocks(css=CSS, title=APP_TITLE, theme=gr.themes.Soft()) as demo:

        # ── Per-session state ──────────────────────────────────────────────
        # gr.State holds one CallSession object per browser tab.
        session_state = gr.State(value=None)

        # ── Header ─────────────────────────────────────────────────────────
        gr.Markdown(f"# {APP_TITLE}\n_{APP_DESCRIPTION}_")

        status_badge = gr.Textbox(
            value       = "🟢 Ready — press Start Call",
            label       = "",
            interactive = False,
            elem_id     = "status-badge",
        )

        # ── Main row ───────────────────────────────────────────────────────
        with gr.Row():

            # ── Left column: call controls + transcript ────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### 🎤 Live Call")

                audio_input = gr.Audio(
                    sources    = ["microphone"],
                    streaming  = True,
                    type       = "numpy",
                    label      = "Microphone (auto-detecting speech)",
                    interactive= True,
                    elem_id    = "audio-input",
                )

                vad_dot = gr.Markdown("⚫ _mic idle_", elem_id="vad-dot")

                with gr.Row():
                    btn_start = gr.Button(
                        "📞 Start Call", variant="primary",
                        elem_classes=["call-btn-start"],
                    )
                    btn_end = gr.Button(
                        "📵 End Call", variant="stop",
                        elem_classes=["call-btn-end"],
                    )
                    btn_reset = gr.Button(
                        "🔄 Reset", variant="secondary",
                        elem_classes=["call-btn-reset"],
                    )

                transcript_box = gr.Textbox(
                    label       = "📝 Live Transcript",
                    value       = "(waiting for speech…)",
                    lines       = 8,
                    max_lines   = 20,
                    interactive = False,
                    elem_id     = "transcript-box",
                )

            # ── Right column: intent + agent response + booking ────────────
            with gr.Column(scale=1):
                gr.Markdown("### 📋 Extracted Data")

                intent_panel = gr.Markdown(
                    "_No data yet — waiting for first utterance…_",
                    elem_id = "intent-panel",
                )

                gr.Markdown("### 🤖 Agent Response")

                agent_box = gr.Textbox(
                    value       = "",
                    label       = "",
                    lines       = 3,
                    interactive = False,
                    elem_id     = "agent-box",
                    placeholder = "Agent will respond here…",
                )

                booking_box = gr.Textbox(
                    value       = "",
                    label       = "📅 Booking Status",
                    lines       = 3,
                    interactive = False,
                    elem_id     = "booking-box",
                    visible     = False,
                )

        # ── Call log ───────────────────────────────────────────────────────
        gr.Markdown("### 📁 Call Log")

        call_log_table = gr.Dataframe(
            headers     = ["ID", "Timestamp", "Caller", "Intent", "Decision", "Status"],
            datatype    = ["number", "str", "str", "str", "str", "str"],
            value       = [],
            interactive = False,
            elem_id     = "call-log",
            row_count   = (5, "dynamic"),
        )

        # ── Helper: unpack PipelineUpdate → tuple of component values ─────
        def _unpack(u: PipelineUpdate):
            """Return values in the exact order of outputs lists below."""
            vad_label = "🔴 _Speaking…_" if u.vad_speaking else "⚫ _mic idle_"
            booking_text    = _format_booking(u.booking_confirmed)
            booking_visible = bool(booking_text)
            return (
                u.status,                           # status_badge
                vad_label,                          # vad_dot
                _format_transcript(u.transcript_lines),  # transcript_box
                u.intent_md,                        # intent_panel
                u.agent_response,                   # agent_box
                booking_text,                       # booking_box value
                gr.update(visible=booking_visible), # booking_box visible
                _call_log_rows(u.call_log),         # call_log_table
            )

        # ── All output components in one list (matches _unpack order) ─────
        ALL_OUTPUTS = [
            status_badge,
            vad_dot,
            transcript_box,
            intent_panel,
            agent_box,
            booking_box,
            booking_box,       # second entry → gr.update(visible=…)
            call_log_table,
        ]

        # ── Session factory ────────────────────────────────────────────────
        def _get_or_create_session(state):
            if state is None:
                state = CallSession()
            return state

        # ── Button callbacks ───────────────────────────────────────────────

        def on_start(state):
            state = _get_or_create_session(state)
            update = state.start_call()
            return (state, *_unpack(update))

        def on_end(state):
            state = _get_or_create_session(state)
            update = state.end_call()
            return (state, *_unpack(update))

        def on_reset(state):
            state = _get_or_create_session(state)
            update = state.reset()
            return (state, *_unpack(update))

        BTN_OUTPUTS = [session_state] + ALL_OUTPUTS

        btn_start.click(on_start, inputs=[session_state], outputs=BTN_OUTPUTS)
        btn_end.click  (on_end,   inputs=[session_state], outputs=BTN_OUTPUTS)
        btn_reset.click(on_reset, inputs=[session_state], outputs=BTN_OUTPUTS)

        # ── Audio streaming callback ───────────────────────────────────────
        # Fires every `stream_every` seconds with (sample_rate, np.ndarray).
        # We pass the current session state in and get it back (updated).

        def on_audio_stream(audio_chunk, state):
            """
            Called by Gradio every 0.5 s while the mic is active.
            audio_chunk: (sample_rate: int, data: np.ndarray) | None
            """
            state = _get_or_create_session(state)

            if not state.call_active:
                # Return current state without processing
                u = state._build_update()
                return (state, *_unpack(u))

            if audio_chunk is None:
                u = state._build_update()
                return (state, *_unpack(u))

            sample_rate, audio_np = audio_chunk

            # Ensure float32 mono
            audio_np = np.array(audio_np, dtype=np.float32)
            if audio_np.ndim == 2:
                audio_np = audio_np.mean(axis=1)

            update = state.process_audio_chunk(sample_rate, audio_np)
            return (state, *_unpack(update))

        audio_input.stream(
            fn           = on_audio_stream,
            inputs       = [audio_input, session_state],
            outputs      = [session_state] + ALL_OUTPUTS,
            stream_every = 0.5,      # seconds — half-second chunks
            time_limit   = 3600,     # allow up to 1-hour calls
        )

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Starting Gradio app; prefetching deployed ASR model if needed...")
    try:
        get_transcriber().prefetch()
    except Exception as exc:
        logger.error(
            "ASR prefetch failed at startup; continuing with lazy loading: %s",
            exc,
        )

    app = build_app()
    app.launch(
        server_name = SERVER_NAME,
        server_port = SERVER_PORT,
        show_error  = True,
    )