"""
Database layer — SQLite via Python's built-in sqlite3.
Handles schema creation and all CRUD operations for the telecalling agent.
"""

import sqlite3
import json
import logging
from datetime import datetime
from contextlib import contextmanager
from typing import Optional

from config import DB_PATH

logger = logging.getLogger(__name__)


# ── Schema ────────────────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,           -- ISO-8601 UTC
    session_id      TEXT    NOT NULL,           -- UUID per Gradio session
    raw_transcript  TEXT    NOT NULL DEFAULT '',
    intent          TEXT,                       -- book_meeting | reschedule | cancel | ...
    caller_name     TEXT,
    preferred_date  TEXT,                       -- YYYY-MM-DD
    preferred_time  TEXT,                       -- HH:MM (24h)
    duration_mins   INTEGER,
    participants    TEXT    DEFAULT '[]',        -- JSON array of strings
    meeting_type    TEXT,                       -- phone | video | in_person
    notes           TEXT,
    confidence      REAL,                       -- 0.0–1.0 from Qwen
    decision        TEXT,                       -- schedule | ask_followup | reject
    reasoning       TEXT,                       -- MiniCPM's explanation
    status          TEXT    NOT NULL DEFAULT 'open'  -- open | confirmed | cancelled
);

CREATE TABLE IF NOT EXISTS bookings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    call_id         INTEGER NOT NULL REFERENCES calls(id),
    booked_date     TEXT    NOT NULL,           -- YYYY-MM-DD
    booked_time     TEXT    NOT NULL,           -- HH:MM
    duration_mins   INTEGER NOT NULL,
    caller_name     TEXT    NOT NULL,
    meeting_type    TEXT,
    created_at      TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_calls_session   ON calls(session_id);
CREATE INDEX IF NOT EXISTS idx_bookings_date   ON bookings(booked_date);
"""


# ── Connection helper ─────────────────────────────────────────────────────────

@contextmanager
def get_conn():
    """Context manager — always commits or rolls back cleanly."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # safe for concurrent Gradio threads
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Create tables if they don't exist. Safe to call on every startup."""
    with get_conn() as conn:
        conn.executescript(SCHEMA)
    logger.info(f"Database ready at {DB_PATH}")


# ── Call record helpers ───────────────────────────────────────────────────────

def create_call(session_id: str) -> int:
    """
    Insert a bare call record at the start of a session.
    Returns the new call id so downstream steps can update it.
    """
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO calls (timestamp, session_id, raw_transcript) VALUES (?, ?, '')",
            (datetime.utcnow().isoformat(), session_id)
        )
        return cur.lastrowid


def append_transcript(call_id: int, new_text: str):
    """Append a transcribed utterance to the running transcript."""
    with get_conn() as conn:
        conn.execute(
            """UPDATE calls
               SET raw_transcript = raw_transcript || '\n' || ?
               WHERE id = ?""",
            (new_text.strip(), call_id)
        )


def update_call_intent(call_id: int, parsed: dict):
    """
    Write Qwen's structured JSON output into the call record.
    `parsed` is expected to match the scheduling JSON schema from config.
    """
    with get_conn() as conn:
        conn.execute(
            """UPDATE calls SET
                intent         = :intent,
                caller_name    = :caller_name,
                preferred_date = :preferred_date,
                preferred_time = :preferred_time,
                duration_mins  = :duration_minutes,
                participants   = :participants,
                meeting_type   = :meeting_type,
                notes          = :notes,
                confidence     = :confidence
               WHERE id = :id""",
            {
                "intent":           parsed.get("intent"),
                "caller_name":      parsed.get("caller_name"),
                "preferred_date":   parsed.get("preferred_date"),
                "preferred_time":   parsed.get("preferred_time"),
                "duration_minutes": parsed.get("duration_minutes"),
                "participants":     json.dumps(parsed.get("participants", [])),
                "meeting_type":     parsed.get("meeting_type"),
                "notes":            parsed.get("notes"),
                "confidence":       parsed.get("confidence"),
                "id":               call_id,
            }
        )


def update_call_decision(call_id: int, decision: str, reasoning: str):
    """Write MiniCPM's evaluation result back to the call record."""
    with get_conn() as conn:
        conn.execute(
            "UPDATE calls SET decision = ?, reasoning = ? WHERE id = ?",
            (decision, reasoning, call_id)
        )


def confirm_booking(call_id: int, parsed: dict) -> int:
    """
    Insert a confirmed booking row and mark the call as confirmed.
    Returns the booking id.
    """
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO bookings
               (call_id, booked_date, booked_time, duration_mins,
                caller_name, meeting_type, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                call_id,
                parsed["preferred_date"],
                parsed["preferred_time"],
                parsed.get("duration_minutes", 30),
                parsed.get("caller_name", "Unknown"),
                parsed.get("meeting_type", "phone"),
                datetime.utcnow().isoformat(),
            )
        )
        conn.execute(
            "UPDATE calls SET status = 'confirmed' WHERE id = ?", (call_id,)
        )
        return cur.lastrowid


# ── Availability check ────────────────────────────────────────────────────────

def is_slot_available(date: str, time: str, duration_mins: int = 30) -> bool:
    """
    Returns True if no existing booking overlaps with the requested slot
    (including the mandatory 15-minute buffer).
    date: YYYY-MM-DD, time: HH:MM
    """
    from datetime import datetime, timedelta

    try:
        start = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")
        end   = start + timedelta(minutes=duration_mins + 15)   # +15 min buffer
    except ValueError:
        return False

    with get_conn() as conn:
        rows = conn.execute(
            "SELECT booked_time, duration_mins FROM bookings WHERE booked_date = ?",
            (date,)
        ).fetchall()

    for row in rows:
        existing_start = datetime.strptime(f"{date} {row['booked_time']}", "%Y-%m-%d %H:%M")
        existing_end   = existing_start + timedelta(minutes=row["duration_mins"] + 15)
        # overlap check
        if start < existing_end and end > existing_start:
            return False

    return True


def get_booked_slots(date: str) -> list[dict]:
    """Return all bookings for a given date for display in the UI."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT booked_time, duration_mins, caller_name, meeting_type
               FROM bookings WHERE booked_date = ? ORDER BY booked_time""",
            (date,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_recent_calls(limit: int = 20) -> list[dict]:
    """Fetch the most recent call records for the call log panel."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT id, timestamp, caller_name, intent, decision, status
               FROM calls ORDER BY id DESC LIMIT ?""",
            (limit,)
        ).fetchall()
    return [dict(r) for r in rows]