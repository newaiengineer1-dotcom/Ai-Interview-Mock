import sqlite3
import json
from datetime import datetime
from pathlib import Path

DB_PATH = Path("interview_history.db")

def _conn():
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c

def init_db():
    with _conn() as c:
        c.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            role TEXT,
            interview_type TEXT,
            personality TEXT,
            difficulty TEXT,
            avg_score REAL DEFAULT 0,
            summary TEXT,
            metrics TEXT
        )""")
        c.execute("""
        CREATE TABLE IF NOT EXISTS turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            idx INTEGER,
            question TEXT,
            answer TEXT,
            evaluation TEXT,
            scores TEXT,
            created_at TEXT
        )""")

def create_session(role, itype, personality, difficulty):
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO sessions(created_at, role, interview_type, personality, difficulty, avg_score) "
            "VALUES(?,?,?,?,?,?)",
            (datetime.utcnow().isoformat(), role, itype, personality, difficulty, 0.0),
        )
        return cur.lastrowid

def add_turn(session_id, idx, question, answer, evaluation, scores):
    with _conn() as c:
        c.execute(
            "INSERT INTO turns(session_id, idx, question, answer, evaluation, scores, created_at) "
            "VALUES(?,?,?,?,?,?,?)",
            (session_id, idx, question, answer, json.dumps(evaluation),
             json.dumps(scores), datetime.utcnow().isoformat()),
        )

def update_session(session_id, avg_score, summary=None, metrics=None):
    with _conn() as c:
        c.execute(
            "UPDATE sessions SET avg_score=?, summary=?, metrics=? WHERE id=?",
            (avg_score, summary, json.dumps(metrics or {}), session_id),
        )

def get_sessions():
    with _conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM sessions ORDER BY id DESC").fetchall()]

def get_turns(session_id):
    with _conn() as c:
        rows = [dict(r) for r in c.execute(
            "SELECT * FROM turns WHERE session_id=? ORDER BY idx", (session_id,)
        ).fetchall()]
    for r in rows:
        try:
            r["scores"] = json.loads(r["scores"] or "{}")
        except Exception:
            r["scores"] = {}
        try:
            r["evaluation"] = json.loads(r["evaluation"] or "{}")
        except Exception:
            r["evaluation"] = {}
    return rows
