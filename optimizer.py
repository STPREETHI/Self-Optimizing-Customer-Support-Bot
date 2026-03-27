"""Persistence, retrieval, ranking, and feedback-driven prompt optimization."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


DB_PATH = Path("support_system.db")

DEFAULT_STYLE_WEIGHTS = {
    "professional": 1.0,
    "empathetic": 1.0,
    "technical": 1.0,
    "simple": 1.0,
}

KB_SEED = [
    "Login issue: clear stale sessions, sync system clock, reset password tokens.",
    "Refund issue: verify transaction ID, eligibility window, and payment gateway status.",
    "Performance issue: inspect latency, deployment changes, and cache hit rate.",
    "API errors: validate key scope, rate limits, and request schema.",
]


@dataclass
class InteractionRecord:
    user_id: str
    query: str
    intent: str
    entities: str
    chosen_style: str
    response: str
    score: float
    confidence: float
    resolved: Optional[bool]
    created_at: str


class DataStore:
    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    intent TEXT NOT NULL,
                    entities TEXT NOT NULL,
                    chosen_style TEXT NOT NULL,
                    response TEXT NOT NULL,
                    score REAL NOT NULL,
                    confidence REAL NOT NULL,
                    resolved INTEGER,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    interaction_id INTEGER NOT NULL,
                    liked INTEGER,
                    feedback_text TEXT,
                    resolved INTEGER,
                    reward REAL NOT NULL,
                    failure_reason TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS prompt_profiles (
                    style TEXT PRIMARY KEY,
                    weight REAL NOT NULL,
                    uses INTEGER NOT NULL DEFAULT 0,
                    avg_reward REAL NOT NULL DEFAULT 0,
                    prompt_hint TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS prompt_evolution (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    style TEXT NOT NULL,
                    before_hint TEXT NOT NULL,
                    after_hint TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS kb_vectors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_text TEXT NOT NULL,
                    embedding TEXT NOT NULL
                )
                """
            )
            conn.commit()

        self._seed_defaults()

    def _seed_defaults(self) -> None:
        with self._connect() as conn:
            for style, weight in DEFAULT_STYLE_WEIGHTS.items():
                conn.execute(
                    """
                    INSERT INTO prompt_profiles(style, weight, prompt_hint)
                    VALUES(?, ?, ?)
                    ON CONFLICT(style) DO NOTHING
                    """,
                    (style, weight, f"Base tone for {style} responses."),
                )

            existing = conn.execute("SELECT COUNT(*) FROM kb_vectors").fetchone()[0]
            if existing == 0:
                for text in KB_SEED:
                    emb = self._embed(text)
                    conn.execute(
                        "INSERT INTO kb_vectors(source_text, embedding) VALUES(?, ?)",
                        (text, self._serialize_vector(emb)),
                    )
            conn.commit()

    @staticmethod
    def _embed(text: str, dim: int = 128) -> np.ndarray:
        vec = np.zeros(dim, dtype=float)
        for token in text.lower().split():
            vec[hash(token) % dim] += 1
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    @staticmethod
    def _serialize_vector(vec: np.ndarray) -> str:
        return ",".join([f"{x:.6f}" for x in vec.tolist()])

    @staticmethod
    def _deserialize_vector(raw: str) -> np.ndarray:
        return np.array([float(x) for x in raw.split(",")], dtype=float)

    def add_kb_document(self, text: str) -> None:
        emb = self._embed(text)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO kb_vectors(source_text, embedding) VALUES(?, ?)",
                (text, self._serialize_vector(emb)),
            )
            conn.commit()

    def search_kb(self, query: str, top_k: int = 3) -> List[str]:
        q = self._embed(query)
        with self._connect() as conn:
            rows = conn.execute("SELECT source_text, embedding FROM kb_vectors").fetchall()

        scored = []
        for source_text, raw_emb in rows:
            emb = self._deserialize_vector(raw_emb)
            score = float(np.dot(q, emb))
            scored.append((score, source_text))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored[:top_k]]

    def save_interaction(self, record: InteractionRecord) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO interactions(
                    user_id, query, intent, entities, chosen_style, response, score,
                    confidence, resolved, created_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.user_id,
                    record.query,
                    record.intent,
                    record.entities,
                    record.chosen_style,
                    record.response,
                    record.score,
                    record.confidence,
                    None if record.resolved is None else int(record.resolved),
                    record.created_at,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    def save_feedback(
        self,
        interaction_id: int,
        liked: Optional[bool],
        feedback_text: str,
        resolved: Optional[bool],
        reward: float,
        failure_reason: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO feedback(interaction_id, liked, feedback_text, resolved, reward, failure_reason, created_at)
                VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    interaction_id,
                    None if liked is None else int(liked),
                    feedback_text,
                    None if resolved is None else int(resolved),
                    reward,
                    failure_reason,
                    timestamp_utc(),
                ),
            )
            conn.commit()

    def get_recent_memory(self, user_id: str, limit: int = 4) -> List[Tuple[str, str]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT query, response FROM interactions WHERE user_id=? ORDER BY id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return rows

    def get_style_boost(self) -> Dict[str, float]:
        with self._connect() as conn:
            rows = conn.execute("SELECT style, weight FROM prompt_profiles").fetchall()
        return {style: weight for style, weight in rows}

    def update_style_reward(self, style: str, reward: float) -> None:
        with self._connect() as conn:
            uses, avg_reward, weight, hint = conn.execute(
                "SELECT uses, avg_reward, weight, prompt_hint FROM prompt_profiles WHERE style=?",
                (style,),
            ).fetchone()
            new_uses = uses + 1
            new_avg = ((avg_reward * uses) + reward) / new_uses
            new_weight = min(2.5, max(0.5, weight + (0.08 * reward)))

            conn.execute(
                "UPDATE prompt_profiles SET uses=?, avg_reward=?, weight=? WHERE style=?",
                (new_uses, round(new_avg, 4), round(new_weight, 4), style),
            )
            conn.commit()

    def evolve_weak_prompts(self) -> None:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT style, weight, avg_reward, prompt_hint FROM prompt_profiles"
            ).fetchall()
            if len(rows) < 2:
                return

            ranked = sorted(rows, key=lambda r: r[2], reverse=True)
            best_style, _, best_avg, best_hint = ranked[0]
            weak_style, _, weak_avg, weak_hint = ranked[-1]

            if best_avg - weak_avg < 0.2:
                return

            improved_hint = (
                f"{weak_hint} Add concrete examples, shorten instructions, and include "
                f"a resolution-check question. Borrow successful pattern from {best_style}: {best_hint}"
            )

            conn.execute(
                "UPDATE prompt_profiles SET prompt_hint=? WHERE style=?",
                (improved_hint, weak_style),
            )
            conn.execute(
                "INSERT INTO prompt_evolution(style, before_hint, after_hint, reason, created_at) VALUES(?, ?, ?, ?, ?)",
                (
                    weak_style,
                    weak_hint,
                    improved_hint,
                    "low reward vs top style",
                    timestamp_utc(),
                ),
            )
            conn.commit()

    def analytics(self) -> Dict[str, object]:
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
            avg_score = conn.execute("SELECT AVG(score) FROM interactions").fetchone()[0]
            resolution_rate = conn.execute(
                "SELECT AVG(CASE WHEN resolved=1 THEN 1.0 ELSE 0.0 END) FROM interactions WHERE resolved IS NOT NULL"
            ).fetchone()[0]
            satisfaction = conn.execute("SELECT AVG(reward) FROM feedback").fetchone()[0]
            common_issues = conn.execute(
                "SELECT intent, COUNT(*) c FROM interactions GROUP BY intent ORDER BY c DESC LIMIT 5"
            ).fetchall()
            trend = conn.execute(
                "SELECT substr(created_at,1,10) d, AVG(score) FROM interactions GROUP BY d ORDER BY d"
            ).fetchall()
            best_prompt = conn.execute(
                "SELECT style FROM prompt_profiles ORDER BY avg_reward DESC LIMIT 1"
            ).fetchone()

        return {
            "total_interactions": total,
            "response_accuracy": round(avg_score, 3) if avg_score is not None else None,
            "resolution_rate": round(resolution_rate, 3) if resolution_rate is not None else None,
            "user_satisfaction": round(satisfaction, 3) if satisfaction is not None else None,
            "most_common_issues": common_issues,
            "trend": trend,
            "best_prompt": best_prompt[0] if best_prompt else None,
        }

    def recent_prompt_evolution(self, limit: int = 10) -> List[Tuple]:
        with self._connect() as conn:
            return conn.execute(
                "SELECT style, before_hint, after_hint, reason, created_at FROM prompt_evolution ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()


def timestamp_utc() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")
