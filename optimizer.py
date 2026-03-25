"""SQLite storage + GEPA-style prompt evolution + analytics utilities."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from prompts import PROMPT_STYLES


DB_PATH = Path("support_bot.db")


@dataclass
class InteractionRecord:
    query: str
    selected_style: str
    response: str
    total_score: float
    feedback: str
    implicit_dissatisfied: bool
    solved: bool
    confidence: float
    user_level: str
    emotion: str
    created_at: str


class DataStore:
    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = db_path
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    selected_style TEXT NOT NULL,
                    response TEXT NOT NULL,
                    total_score REAL NOT NULL,
                    feedback TEXT NOT NULL,
                    implicit_dissatisfied INTEGER NOT NULL,
                    solved INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    user_level TEXT NOT NULL,
                    emotion TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS prompt_bank (
                    style_name TEXT PRIMARY KEY,
                    prompt_text TEXT NOT NULL,
                    uses INTEGER NOT NULL DEFAULT 0,
                    avg_score REAL NOT NULL DEFAULT 0
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS prompt_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    style_name TEXT NOT NULL,
                    old_prompt TEXT NOT NULL,
                    new_prompt TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()
        self._seed_defaults()

    def _seed_defaults(self) -> None:
        with self._connect() as conn:
            for style_name, prompt_text in PROMPT_STYLES.items():
                conn.execute(
                    "INSERT INTO prompt_bank(style_name, prompt_text) VALUES(?, ?) ON CONFLICT(style_name) DO NOTHING",
                    (style_name, prompt_text),
                )
            conn.commit()

    def get_prompt_bank(self) -> Dict[str, str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT style_name, prompt_text FROM prompt_bank").fetchall()
        return {name: text for name, text in rows}

    def update_prompt_metrics(self, style_name: str, new_score: float) -> None:
        with self._connect() as conn:
            uses, avg = conn.execute(
                "SELECT uses, avg_score FROM prompt_bank WHERE style_name = ?",
                (style_name,),
            ).fetchone()
            uses += 1
            avg = ((avg * (uses - 1)) + new_score) / uses
            conn.execute(
                "UPDATE prompt_bank SET uses = ?, avg_score = ? WHERE style_name = ?",
                (uses, round(avg, 4), style_name),
            )
            conn.commit()

    def save_interaction(self, record: InteractionRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO interactions(
                    query, selected_style, response, total_score, feedback,
                    implicit_dissatisfied, solved, confidence, user_level, emotion, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.query,
                    record.selected_style,
                    record.response,
                    record.total_score,
                    record.feedback,
                    int(record.implicit_dissatisfied),
                    int(record.solved),
                    record.confidence,
                    record.user_level,
                    record.emotion,
                    record.created_at,
                ),
            )
            conn.commit()

    def fetch_recent_queries(self, limit: int = 20) -> List[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT query FROM interactions ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [r[0] for r in rows]

    def fetch_recent_context(self, limit: int = 3) -> List[Tuple[str, str]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT query, response FROM interactions ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return rows

    def fetch_last_interaction_time(self) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT created_at FROM interactions ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return row[0] if row else None

    def fetch_history(self, limit: int = 50) -> List[Tuple]:
        with self._connect() as conn:
            return conn.execute(
                """
                SELECT query, selected_style, total_score, feedback, solved,
                       implicit_dissatisfied, confidence, user_level, emotion, created_at
                FROM interactions ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()

    def fetch_prompt_rankings(self) -> List[Tuple]:
        with self._connect() as conn:
            return conn.execute(
                "SELECT style_name, uses, avg_score FROM prompt_bank ORDER BY avg_score DESC"
            ).fetchall()

    def save_prompt_version(self, style_name: str, old_prompt: str, new_prompt: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO prompt_versions(style_name, old_prompt, new_prompt, created_at) VALUES(?, ?, ?, ?)",
                (style_name, old_prompt, new_prompt, timestamp_utc()),
            )
            conn.commit()

    def fetch_prompt_improvements(self, limit: int = 20) -> List[Tuple]:
        with self._connect() as conn:
            return conn.execute(
                "SELECT style_name, old_prompt, new_prompt, created_at FROM prompt_versions ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()

    def analytics(self) -> Dict[str, object]:
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
            sat = conn.execute(
                "SELECT COUNT(*) FROM interactions WHERE feedback IN ('like','positive_rating','solved_yes')"
            ).fetchone()[0]
            failed_rows = conn.execute(
                "SELECT query, COUNT(*) c FROM interactions WHERE solved = 0 GROUP BY query ORDER BY c DESC LIMIT 5"
            ).fetchall()
            trend = conn.execute(
                "SELECT substr(created_at,1,10) d, AVG(total_score) FROM interactions GROUP BY d ORDER BY d"
            ).fetchall()
            best_prompt = conn.execute(
                "SELECT style_name FROM prompt_bank ORDER BY avg_score DESC LIMIT 1"
            ).fetchone()
        return {
            "total_interactions": total,
            "satisfaction_rate": round(sat / total, 3) if total else None,
            "best_prompt": best_prompt[0] if best_prompt else None,
            "most_failed_queries": failed_rows,
            "improvement_trend": trend,
        }


class PromptOptimizer:
    def __init__(self, store: DataStore) -> None:
        self.store = store

    def register_score(self, style_name: str, score: float) -> None:
        self.store.update_prompt_metrics(style_name, score)

    def optimize(self) -> Dict[str, str]:
        with self.store._connect() as conn:
            rows = conn.execute("SELECT style_name, prompt_text, uses, avg_score FROM prompt_bank").fetchall()
            if any(uses == 0 for _, _, uses, _ in rows):
                return {n: p for n, p, _, _ in rows}

            ranked = sorted(rows, key=lambda r: r[3], reverse=True)
            top_style, top_prompt, _, top_score = ranked[0]
            weak_style, weak_prompt, _, weak_score = ranked[-1]

            if top_score - weak_score < 0.7:
                return {n: p for n, p, _, _ in rows}

            improved = self._mutate_prompt(weak_prompt, top_prompt, weak_style)
            conn.execute("UPDATE prompt_bank SET prompt_text = ? WHERE style_name = ?", (improved, weak_style))
            conn.commit()

        self.store.save_prompt_version(weak_style, weak_prompt, improved)
        return self.store.get_prompt_bank()

    @staticmethod
    def _mutate_prompt(weak_prompt: str, top_prompt: str, weak_style: str) -> str:
        example_block = (
            "Example: User says 'app keeps timing out'. Assistant should provide 3 checks, "
            "a quick fix, and escalation path."
        )
        constraints = "Constraints: keep under 180 words; include one verification step and one fallback path."
        return (
            f"[GEPA-Mutated for {weak_style}] {weak_prompt} "
            f"Borrow strengths from high performer: {top_prompt}. "
            f"{example_block} Change tone to be clearer and more action-oriented. {constraints}"
        )


def timestamp_utc() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")
