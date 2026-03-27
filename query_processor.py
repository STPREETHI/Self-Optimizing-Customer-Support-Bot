"""Query understanding module: intent detection, entity extraction, and embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


INTENT_KEYWORDS: Dict[str, List[str]] = {
    "refund": ["refund", "money back", "charged", "billing"],
    "complaint": ["bad", "terrible", "angry", "frustrated", "complaint"],
    "technical_issue": ["error", "bug", "crash", "timeout", "not working", "failed"],
    "query": ["how", "what", "why", "where", "help"],
    "account": ["login", "password", "account", "locked"],
}

URGENCY_MARKERS = ["urgent", "asap", "immediately", "today", "right now"]


@dataclass
class QueryFeatures:
    intent: str
    entities: Dict[str, str]
    embedding: List[float]


def _simple_embed(text: str, dim: int = 128) -> np.ndarray:
    """Lightweight deterministic embedding (replaceable with OpenAI/ST in prod)."""

    vec = np.zeros(dim, dtype=float)
    for token in text.lower().split():
        idx = hash(token) % dim
        vec[idx] += 1.0

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def classify_intent(text: str) -> str:
    lowered = text.lower()
    best_intent = "query"
    best_score = 0

    for intent, keywords in INTENT_KEYWORDS.items():
        score = sum(keyword in lowered for keyword in keywords)
        if score > best_score:
            best_intent = intent
            best_score = score

    return best_intent


def extract_entities(text: str) -> Dict[str, str]:
    lowered = text.lower()
    urgency = "high" if any(marker in lowered for marker in URGENCY_MARKERS) else "normal"

    product = "platform"
    for candidate in ["mobile app", "web app", "api", "dashboard", "checkout"]:
        if candidate in lowered:
            product = candidate
            break

    issue_type = "technical" if any(k in lowered for k in ["error", "crash", "timeout", "failed"]) else "general"

    return {"product": product, "issue_type": issue_type, "urgency": urgency}


def process_query(text: str) -> QueryFeatures:
    intent = classify_intent(text)
    entities = extract_entities(text)
    embedding = _simple_embed(text).tolist()
    return QueryFeatures(intent=intent, entities=entities, embedding=embedding)
