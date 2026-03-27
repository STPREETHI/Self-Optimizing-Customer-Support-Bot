"""FastAPI backend for feedback-driven self-optimizing customer support system."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Dict, Optional

import dspy
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from feedback_engine import build_feedback_event
from optimizer import DataStore, InteractionRecord, timestamp_utc
from query_processor import process_query
from response_generator import generate_multi_candidates


def configure_llm() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return
    lm = dspy.LM("openai/gpt-4o-mini", api_key=api_key, temperature=0.2)
    dspy.settings.configure(lm=lm)


@dataclass
class ResponseScore:
    clarity: float
    correctness: float
    helpfulness: float
    total: float
    confidence: float


def score_candidate(query: str, response: str, style_weight: float) -> ResponseScore:
    q_tokens = {w.lower().strip(".,?!") for w in query.split() if len(w) > 3}
    r_tokens = {w.lower().strip(".,?!") for w in response.split()}
    overlap = len(q_tokens & r_tokens)

    clarity = 6.0 + (1.0 if len(response.split()) <= 180 else -0.5)
    correctness = 5.7 + min(2.5, overlap * 0.4)
    helpfulness = 6.1 + (
        1.4 if any(x in response.lower() for x in ["step", "check", "verify", "if unresolved"]) else 0.0
    )

    clarity = min(10.0, max(0.0, clarity))
    correctness = min(10.0, max(0.0, correctness))
    helpfulness = min(10.0, max(0.0, helpfulness))

    weighted = (clarity * 0.3 + correctness * 0.3 + helpfulness * 0.4) * style_weight
    total = min(10.0, round(weighted, 3))
    confidence = round(min(0.98, max(0.2, total / 10)), 3)

    return ResponseScore(
        clarity=round(clarity, 3),
        correctness=round(correctness, 3),
        helpfulness=round(helpfulness, 3),
        total=total,
        confidence=confidence,
    )


class ChatRequest(BaseModel):
    user_id: str = Field(default="demo-user")
    query: str


class ChatResponse(BaseModel):
    interaction_id: int
    query: str
    intent: str
    entities: Dict[str, str]
    chosen_style: str
    response: str
    score: Dict[str, float]
    retrieved_knowledge: list[str]


class FeedbackRequest(BaseModel):
    interaction_id: int
    liked: Optional[bool] = None
    feedback_text: str = ""
    resolved: Optional[bool] = None


class KBDocRequest(BaseModel):
    text: str


configure_llm()
store = DataStore()
app = FastAPI(title="Self-Optimizing Customer Support System", version="2.1.0")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    qf = process_query(payload.query)
    memory = store.get_recent_memory(user_id=payload.user_id)
    rag_snippets = store.search_kb(payload.query, top_k=3)
    style_boost = store.get_style_boost()

    candidates = generate_multi_candidates(
        query=payload.query,
        intent=qf.intent,
        entities=qf.entities,
        history=memory,
        retrieved_snippets=rag_snippets,
        style_boost=style_boost,
    )

    ranked = []
    for cand in candidates:
        score = score_candidate(payload.query, cand.response, style_boost.get(cand.style, 1.0))
        ranked.append((score.total, cand, score))
    ranked.sort(key=lambda x: x[0], reverse=True)
    _, best_cand, best_score = ranked[0]

    interaction_id = store.save_interaction(
        InteractionRecord(
            user_id=payload.user_id,
            query=payload.query,
            intent=qf.intent,
            entities=str(qf.entities),
            chosen_style=best_cand.style,
            response=best_cand.response,
            score=best_score.total,
            confidence=best_score.confidence,
            resolved=None,
            created_at=timestamp_utc(),
        )
    )

    return ChatResponse(
        interaction_id=interaction_id,
        query=payload.query,
        intent=qf.intent,
        entities=qf.entities,
        chosen_style=best_cand.style,
        response=best_cand.response,
        score=asdict(best_score),
        retrieved_knowledge=rag_snippets,
    )


@app.post("/feedback")
def feedback(payload: FeedbackRequest) -> Dict[str, object]:
    event = build_feedback_event(
        interaction_id=payload.interaction_id,
        liked=payload.liked,
        feedback_text=payload.feedback_text,
        resolved=payload.resolved,
    )

    store.save_feedback(
        interaction_id=event.interaction_id,
        liked=event.liked,
        feedback_text=event.feedback_text,
        resolved=event.resolved,
        reward=event.reward,
        failure_reason=event.failure_reason,
    )

    with store._connect() as conn:
        row = conn.execute(
            "SELECT chosen_style FROM interactions WHERE id = ?",
            (payload.interaction_id,),
        ).fetchone()
    if row:
        store.update_style_reward(style=row[0], reward=event.reward)
        store.evolve_weak_prompts()

    return {
        "interaction_id": payload.interaction_id,
        "reward": event.reward,
        "failure_reason": event.failure_reason,
        "status": "updated",
    }


@app.get("/analytics")
def analytics() -> Dict[str, object]:
    return store.analytics()


@app.get("/prompt-evolution")
def prompt_evolution() -> Dict[str, object]:
    rows = store.recent_prompt_evolution(limit=10)
    return {"evolution": rows}


@app.get("/style-weights")
def style_weights() -> Dict[str, object]:
    return {"style_weights": store.get_style_boost()}


@app.post("/kb/add")
def kb_add(payload: KBDocRequest) -> Dict[str, str]:
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="KB text cannot be empty")
    store.add_kb_document(text)
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
