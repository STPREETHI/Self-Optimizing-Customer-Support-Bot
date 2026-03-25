"""Main orchestration for Self-Optimizing Customer Support Bot (Tech Support domain)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

import dspy

from domain_kb import retrieve_relevant_faq
from evaluator import EvaluationResult, choose_best, evaluate_response
from feedback import (
    detect_implicit_feedback,
    detect_user_emotion,
    detect_user_level,
    parse_explicit_feedback,
)
from optimizer import DataStore, InteractionRecord, PromptOptimizer, timestamp_utc
from prompts import generate_candidates


LOW_SCORE_THRESHOLD = 6.8


def configure_dspy_model() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return
    lm = dspy.LM("openai/gpt-4o-mini", api_key=api_key, temperature=0.2)
    dspy.settings.configure(lm=lm)


def _seconds_between(now_iso: str, previous_iso: Optional[str]) -> Optional[float]:
    if not previous_iso:
        return None
    now = datetime.fromisoformat(now_iso)
    prev = datetime.fromisoformat(previous_iso)
    return (now - prev).total_seconds()


def _format_context(history: list[tuple[str, str]]) -> str:
    if not history:
        return "No prior context."
    chunks = []
    for q, r in history:
        chunks.append(f"User: {q}\nAssistant: {r[:240]}")
    return "\n---\n".join(chunks)


def _format_faq(query: str) -> str:
    items = retrieve_relevant_faq(query)
    if not items:
        return "No matching FAQ entries."
    return "\n".join([f"Q: {item.question}\nA: {item.answer}" for item in items])


@dataclass
class BotResult:
    query: str
    selected_style: str
    response: str
    evaluation: EvaluationResult
    feedback_label: str
    retry_used: bool
    implicit_feedback: Dict[str, bool]
    user_level: str
    emotion: str


class SelfOptimizingSupportBot:
    def __init__(self) -> None:
        configure_dspy_model()
        self.store = DataStore()
        self.optimizer = PromptOptimizer(self.store)

    def handle_query(
        self,
        query: str,
        rating: Optional[int] = None,
        liked: Optional[bool] = None,
        solved: Optional[bool] = None,
        user_level: Optional[str] = None,
    ) -> BotResult:
        now = timestamp_utc()
        seconds_since_last = _seconds_between(now, self.store.fetch_last_interaction_time())
        emotion = detect_user_emotion(query)
        level = detect_user_level(query, user_level)

        prompt_bank = self.store.get_prompt_bank()
        context = _format_context(self.store.fetch_recent_context(limit=3))
        faq_context = _format_faq(query)

        candidates = generate_candidates(
            query=query,
            prompt_bank=prompt_bank,
            conversation_context=context,
            faq_context=faq_context,
            personalization=level,
            emotion=emotion,
        )
        scored = {c.style_name: evaluate_response(query, c.response_text) for c in candidates}

        best_style = choose_best(scored)
        best_candidate = next(c for c in candidates if c.style_name == best_style)
        best_eval = scored[best_style]

        recent_queries = self.store.fetch_recent_queries()
        implicit = detect_implicit_feedback(query, recent_queries, seconds_since_last)

        retry_used = False
        if best_eval.total_score < LOW_SCORE_THRESHOLD or implicit.is_dissatisfied:
            retry_used = True
            alternatives = [c for c in candidates if c.style_name != best_style]
            alternatives = sorted(alternatives, key=lambda c: scored[c.style_name].total_score, reverse=True)
            if alternatives:
                alt = alternatives[0]
                best_style = alt.style_name
                best_eval = scored[best_style]
                best_candidate = alt
                best_candidate.response_text = (
                    "Let me explain this in a simpler way. " + best_candidate.response_text
                    if level == "beginner"
                    else "I'll provide a more technical fallback approach. " + best_candidate.response_text
                )

        feedback_label = parse_explicit_feedback(rating=rating, liked=liked, solved=solved)

        self.store.save_interaction(
            InteractionRecord(
                query=query,
                selected_style=best_style,
                response=best_candidate.response_text,
                total_score=best_eval.total_score,
                feedback=feedback_label,
                implicit_dissatisfied=implicit.is_dissatisfied,
                solved=solved if solved is not None else best_eval.solved_yes_no,
                confidence=best_eval.confidence,
                user_level=level,
                emotion=emotion,
                created_at=now,
            )
        )

        for style, evaluation in scored.items():
            self.optimizer.register_score(style, evaluation.total_score)
        self.optimizer.optimize()

        return BotResult(
            query=query,
            selected_style=best_style,
            response=best_candidate.response_text,
            evaluation=best_eval,
            feedback_label=feedback_label,
            retry_used=retry_used,
            implicit_feedback={
                "repeated_query": implicit.repeated_query,
                "dissatisfaction_phrase": implicit.dissatisfaction_phrase,
                "negative_sentiment": implicit.negative_sentiment,
                "long_delay": implicit.long_delay,
                "abandonment_risk": implicit.abandonment_risk,
            },
            user_level=level,
            emotion=emotion,
        )


if __name__ == "__main__":
    bot = SelfOptimizingSupportBot()
    result = bot.handle_query("I'm frustrated, app keeps logging me out and this is not helpful")
    print(result)
