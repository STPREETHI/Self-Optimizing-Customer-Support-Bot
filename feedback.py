"""Explicit + implicit feedback, personalization hints, and emotion detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional


DISSATISFACTION_PHRASES = ["not helpful", "wrong", "didn't help", "still broken", "this is useless"]
FRUSTRATED_WORDS = {"angry", "frustrated", "annoyed", "upset", "hate", "terrible"}
CONFUSED_WORDS = {"confused", "don't understand", "unclear", "what does this mean"}


@dataclass
class ImplicitFeedback:
    repeated_query: bool
    dissatisfaction_phrase: bool
    negative_sentiment: bool
    long_delay: bool
    abandonment_risk: bool

    @property
    def is_dissatisfied(self) -> bool:
        return any(
            [
                self.repeated_query,
                self.dissatisfaction_phrase,
                self.negative_sentiment,
                self.long_delay,
                self.abandonment_risk,
            ]
        )


def detect_user_emotion(user_query: str) -> str:
    lowered = user_query.lower()
    if any(token in lowered for token in FRUSTRATED_WORDS):
        return "frustrated"
    if any(token in lowered for token in CONFUSED_WORDS):
        return "confused"
    return "neutral"


def detect_user_level(user_query: str, selected_level: Optional[str]) -> str:
    if selected_level in {"beginner", "expert"}:
        return selected_level
    lowered = user_query.lower()
    if any(token in lowered for token in ["api", "latency", "logs", "stack trace", "timeout"]):
        return "expert"
    return "beginner"


def detect_implicit_feedback(
    user_query: str,
    prior_queries: Iterable[str],
    seconds_since_last_message: Optional[float],
) -> ImplicitFeedback:
    lowered_query = user_query.lower().strip()
    normalized_prior = [q.lower().strip() for q in prior_queries]

    repeated_query = lowered_query in normalized_prior[:5]
    dissatisfaction_phrase = any(phrase in lowered_query for phrase in DISSATISFACTION_PHRASES)
    negative_sentiment = sum(word in lowered_query for word in FRUSTRATED_WORDS) >= 1

    long_delay = seconds_since_last_message is not None and seconds_since_last_message > 240
    abandonment_risk = seconds_since_last_message is not None and seconds_since_last_message > 900

    return ImplicitFeedback(
        repeated_query=repeated_query,
        dissatisfaction_phrase=dissatisfaction_phrase,
        negative_sentiment=negative_sentiment,
        long_delay=long_delay,
        abandonment_risk=abandonment_risk,
    )


def parse_explicit_feedback(
    rating: Optional[int] = None,
    liked: Optional[bool] = None,
    solved: Optional[bool] = None,
) -> str:
    if liked is True:
        return "like"
    if liked is False:
        return "dislike"
    if solved is True:
        return "solved_yes"
    if solved is False:
        return "solved_no"
    if rating is None:
        return "none"
    if rating >= 4:
        return "positive_rating"
    if rating <= 2:
        return "negative_rating"
    return "neutral_rating"
