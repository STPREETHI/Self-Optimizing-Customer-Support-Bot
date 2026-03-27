"""Feedback collection and reward modeling for continuous learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class FeedbackEvent:
    interaction_id: int
    liked: Optional[bool]
    feedback_text: str
    resolved: Optional[bool]
    reward: float
    failure_reason: str


def infer_failure_reason(feedback_text: str, resolved: Optional[bool]) -> str:
    lowered = feedback_text.lower().strip()

    if resolved is False and any(token in lowered for token in ["wrong", "incorrect"]):
        return "incorrect_answer"
    if any(token in lowered for token in ["unclear", "confusing"]):
        return "low_clarity"
    if any(token in lowered for token in ["too long", "verbose"]):
        return "verbosity_issue"
    if any(token in lowered for token in ["rude", "tone"]):
        return "tone_issue"
    if resolved is False:
        return "unresolved_issue"
    return "none"


def compute_reward(liked: Optional[bool], resolved: Optional[bool], feedback_text: str) -> float:
    reward = 0.0
    if liked is True:
        reward += 1.0
    if liked is False:
        reward -= 1.0
    if resolved is True:
        reward += 1.2
    if resolved is False:
        reward -= 1.2

    lowered = feedback_text.lower()
    if any(token in lowered for token in ["great", "helpful", "thanks"]):
        reward += 0.4
    if any(token in lowered for token in ["not helpful", "wrong", "bad"]):
        reward -= 0.6

    return round(reward, 3)


def build_feedback_event(
    interaction_id: int,
    liked: Optional[bool],
    feedback_text: Optional[str],
    resolved: Optional[bool],
) -> FeedbackEvent:
    feedback_text = feedback_text or ""
    failure_reason = infer_failure_reason(feedback_text=feedback_text, resolved=resolved)
    reward = compute_reward(liked=liked, resolved=resolved, feedback_text=feedback_text)
    return FeedbackEvent(
        interaction_id=interaction_id,
        liked=liked,
        feedback_text=feedback_text,
        resolved=resolved,
        reward=reward,
        failure_reason=failure_reason,
    )
