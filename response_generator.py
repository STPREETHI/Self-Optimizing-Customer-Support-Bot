"""Response generation module with RAG, multi-style drafting, and ranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import dspy


PROMPT_STYLES = {
    "professional": "Respond politely and professionally, keeping it concise.",
    "empathetic": "Acknowledge frustration, then provide clear support steps.",
    "technical": "Provide technical root-cause and concrete diagnostic steps.",
    "simple": "Explain in beginner-friendly language with numbered steps.",
}


class SupportResponseSignature(dspy.Signature):
    query = dspy.InputField()
    intent = dspy.InputField()
    entities = dspy.InputField()
    memory = dspy.InputField(desc="Recent conversation turns")
    retrieved_knowledge = dspy.InputField(desc="Relevant FAQ/past solved answers")
    style = dspy.InputField(desc="Prompt style instruction")
    response = dspy.OutputField(desc="Final support answer")


class SupportResponder(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.predict = dspy.Predict(SupportResponseSignature)

    def forward(
        self,
        query: str,
        intent: str,
        entities: str,
        memory: str,
        retrieved_knowledge: str,
        style: str,
    ) -> dspy.Prediction:
        return self.predict(
            query=query,
            intent=intent,
            entities=entities,
            memory=memory,
            retrieved_knowledge=retrieved_knowledge,
            style=style,
        )


@dataclass
class Candidate:
    style: str
    prompt: str
    response: str


def _fallback_response(query: str, style: str) -> str:
    if style == "simple":
        return (
            f"I can help with '{query}'. 1) Check account/session settings. "
            "2) Retry the action. 3) Share error logs if it still fails."
        )
    if style == "technical":
        return (
            f"For '{query}', inspect auth tokens, request traces, and deployment logs. "
            "Apply fix, rerun tests, and monitor error rate."
        )
    if style == "empathetic":
        return (
            "I understand this is frustrating. "
            f"For '{query}', let’s verify settings, test again, and escalate quickly if needed."
        )
    return f"For '{query}', I recommend validating configuration, reproducing the issue, and confirming the fix."


def _format_memory(history: List[Tuple[str, str]]) -> str:
    if not history:
        return "No previous context."
    return "\n".join([f"User: {q}\nBot: {r[:220]}" for q, r in history])


def _format_retrieved(snippets: List[str]) -> str:
    if not snippets:
        return "No retrieved snippets."
    return "\n---\n".join(snippets)


def generate_multi_candidates(
    query: str,
    intent: str,
    entities: Dict[str, str],
    history: List[Tuple[str, str]],
    retrieved_snippets: List[str],
    style_boost: Dict[str, float],
) -> List[Candidate]:
    responder = SupportResponder()
    memory = _format_memory(history)
    retrieved_text = _format_retrieved(retrieved_snippets)

    candidates: List[Candidate] = []
    for style, prompt in PROMPT_STYLES.items():
        tuned_prompt = f"{prompt} Priority multiplier: {style_boost.get(style, 1.0):.2f}."
        try:
            prediction = responder(
                query=query,
                intent=intent,
                entities=str(entities),
                memory=memory,
                retrieved_knowledge=retrieved_text,
                style=tuned_prompt,
            )
            response = prediction.response.strip() if prediction.response else ""
            if not response:
                response = _fallback_response(query, style)
        except Exception:
            response = _fallback_response(query, style)

        candidates.append(Candidate(style=style, prompt=tuned_prompt, response=response))

    return candidates
