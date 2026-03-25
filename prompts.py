"""Prompt management and DSPy response generation for tech-support assistance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import dspy


PROMPT_STYLES: Dict[str, str] = {
    "concise": (
        "You are a tech-support assistant. Respond in short, practical language, "
        "max 5 sentences, with one actionable next step."
    ),
    "step_by_step": (
        "You are a troubleshooting agent. Give numbered steps with checks after each step, "
        "then provide escalation conditions."
    ),
    "detailed": (
        "You are a senior support engineer. Explain cause, diagnostics, fix, and prevention. "
        "Include caveats and monitoring recommendations."
    ),
    "empathetic": (
        "You are an empathetic support specialist. Acknowledge feelings, de-escalate frustration, "
        "and provide calm, clear steps with reassurance."
    ),
}


class StyledSupportSignature(dspy.Signature):
    user_query = dspy.InputField(desc="Current customer message")
    conversation_context = dspy.InputField(desc="Important previous conversation points")
    faq_context = dspy.InputField(desc="Retrieved tech-support knowledge snippets")
    personalization = dspy.InputField(desc="User profile preferences: beginner/expert")
    emotion = dspy.InputField(desc="Detected user emotion signal")
    style_instruction = dspy.InputField(desc="Prompt strategy instructions")
    response = dspy.OutputField(desc="Final support response")


class StyledSupportResponder(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.generator = dspy.Predict(StyledSupportSignature)

    def forward(
        self,
        user_query: str,
        conversation_context: str,
        faq_context: str,
        personalization: str,
        emotion: str,
        style_instruction: str,
    ) -> dspy.Prediction:
        return self.generator(
            user_query=user_query,
            conversation_context=conversation_context,
            faq_context=faq_context,
            personalization=personalization,
            emotion=emotion,
            style_instruction=style_instruction,
        )


@dataclass
class CandidateResponse:
    style_name: str
    prompt_text: str
    response_text: str


def _fallback_response(query: str, style_name: str, emotion: str, profile: str) -> str:
    tone_prefix = "I understand this is frustrating. " if emotion == "frustrated" else ""
    audience = "simple terms" if profile == "beginner" else "technical detail"

    if style_name == "step_by_step":
        body = (
            f"1) Reproduce '{query}' and capture exact error.\n"
            "2) Check auth/session, network, and latest deploy logs.\n"
            "3) Apply fix and run validation test.\n"
            "4) If unresolved, escalate with logs + timestamps."
        )
    elif style_name == "concise":
        body = f"For '{query}', restart the service, verify config, and test again. Escalate with logs if issue persists."
    elif style_name == "empathetic":
        body = f"You're doing the right thing checking this. For '{query}', let's verify settings, retry once, then escalate quickly if needed."
    else:
        body = (
            f"For '{query}', probable causes include config drift, stale sessions, or dependency issues. "
            "Validate logs, apply fix, retest, and add preventive monitoring."
        )

    return f"{tone_prefix}({audience}) {body}"


def generate_candidates(
    query: str,
    prompt_bank: Dict[str, str],
    conversation_context: str,
    faq_context: str,
    personalization: str,
    emotion: str,
) -> List[CandidateResponse]:
    responder = StyledSupportResponder()
    candidates: List[CandidateResponse] = []

    for style_name, prompt_text in prompt_bank.items():
        try:
            prediction = responder(
                user_query=query,
                conversation_context=conversation_context,
                faq_context=faq_context,
                personalization=personalization,
                emotion=emotion,
                style_instruction=prompt_text,
            )
            response_text = prediction.response.strip()
            if not response_text:
                response_text = _fallback_response(query, style_name, emotion, personalization)
        except Exception:
            response_text = _fallback_response(query, style_name, emotion, personalization)

        candidates.append(CandidateResponse(style_name, prompt_text, response_text))

    return candidates
