"""Hybrid (rule-based + optional LLM) evaluation engine with weighted scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import dspy


WEIGHTS = {"clarity": 0.30, "correctness": 0.30, "helpfulness": 0.40}


@dataclass
class EvaluationResult:
    clarity: float
    correctness: float
    helpfulness: float
    total_score: float
    solved_yes_no: bool
    confidence: float
    reasoning: str


class LLMJudgeSignature(dspy.Signature):
    query = dspy.InputField(desc="User issue")
    response = dspy.InputField(desc="Candidate support response")
    clarity = dspy.OutputField(desc="0-10 clarity score")
    correctness = dspy.OutputField(desc="0-10 correctness score")
    helpfulness = dspy.OutputField(desc="0-10 helpfulness score")
    solved = dspy.OutputField(desc="yes or no whether likely solved")
    rationale = dspy.OutputField(desc="brief reason")


class LLMJudge(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.predictor = dspy.Predict(LLMJudgeSignature)

    def forward(self, query: str, response: str) -> dspy.Prediction:
        return self.predictor(query=query, response=response)


def _bounded(score: float) -> float:
    return max(0.0, min(10.0, round(score, 2)))


def _rule_based_scores(user_query: str, response_text: str) -> Dict[str, float]:
    words = response_text.split()
    wc = len(words)
    query_terms = {w.lower().strip(".,?!") for w in user_query.split() if len(w) > 3}
    response_terms = {w.lower().strip(".,?!") for w in words}
    overlap = len(query_terms & response_terms)

    clarity = 5.8 + (1.2 if "\n" in response_text or "1)" in response_text else 0.0)
    clarity += 1.2 if 25 <= wc <= 220 else -0.8

    correctness = 5.5 + min(2.5, overlap * 0.5)
    correctness += 1.2 if any(x in response_text.lower() for x in ["verify", "check", "logs"]) else 0.0

    helpfulness = 5.8
    helpfulness += 1.8 if any(x in response_text.lower() for x in ["step", "1)", "2)", "3)"]) else 0.0
    helpfulness += 1.0 if any(x in response_text.lower() for x in ["if unresolved", "escalate", "next"]) else 0.0

    return {
        "clarity": _bounded(clarity),
        "correctness": _bounded(correctness),
        "helpfulness": _bounded(helpfulness),
    }


def _llm_scores(user_query: str, response_text: str) -> Optional[Dict[str, float]]:
    try:
        judge = LLMJudge()
        result = judge(query=user_query, response=response_text)
        return {
            "clarity": _bounded(float(result.clarity)),
            "correctness": _bounded(float(result.correctness)),
            "helpfulness": _bounded(float(result.helpfulness)),
            "solved": str(result.solved).strip().lower() in {"yes", "true", "1"},
            "rationale": str(result.rationale).strip(),
        }
    except Exception:
        return None


def evaluate_response(user_query: str, response_text: str) -> EvaluationResult:
    rule = _rule_based_scores(user_query, response_text)
    llm = _llm_scores(user_query, response_text)

    if llm:
        clarity = round((rule["clarity"] + llm["clarity"]) / 2, 2)
        correctness = round((rule["correctness"] + llm["correctness"]) / 2, 2)
        helpfulness = round((rule["helpfulness"] + llm["helpfulness"]) / 2, 2)
        solved = bool(llm["solved"])
        llm_reason = llm["rationale"]
    else:
        clarity, correctness, helpfulness = rule["clarity"], rule["correctness"], rule["helpfulness"]
        solved = helpfulness >= 7.0 and correctness >= 6.5
        llm_reason = "LLM judge unavailable; used rule-based evaluator only."

    total = round(
        clarity * WEIGHTS["clarity"]
        + correctness * WEIGHTS["correctness"]
        + helpfulness * WEIGHTS["helpfulness"],
        2,
    )
    confidence = round(min(0.99, max(0.15, total / 10)), 2)

    reasoning = (
        f"Weighted score => clarity({clarity})*0.30 + correctness({correctness})*0.30 + "
        f"helpfulness({helpfulness})*0.40 = {total}. "
        f"Solved estimate: {'Yes' if solved else 'No'}. {llm_reason}"
    )

    return EvaluationResult(
        clarity=clarity,
        correctness=correctness,
        helpfulness=helpfulness,
        total_score=total,
        solved_yes_no=solved,
        confidence=confidence,
        reasoning=reasoning,
    )


def choose_best(scored_candidates: Dict[str, EvaluationResult]) -> str:
    return max(scored_candidates.items(), key=lambda pair: pair[1].total_score)[0]
