"""Domain-specialized knowledge base for Tech Support use-cases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class FAQItem:
    question: str
    answer: str
    tags: List[str]


TECH_SUPPORT_FAQ: List[FAQItem] = [
    FAQItem(
        question="Why am I getting logged out repeatedly?",
        answer=(
            "Common causes include expired session tokens, browser cookie cleanup, "
            "clock drift, or server-side auth timeout. Fix by syncing device time, "
            "allowing cookies, and re-authenticating after clearing stale sessions."
        ),
        tags=["login", "session", "authentication"],
    ),
    FAQItem(
        question="The app is slow. What should I check first?",
        answer=(
            "Check network latency, CPU/memory usage, and recent deployments. "
            "Then clear local cache, disable heavy extensions, and inspect error logs."
        ),
        tags=["performance", "latency", "slow"],
    ),
    FAQItem(
        question="How do I troubleshoot API errors?",
        answer=(
            "Validate API key scope, check rate limits, verify request schema, "
            "and correlate timestamps with server logs for 4xx/5xx diagnostics."
        ),
        tags=["api", "error", "integration"],
    ),
    FAQItem(
        question="Notifications are not arriving.",
        answer=(
            "Check notification channel status, spam filters, device permissions, "
            "and webhook delivery logs. Re-enable channel and send a test notification."
        ),
        tags=["notification", "email", "webhook"],
    ),
]


def retrieve_relevant_faq(query: str, limit: int = 2) -> List[FAQItem]:
    """Simple keyword overlap retrieval for domain grounding."""

    query_tokens = {w.strip(".,?!").lower() for w in query.split() if len(w) > 2}
    scored = []
    for item in TECH_SUPPORT_FAQ:
        bag = set(item.tags) | {w.lower() for w in item.question.split()}
        overlap = len(query_tokens & bag)
        scored.append((overlap, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for score, item in scored if score > 0][:limit]
