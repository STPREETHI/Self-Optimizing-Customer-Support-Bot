# Self-Optimizing Customer Support Bot (Tech Support Domain)

An end-to-end Python project demonstrating a **real-time self-improving support bot**.

## What’s implemented

- **Multi-style response generation** (concise, step-by-step, detailed, empathetic) with DSPy.
- **Hybrid evaluation** (rule-based + optional LLM judge) with weighted scoring:
  - Clarity: 30%
  - Correctness: 30%
  - Helpfulness: 40%
- **Problem solved signal**: “Did this solve the problem?” (Yes/No).
- **Explicit feedback**: like/dislike, rating, solved.
- **Implicit feedback**:
  - repeated queries
  - dissatisfaction phrases ("not helpful", "wrong")
  - long delay before next message
  - abandonment risk signal
- **Context memory** from prior conversation turns.
- **Domain specialization**: tech support FAQ retrieval grounding.
- **Failure recovery**: retry with alternative style and simplification/technical fallback.
- **GEPA-like prompt evolution**:
  - tracks per-prompt performance
  - ranks prompts
  - mutates weak prompts with examples, tone change, and constraints
  - stores before/after prompt versions
- **Analytics dashboard**:
  - satisfaction rate
  - best-performing prompt
  - most failed queries
  - improvement trend over time
- **Personalization layer**: beginner vs expert response tuning.
- **Emotion detection + adaptive tone**: frustrated/confused/neutral.

## Files

- `main.py` — orchestration + learning loop
- `prompts.py` — DSPy prompt strategies and candidate generation
- `evaluator.py` — weighted hybrid evaluator
- `feedback.py` — explicit/implicit feedback + emotion/profile detection
- `optimizer.py` — SQLite storage, ranking, GEPA-style mutation, analytics
- `domain_kb.py` — tech support FAQ knowledge base
- `ui.py` — Streamlit chat + dashboard + explainability

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional:

```bash
export OPENAI_API_KEY=your_key_here
```

## Run

```bash
python main.py
streamlit run ui.py
```

## Notes

- If LLM or network is unavailable, generation and evaluation gracefully fallback to rule-based/deterministic behavior so the demo remains functional.
- Interactions are persisted in `support_bot.db`.
