# Self-Optimizing Customer Support System using Feedback-Driven Learning

This project implements a production-style support system that improves itself after each interaction.

## 1) System Architecture (Textual Diagram)

```text
[User/UI (Flask HTML/CSS)]
        |
        v
[FastAPI Gateway: /chat, /feedback, /analytics]
        |
        +--> [Query Processor Agent]
        |       - intent classification
        |       - entity extraction
        |       - embedding generation
        |
        +--> [Retrieval Agent (RAG)]
        |       - vector search over KB + historical snippets
        |
        +--> [Response Generator Agent]
        |       - DSPy multi-style candidate generation
        |       - context memory conditioning
        |
        +--> [Evaluator/Ranker Agent]
        |       - weighted scoring + style reward boost
        |
        +--> [Persistence Layer]
                - interactions
                - feedback
                - prompt profiles
                - prompt evolution history
                - vectorized KB docs

Feedback Loop:
/user feedback -> reward calculation -> prompt/style weight updates
-> weak-prompt evolution -> better ranking in next queries
```

## 2) Modules

- `query_processor.py` — query understanding (intent, entities, embeddings)
- `response_generator.py` — DSPy generation with context + RAG snippets
- `feedback_engine.py` — explicit feedback parsing + reward/failure reason detection
- `optimizer.py` — SQLite store, RAG retrieval, style weighting, prompt evolution, analytics
- `main.py` — FastAPI backend APIs and orchestration
- `ui.py` — Flask UI using `templates/index.html` + `static/style.css`

## 3) Sample Dataset Schema (SQLite)

### `interactions`
- `id` (PK)
- `user_id`
- `query`
- `intent`
- `entities`
- `chosen_style`
- `response`
- `score`
- `confidence`
- `resolved`
- `created_at`

### `feedback`
- `id` (PK)
- `interaction_id`
- `liked`
- `feedback_text`
- `resolved`
- `reward`
- `failure_reason`
- `created_at`

### `prompt_profiles`
- `style` (PK)
- `weight`
- `uses`
- `avg_reward`
- `prompt_hint`

### `prompt_evolution`
- `id` (PK)
- `style`
- `before_hint`
- `after_hint`
- `reason`
- `created_at`

### `kb_vectors`
- `id` (PK)
- `source_text`
- `embedding`

## 4) API Endpoints (Examples)

- `GET /health`
- `POST /chat`
  - request: `{"user_id": "u1", "query": "my app keeps timing out"}`
  - response: ranked answer with intent/entities/style/score/retrieved context
- `POST /feedback`
  - request: `{"interaction_id": 10, "liked": false, "feedback_text": "not helpful", "resolved": false}`
  - effect: reward update + style/prompt optimization
- `GET /analytics`
- `GET /prompt-evolution`

## 5) Feedback Learning Loop (How it improves)

1. Generate multiple candidate responses using different styles.
2. Score and rank candidates (content score × historical style weight).
3. Save chosen response as an interaction.
4. Collect explicit feedback (like/dislike, text, resolved).
5. Convert feedback into a numeric reward.
6. Update style weights and average rewards.
7. Mutate weak prompts (GEPA-like) using stronger prompt patterns.
8. Use updated weights/hints in next query.

This creates **continuous online improvement** without retraining.

## 6) Run Instructions

Install:

```bash
pip install -r requirements.txt
```

Start backend:

```bash
python main.py
```

Start Flask UI (new terminal):

```bash
python ui.py
```

Open: `http://127.0.0.1:5000`

## 7) Evaluation Dimensions

- Response quality score (proxy for accuracy/helpfulness)
- User satisfaction (`feedback.reward`)
- Resolution rate
- Most common issue intents
- Trend over time

## 8) Notes

- Embedding layer is implemented with deterministic vectors for portability. In production, swap with OpenAI embeddings or sentence-transformers + FAISS/Pinecone.
- DSPy is used for prompt-structured generation.
