"""Streamlit interface with chat, explainability, and analytics dashboard."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from main import SelfOptimizingSupportBot


st.set_page_config(page_title="Self-Optimizing Customer Support Bot", layout="wide")

if "bot" not in st.session_state:
    st.session_state.bot = SelfOptimizingSupportBot()
if "messages" not in st.session_state:
    st.session_state.messages = []

bot = st.session_state.bot

st.title("🤖 Self-Optimizing Customer Support Bot (Tech Support)")
st.caption("Multi-style DSPy responses, memory, retries, prompt evolution, and real-time learning")

with st.sidebar:
    st.header("Analytics Dashboard")
    stats = bot.store.analytics()
    st.metric("Total interactions", stats["total_interactions"])
    st.metric(
        "Satisfaction rate",
        f"{stats['satisfaction_rate']:.1%}" if stats["satisfaction_rate"] is not None else "N/A",
    )
    st.metric("Best prompt", stats["best_prompt"] or "N/A")

    st.subheader("Most Failed Queries")
    if stats["most_failed_queries"]:
        st.table(pd.DataFrame(stats["most_failed_queries"], columns=["Query", "Failures"]))
    else:
        st.info("No failed-query records yet.")

    st.subheader("Improvement Over Time")
    if stats["improvement_trend"]:
        trend_df = pd.DataFrame(stats["improvement_trend"], columns=["date", "avg_score"]).set_index("date")
        st.line_chart(trend_df)
    else:
        st.info("Not enough data to show trend.")

st.subheader("Chat")
user_query = st.text_input("Describe your technical issue")
cols = st.columns(5)
with cols[0]:
    liked = st.button("👍 Like")
with cols[1]:
    disliked = st.button("👎 Dislike")
with cols[2]:
    rating = st.slider("Rating", 1, 5, 3)
with cols[3]:
    solved = st.selectbox("Solved?", ["Unknown", "Yes", "No"])
with cols[4]:
    user_level = st.selectbox("User Level", ["auto", "beginner", "expert"])

if st.button("Send") and user_query.strip():
    solved_value = None if solved == "Unknown" else solved == "Yes"
    result = bot.handle_query(
        query=user_query.strip(),
        rating=rating,
        liked=True if liked else False if disliked else None,
        solved=solved_value,
        user_level=None if user_level == "auto" else user_level,
    )

    st.session_state.messages.append(
        {
            "query": user_query,
            "response": result.response,
            "style": result.selected_style,
            "score": result.evaluation.total_score,
            "confidence": result.evaluation.confidence,
            "reasoning": result.evaluation.reasoning,
            "solved": result.evaluation.solved_yes_no,
            "retry": result.retry_used,
            "implicit": result.implicit_feedback,
            "user_level": result.user_level,
            "emotion": result.emotion,
        }
    )

if st.session_state.messages:
    st.subheader("Conversation History")
    for item in reversed(st.session_state.messages):
        with st.expander(f"Q: {item['query'][:90]}"):
            st.markdown(f"**Response style:** `{item['style']}`")
            st.markdown(f"**Detected profile:** `{item['user_level']}` | **Emotion:** `{item['emotion']}`")
            st.markdown(f"**Answer:** {item['response']}")
            st.markdown(f"**Score:** `{item['score']}` | **Confidence:** `{item['confidence']}`")
            st.markdown(f"**Did this solve the problem?** `{item['solved']}`")
            st.markdown(f"**Why this answer?** {item['reasoning']}")
            st.markdown(f"**Retry used:** `{item['retry']}`")
            st.json(item["implicit"])

st.subheader("Prompt Evolution (Before vs After)")
versions = bot.store.fetch_prompt_improvements(limit=5)
if versions:
    for style, old_prompt, new_prompt, created_at in versions:
        with st.expander(f"{style} | {created_at}"):
            st.markdown("**Before**")
            st.code(old_prompt)
            st.markdown("**After**")
            st.code(new_prompt)
else:
    st.info("No prompt mutations yet. Continue chatting to trigger optimization.")
