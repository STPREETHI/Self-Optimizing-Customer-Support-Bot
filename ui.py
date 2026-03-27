"""Flask UI with HTML/CSS for the self-optimizing support system."""

from __future__ import annotations

import os

import requests
from flask import Flask, redirect, render_template, request, url_for


API_BASE = os.getenv("SUPPORT_API_BASE", "http://127.0.0.1:8000")

app = Flask(__name__)
SESSION_CACHE = []


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        query = request.form.get("query", "").strip()
        user_id = request.form.get("user_id", "demo-user").strip() or "demo-user"

        if query:
            resp = requests.post(f"{API_BASE}/chat", json={"user_id": user_id, "query": query}, timeout=15)
            if resp.ok:
                payload = resp.json()
                SESSION_CACHE.append(payload)
            else:
                SESSION_CACHE.append({"error": f"Chat request failed: {resp.text}"})

        return redirect(url_for("index"))

    analytics = {}
    try:
        analytics = requests.get(f"{API_BASE}/analytics", timeout=10).json()
    except Exception:
        analytics = {"error": "Backend unavailable. Run FastAPI service first."}

    return render_template("index.html", interactions=list(reversed(SESSION_CACHE[-20:])), analytics=analytics)


@app.route("/submit-feedback", methods=["POST"])
def submit_feedback():
    interaction_id = int(request.form["interaction_id"])
    liked_raw = request.form.get("liked", "")
    liked = None if liked_raw == "" else liked_raw == "true"
    resolved_raw = request.form.get("resolved", "")
    resolved = None if resolved_raw == "" else resolved_raw == "true"
    feedback_text = request.form.get("feedback_text", "")

    requests.post(
        f"{API_BASE}/feedback",
        json={
            "interaction_id": interaction_id,
            "liked": liked,
            "feedback_text": feedback_text,
            "resolved": resolved,
        },
        timeout=15,
    )
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
