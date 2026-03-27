"""Flask UI — upgraded AJAX-compatible proxy layer for the support system."""

from __future__ import annotations

import os

import requests
from flask import Flask, jsonify, render_template, request


API_BASE = os.getenv("SUPPORT_API_BASE", "http://127.0.0.1:8000")

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        return jsonify(r.json()), r.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 503


@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        payload = request.get_json()
        r = requests.post(f"{API_BASE}/chat", json=payload, timeout=20)
        return jsonify(r.json()), r.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 503


@app.route("/api/feedback", methods=["POST"])
def feedback():
    try:
        payload = request.get_json()
        r = requests.post(f"{API_BASE}/feedback", json=payload, timeout=10)
        return jsonify(r.json()), r.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 503


@app.route("/api/analytics")
def analytics():
    try:
        r = requests.get(f"{API_BASE}/analytics", timeout=10)
        return jsonify(r.json()), r.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 503


@app.route("/api/prompt-evolution")
def prompt_evolution():
    try:
        r = requests.get(f"{API_BASE}/prompt-evolution", timeout=10)
        return jsonify(r.json()), r.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 503


@app.route("/api/style-weights")
def style_weights():
    try:
        r = requests.get(f"{API_BASE}/style-weights", timeout=10)
        return jsonify(r.json()), r.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 503


@app.route("/api/kb/add", methods=["POST"])
def kb_add():
    try:
        payload = request.get_json()
        r = requests.post(f"{API_BASE}/kb/add", json=payload, timeout=10)
        return jsonify(r.json()), r.status_code
    except Exception:
        return jsonify({"status": "accepted", "note": "Add /kb/add to FastAPI to persist."}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
