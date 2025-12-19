"""
Seeds of Truth Flask App

Features:
- Serves templates/index.html at "/"
- RAG search endpoint at POST /api/search (always allowed)
- Password-gated endpoints:
    - POST /api/unlock  -> sets session["unlocked"] = True/False
    - POST /api/chat    -> requires unlocked
    - POST /api/ab      -> requires unlocked
- Basic stub endpoints: /api/feedback, /api/status
- Dev-friendly HTML no-cache + optional template auto-reload
- Initializes global retrieval_state once at startup
"""

import os
import hmac
import asyncio
import logging
from datetime import datetime
from typing import List
import model_controller
import db

from flask import Flask, render_template, request, jsonify, session

import random
import time

# ---------------------------------------------------------------------------
# Logging (uses logging_config if present, otherwise falls back to std logging)
# ---------------------------------------------------------------------------

def _setup_logger() -> logging.Logger:
    try:
        import logging_config  # type: ignore
        return logging_config.setup_logging(logging.INFO)
    except Exception:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
        return logging.getLogger("seedsoftruth")


app_logger = _setup_logger()


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)

# IMPORTANT: set a real secret in production or sessions won't be secure/stable
# Put this into systemd Environment= "FLASK_SECRET_KEY=..."
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-change-me")

# chat requests shouldn't exceed this
MAX_ALLOWED_NEW_TOKENS = 1200


# Dev-style template reloading:
# - Safe for dev
# - Turn OFF for production by setting TEMPLATES_AUTO_RELOAD=0
if os.environ.get("TEMPLATES_AUTO_RELOAD", "1") == "1":
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.jinja_env.auto_reload = True


# ---------------------------------------------------------------------------
# Password gating configuration
# ---------------------------------------------------------------------------

def _parse_passwords(env_value: str) -> List[str]:
    """
    Parse comma-separated passwords.
    Example: SOT_PASSWORDS="alpha123,beta456"
    """
    return [p.strip() for p in (env_value or "").split(",") if p.strip()]


ALLOWED_PASSWORDS = _parse_passwords(os.environ.get("SOT_PASSWORDS", ""))


def _is_unlocked() -> bool:
    """Session-based gate. Set by /api/unlock."""
    return bool(session.get("unlocked", False))


def _require_unlocked():
    """Helper: return (json, status_code) if locked, else None."""
    if not _is_unlocked():
        return jsonify({"ok": False, "error": "locked", "message": "Not today"}), 403
    return None


# ---------------------------------------------------------------------------
# RAG retrieval state 
# ---------------------------------------------------------------------------

retrieval_state = None

def init_state():
    """
    Initialize global retrieval_state once at startup.

    Notes:
    - This should run once per gunicorn worker process.
    - If boot() is expensive, consider moving it to a lazy init or shared service.
    """
    global retrieval_state
    app_logger.info("init_state called")

    try:
        import rag_controller  # type: ignore
    except Exception as e:
        app_logger.error("rag_controller import failed: %s", e)
        retrieval_state = None
        return

    try:
        # colleague's pattern: from rag_controller import boot
        boot = getattr(rag_controller, "boot", None)
        if boot is None:
            app_logger.error("rag_controller.boot not found")
            retrieval_state = None
            return

        retrieval_state = boot()
        if retrieval_state is None:
            app_logger.error("boot() returned None retrieval_state")
        else:
            app_logger.info("init_state done (retrieval_state initialized)")
    except Exception as e:
        app_logger.error("init_state error: %s", e)
        retrieval_state = None


# Initialize once at import time (works well under systemd+gunicorn)
init_state()
db.init_db()


# ---------------------------------------------------------------------------
# Routes: UI
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return render_template("index.html")


@app.after_request
def no_cache_html(resp):
    """
    Prevent caching for HTML pages (helps when iterating).
    Static assets should be cache-busted via ?v=... or hashed filenames.
    """
    if resp.mimetype == "text/html":
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
    return resp


# ---------------------------------------------------------------------------
# Routes: Auth / Unlock
# ---------------------------------------------------------------------------

@app.post("/api/unlock")
def api_unlock():
    """
    Verify a password and set the session gate.

    Request JSON:
      { "password": "..." }

    Response:
      200 { ok: true,  message: "Access Granted" }
      403 { ok: false, message: "Not today" }
    """
    payload = request.get_json(silent=True) or {}
    pw = (payload.get("password") or "").strip()

    if not pw or not ALLOWED_PASSWORDS:
        # If no passwords configured, default locked
        session["unlocked"] = False
        return jsonify({"ok": False, "message": "Not today"}), 403

    # constant-time compare across allowed passwords
    ok = any(hmac.compare_digest(pw, real) for real in ALLOWED_PASSWORDS)

    session["unlocked"] = bool(ok)
    if ok:
        return jsonify({"ok": True, "message": "Access Granted"}), 200
    return jsonify({"ok": False, "message": "Not today"}), 403


@app.get("/api/access")
def api_access():
    """
    Optional helper endpoint so the front-end can query current access state.
    """
    return jsonify({"ok": True, "unlocked": _is_unlocked()}), 200


# ---------------------------------------------------------------------------
# Routes: Demo / Ping (your original working endpoint)
# ---------------------------------------------------------------------------

@app.post("/api/ping")
def api_ping():
    data = request.get_json(force=True)
    return jsonify({
        "ok": True,
        "received": data,
        "server_time": datetime.utcnow().isoformat() + "Z",
        "message": "Flask received your message successfully."
    })


# ---------------------------------------------------------------------------
# Routes: Search (ALWAYS allowed)
# ---------------------------------------------------------------------------

@app.post("/api/search")
def on_search():
    """
    Search endpoint using colleague's RAG controller.

    Request JSON:
      { "query": "string", "max_n": 20 }

    Response JSON:
      { ok: true, query: "...", num_results: N, results: [...] }
    """
    global retrieval_state

    app_logger.info("search request received")

    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"ok": False, "error": "Invalid or missing JSON body"}), 400

    query = payload.get("query")
    if not isinstance(query, str) or not query.strip():
        app_logger.warning("Search request query invalid")
        return jsonify({"ok": False, "error": "Field 'query' must be a non-empty string"}), 400

    max_n = payload.get("max_n", 20)
    if not isinstance(max_n, int):
        app_logger.warning("Search request max_n invalid type")
        return jsonify({"ok": False, "error": "Field 'max_n' must be an integer"}), 400

    if max_n <= 0 or max_n > 200:
        app_logger.warning("Search request max_n out of range: %s", max_n)
        return jsonify({"ok": False, "error": "Field 'max_n' must be between 1 and 200"}), 400

    if retrieval_state is None:
        app_logger.error("retrieval_state not initialized")
        return jsonify({"ok": False, "error": "Search system not initialized"}), 503

    try:
        import rag_controller  # type: ignore
    except Exception as e:
        app_logger.error("rag_controller import failed during search: %s", e)
        return jsonify({"ok": False, "error": "Search system not available"}), 503

    app_logger.info("Search request", extra={
        "query": query,
        "len": len(query),
        "max_n": max_n,
        "remote_addr": request.remote_addr,
    })

    # Call async function from sync Flask route
    try:
        results = asyncio.run(
            rag_controller.search_references(  # type: ignore
                retrieval_state,
                query,
                top_k=max_n,
            )
        )
    except RuntimeError:
        # If an event loop already exists, use it
        try:
            loop = asyncio.get_event_loop()
            results = loop.run_until_complete(
                rag_controller.search_references(  # type: ignore
                    retrieval_state,
                    query,
                    top_k=max_n,
                )
            )
        except Exception as inner:
            app_logger.error("Internal async execution error: %s", inner)
            return jsonify({"ok": False, "error": "Internal async execution error", "detail": str(inner)}), 500
    except Exception as e:
        app_logger.error("Search failed: %s", e)
        return jsonify({"ok": False, "error": "Search failed", "detail": str(e)}), 500

    app_logger.info("Search request completed.")
    return jsonify({
        "ok": True,
        "query": query,
        "num_results": (results or {}).get("num_results", 0),
        "results": (results or {}).get("results", []),
    }), 200


# ---------------------------------------------------------------------------
# Routes: Chat (LOCKED unless unlocked)
# ---------------------------------------------------------------------------

@app.route("/api/chat", methods=["POST","GET"])
def on_chat():
    """
    Chat endpoint (gated).
    The UI should disable Chat mode unless unlocked, but this is the real enforcement.
    """
    locked = _require_unlocked()
    if locked:
        return locked

    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({
            "ok": False,
            "error": "Invalid or missing JSON body"
        }), 400

    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        app_logger.warn("Chat request prompt was invalid")
        return jsonify({
            "ok": False,
            "error": "Field 'prompt' must be a non-empty string"
        }), 400

    max_new_tokens = payload.get("max_new_tokens", 250)
    if not isinstance(max_new_tokens, int):
        app_logger.warn("Chat request max_new_tokens was invalid")
        return jsonify({
            "ok": False,
            "error": "Field 'max_new_tokens' must be int"
        }), 400
    if max_new_tokens < 1 or max_new_tokens > MAX_ALLOWED_NEW_TOKENS:
        app_logger.warn("Chat request max_new_tokens was out of the allowed range")
        return jsonify({
            "ok": False,
            "error": "Field 'max_new_tokens' must be within allowed range"
        }), 400

    user_id = payload.get("user_id", "none")
    if not isinstance(user_id, str) or not prompt.strip():
        app_logger.warn("Chat request user_id was invalid")
        return jsonify({
            "ok": False,
            "error": "Field 'user_id' must be a string"
        }), 400

    try:
        job_id = db.insert_job(user_id, prompt)

        response = model_controller.send_prompt_to_model(prompt).strip()

        db.mark_done(job_id, response)

        return jsonify({
            "ok": True,
            "response": response,
            "status": "success",
        }), 200
    except RuntimeError as e:
        app_logger.exception("Runtime error occurred during chat")
        return jsonify({
            "ok": False,
            "error": "Search failed",
            "detail": str(e)
        }), 500

# ---------------------------------------------------------------------------
# Routes: A/B test (LOCKED unless unlocked)
# ---------------------------------------------------------------------------

@app.post("/api/ab")
def on_ab():
    """
    A/B endpoint (gated). Return two candidate answers.
    """
    locked = _require_unlocked()
    if locked:
        return locked

    payload = request.get_json(silent=True) or {}
    # TODO: Replace with real A/B logic
    return jsonify({
        "ok": True,
        "a": "A response (placeholder)",
        "b": "B response (placeholder)",
        "received": payload,
    }), 200


# ---------------------------------------------------------------------------
# Routes: Feedback + Status (not gated)
# ---------------------------------------------------------------------------

@app.route("/api/feedback", methods=["POST", "GET"])
def on_feedback():
    return jsonify({"ok": True, "status": "feedback was successful"}), 200


@app.route("/api/status", methods=["POST", "GET"])
def on_status():
    return jsonify({
        "ok": True,
        "status": "status was successful",
        "unlocked": _is_unlocked(),
        "retrieval_state_ready": retrieval_state is not None,
    }), 200

@app.get("/api/queue")
def api_queue():
    # Simple test queue: random 0..7 so you can see UI change
    # Replace later with a real queue length from your worker system.
    return jsonify({
        "ok": True,
        "queries_in_line": random.randint(0, 7),
        "server_time": time.time()
    }), 200
# ---------------------------------------------------------------------------
# Local dev runner (gunicorn ignores this)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # For local dev only. In production you are using gunicorn via systemd.
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=True)

