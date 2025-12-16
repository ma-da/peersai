import logging

from flask import Flask, render_template_string, jsonify, request
import asyncio
import logging_config
import rag_controller
import os

# global retrieval state
retrieval_state = None

app_logger = logging_config.setup_logging(logging.INFO)

# main app
app = Flask(__name__)

# TEST CODE REMOVE LATER
#import http.client
#print("HTTP debuglevel:", http.client.HTTPConnection.debuglevel)
print("WERKZEUG_RUN_MAIN =", os.environ.get("WERKZEUG_RUN_MAIN"))
print("FLASK_DEBUG =", os.environ.get("FLASK_DEBUG"))


def init_state():
    global retrieval_state
    app_logger.info("init_state called")
    from rag_controller import boot
    retrieval_state = boot()
    if retrieval_state is None:
        app_logger.error("boot() did not return valid retrieval_state")
    app_logger.info("init_state done")

# Call once to init the state
#if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
#    init_state()
init_state()

@app.route("/")
def index():
    return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Seeds of Truth</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      background: #111;
      color: #eee;
      display: flex;
      align-items: center;
      justify-content: center;
      height: 100vh;
      flex-direction: column;
      text-align: center;
    }
    h1 { font-size: 2em; margin-bottom: 0.5em; }
    p { color: #aaa; }
  </style>
</head>
<body>
  <h1>🌱 Seeds of Truth</h1>
  <p>This site is under active development.</p>
</body>
</html>
""")

@app.route("/api/chat", methods=["POST","GET"])
def on_chat():
    #payload = request.get_json(silent=True)
    #if payload is None:
    #    return jsonify({
    #        "ok": False,
    #        "error": "Invalid or missing JSON body"
    #    }), 400

    return jsonify({
        "ok": True,
        "status": "chat was successful",
    }), 200

#@app.route("/api/search", methods=["POST","GET"])
#def on_search():
#    #payload = request.get_json(silent=True)
#    #if payload is None:
#    #    return jsonify({
#    #        "ok": False,
#    #        "error": "Invalid or missing JSON body"
#    #    }), 400
#
#    return jsonify({
#        "ok": True,
#        "status": "search was successful",
#    }), 200


@app.route("/api/search", methods=["POST"])
def on_search():
    global retrieval_state, app_logger
    app_logger.info("search msg received")

    # ---------- Parse JSON ----------
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({
            "ok": False,
            "error": "Invalid or missing JSON body"
        }), 400

    # ---------- Validate query ----------
    query = payload.get("query")
    if not isinstance(query, str) or not query.strip():
        app_logger.warn("Search request query was invalid")
        return jsonify({
            "ok": False,
            "error": "Field 'query' must be a non-empty string"
        }), 400

    # ---------- Validate max_n ----------
    max_n = payload.get("max_n", 20)
    if not isinstance(max_n, int):
        app_logger.warn("Search request max_n was invalid type")
        return jsonify({
            "ok": False,
            "error": "Field 'max_n' must be an integer"
        }), 400

    app_logger.info(
        "Search request",
        extra={
            "query": query,
            "len": len(query),
            "max_n": max_n,
            "remote_addr": request.remote_addr,
        }
    )

    if max_n <= 0 or max_n > 200:
        app_logger.warn("Search request max_n was invalid: ", max_n)
        return jsonify({
            "ok": False,
            "error": "Field 'max_n' must be between 1 and 200"
        }), 400

    # ---------- Ensure system is ready ----------
    if retrieval_state is None:
        app_logger.error("retrieval_state was not initialized")
        return jsonify({
            "ok": False,
            "error": "Search system not initialized"
        }), 503

    # ---------- Call async search ----------
    try:
        # Flask is sync → run async function explicitly
        results = asyncio.run(
            rag_controller.search_references(
                retrieval_state,
                query,
                top_k=max_n,
            )
        )
    except RuntimeError as e:
        # asyncio.run() called inside an existing event loop
        # (this happens under some servers / debuggers)
        try:
            loop = asyncio.get_event_loop()
            results = loop.run_until_complete(
                rag_controller.search_references(
                    retrieval_state,
                    query,
                    top_k=max_n,
                )
            )
        except Exception as inner:
            app_logger.error("Internal async execution error occurred")
            return jsonify({
                "ok": False,
                "error": "Internal async execution error",
                "detail": str(inner)
            }), 500
    except Exception as e:
        app_logger.error("Internal async execution error occurred")
        return jsonify({
            "ok": False,
            "error": "Search failed",
            "detail": str(e)
        }), 500

    # ---------- Success ----------
    app_logger.info("Search request completed.")
    return jsonify({
        "ok": True,
        "query": query,
        "num_results": results.get("num_results", 0),
        "results": results.get("results", []),
    }), 200


@app.route("/api/feedback", methods=["POST","GET"])
def on_feedback():
    #payload = request.get_json(silent=True)
    #if payload is None:
    #    return jsonify({
    #        "ok": False,
    #        "error": "Invalid or missing JSON body"
    #    }), 400

    return jsonify({
        "ok": True,
        "status": "feedback was successful",
    }), 200

@app.route("/api/status", methods=["POST","GET"])
def on_status():
    #payload = request.get_json(silent=True)
    #if payload is None:
    #    return jsonify({
    #        "ok": False,
    #        "error": "Invalid or missing JSON body"
    #    }), 400

    return jsonify({
        "ok": True,
        "status": "status was successful",
    }), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
