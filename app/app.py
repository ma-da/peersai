from flask import Flask, render_template_string, jsonify, request

app = Flask(__name__)

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

@app.route("/api/search", methods=["POST","GET"])
def on_search():
    #payload = request.get_json(silent=True)
    #if payload is None:
    #    return jsonify({
    #        "ok": False,
    #        "error": "Invalid or missing JSON body"
    #    }), 400

    return jsonify({
        "ok": True,
        "status": "search was successful",
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
    app.run(host="0.0.0.0", port=8000)
