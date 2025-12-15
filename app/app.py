from flask import Flask, render_template_string

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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
