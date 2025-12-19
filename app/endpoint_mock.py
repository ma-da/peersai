import json
import random
import time
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
from datetime import datetime

HOST = "0.0.0.0"
PORT = 8080

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

RESPONSES = [
    "This is a mock response.",
    "Here is a simulated answer.",
    "Mock model output generated successfully.",
    "Dummy inference completed.",
    "Synthetic response returned.",
]

class MockInferenceHandler(BaseHTTPRequestHandler):
    def _send_json(self, status_code, payload):
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def do_POST(self):
        start_time = time.perf_counter()

        content_length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_length)

        try:
            body = json.loads(raw_body)
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Invalid JSON"})
            return

        prompt = body.get("inputs", "")
        max_new_tokens = body.get("parameters", {}).get("max_new_tokens", 250)

        logging.info(f"received prompt (max_tokens={max_new_tokens}): {prompt!r}")

        # Simulated processing time
        simulated_latency = random.uniform(0.05, 0.25)
        time.sleep(simulated_latency)

        variation = random.choice(RESPONSES)
        suffix = random.randint(100, 999)

        response_text = (
            f"{variation} | "
            f"prompt_len={len(prompt)} | "
            f"run_id={suffix}"
        )

        elapsed = time.perf_counter() - start_time
        logging.info(f"responding in {elapsed:.4f}s")

        self._send_json(
            200,
            {
                "generated_text": response_text,
                "created_at": datetime.utcnow().isoformat(),
            },
        )

    def log_message(self, format, *args):
        # Silence default HTTP logging
        return


def main():
    server = HTTPServer((HOST, PORT), MockInferenceHandler)
    logging.info(f"mock inference server running on http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("shutting down mock server")
        server.server_close()


if __name__ == "__main__":
    main()
