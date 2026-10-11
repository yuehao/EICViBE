"""A stand-in for the model servers (Ollama, OpenAI, Anthropic): a fixed reply, and what they were asked."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

KEY = "sk-test-key-1234"


class FakeOllama:
    def __init__(
        self, models=("fake:1b",), reply="The tracking agrees with the model.", delay=0.0, error=None
    ):
        self.models, self.reply, self.delay, self.error = list(models), reply, delay, error
        self.requests: list[dict] = []
        self.headers: list[dict] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _authorized(self) -> bool:
                if (
                    self.headers.get("x-api-key") == KEY
                    or self.headers.get("Authorization") == f"Bearer {KEY}"
                ):
                    return True
                self._json({"error": {"message": "invalid x-api-key"}}, status=401)
                return False

            def do_GET(self):
                if self.path == "/api/tags":
                    self._json({"models": [{"name": m} for m in fake.models]})
                elif self.path.startswith("/v1/models"):
                    if self._authorized():
                        self._json({"data": [{"id": m} for m in fake.models]})
                else:
                    self.send_error(404)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.requests.append(body)
                fake.headers.append(dict(self.headers))
                if self.path == "/api/chat":
                    self._ollama(body)
                elif self.path == "/v1/chat/completions" and self._authorized():
                    self._events([{"choices": [{"delta": {"content": w}}]} for w in self._words()], done=True)
                elif self.path == "/v1/messages" and self._authorized():
                    deltas = [
                        {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {"type": "text_delta", "text": w},
                        }
                        for w in self._words()
                    ]
                    self._events([{"type": "message_start"}, *deltas, {"type": "message_stop"}])

            def _words(self):
                words = fake.reply.split(" ")
                return [w + (" " if i < len(words) - 1 else "") for i, w in enumerate(words)]

            def _ollama(self, body):
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.end_headers()
                if fake.error:
                    self.wfile.write(json.dumps({"error": fake.error}).encode() + b"\n")
                    return
                if not body.get("stream"):
                    self.wfile.write(json.dumps({"message": {"content": fake.reply}, "done": True}).encode())
                    return
                for word in fake.reply.split(" "):
                    time.sleep(fake.delay)
                    piece = {"message": {"content": word + " "}, "done": False}
                    self.wfile.write(json.dumps(piece).encode() + b"\n")
                    self.wfile.flush()
                self.wfile.write(json.dumps({"message": {"content": ""}, "done": True}).encode() + b"\n")

            def _events(self, events, done=False):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for event in events:
                    time.sleep(fake.delay)
                    kind = event.get("type")
                    head = f"event: {kind}\n" if kind else ""
                    self.wfile.write(f"{head}data: {json.dumps(event)}\n\n".encode())
                    self.wfile.flush()
                if done:
                    self.wfile.write(b"data: [DONE]\n\n")

            def _json(self, value, status=200):
                data = json.dumps(value).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
