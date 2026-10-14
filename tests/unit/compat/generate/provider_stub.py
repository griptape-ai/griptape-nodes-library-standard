"""OpenAI-compatible chat and Ollama model-list endpoints, so main can run providers without keys."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODELS = ["gpt-4.1-mini", "gpt-4.1", "gpt-4o"]


def reply_for(body: dict) -> str:
    last = ""
    for message in body.get("messages", []):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, list):
                content = " ".join(part.get("text", "") for part in content if isinstance(part, dict))
            last = content or ""
    return f"STUB[{body.get('model')}] {last.strip()[:60]}"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # noqa: ANN002
        pass

    def _json(self, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path.rstrip("/").endswith("/api/tags"):
            self._json(
                {
                    "models": [
                        {"name": "llama3.2:latest", "model": "llama3.2:latest", "size": 1, "digest": "d", "details": {}}
                    ]
                }
            )
        elif self.path.rstrip("/").endswith("/models"):
            self._json(
                {
                    "object": "list",
                    "data": [{"id": m, "object": "model", "created": 0, "owned_by": "stub"} for m in MODELS],
                }
            )
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self.send_error(404)
            return
        text = reply_for(body)
        base = {"id": "chatcmpl-stub", "created": int(time.time()), "model": body.get("model", "stub")}
        usage = {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}
        if not body.get("stream"):
            self._json(
                {
                    **base,
                    "object": "chat.completion",
                    "usage": usage,
                    "choices": [
                        {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}
                    ],
                }
            )
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        chunks = [
            {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            {"choices": [], "usage": usage},
        ]
        for chunk in chunks:
            self.wfile.write(f"data: {json.dumps({**base, 'object': 'chat.completion.chunk', **chunk})}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")


def serve(port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
