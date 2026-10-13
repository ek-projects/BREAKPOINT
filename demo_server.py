#!/usr/bin/env python3
"""Zero-dependency BREAKPOINT demo web server."""
from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from demo_runtime import run_demo
from nebius_reasoner import NemotronClient

ROOT = Path(__file__).resolve().parent
INDEX = (ROOT / "web" / "index.html").read_bytes()
CLIENT = NemotronClient()


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, payload, content_type="application/json; charset=utf-8"):
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        if self.path == "/":
            return self._send(200, INDEX, "text/html; charset=utf-8")
        if self.path == "/api/status":
            return self._send(200, {
                "project": "BREAKPOINT",
                "nebius_configured": CLIENT.configured,
                "model": CLIENT.model,
                "mode": "Nebius Token Factory + deterministic planner" if CLIENT.configured else "deterministic mode (set NEBIUS_API_KEY for Token Factory)",
                "nemotron_max_tokens": CLIENT.max_tokens,
                "nemotron_thinking": CLIENT.enable_thinking,
            })
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/api/diagnose":
            return self._send(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(length) or b"{}")
            result = run_demo(str(data.get("fault", "database")), CLIENT)
            return self._send(200, result)
        except Exception as exc:
            return self._send(400, {"error": str(exc)})

    def log_message(self, *_):
        pass


def main():
    host = os.getenv("BREAKPOINT_HOST", "127.0.0.1")
    port = int(os.getenv("PORT", os.getenv("BREAKPOINT_PORT", "8878")))
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"BREAKPOINT demo: http://{host}:{port}")
    print("Nebius configured:", CLIENT.configured, "model:", CLIENT.model)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
