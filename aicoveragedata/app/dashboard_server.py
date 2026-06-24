import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys

APP_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = APP_DIR.parent
ROOT_DIR = PACKAGE_DIR.parent
SITE_DIR = PACKAGE_DIR / "site"
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))

sys.path.append(str(ROOT_DIR))
from aicoveragedata.agent.core.agent import answer_question
from aicoveragedata.agent.core.config import AgentConfig
from aicoveragedata.agent.memory.history import clear_session, get_session_messages


class DashboardRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(SITE_DIR), **kwargs)

    def do_HEAD(self):
        if self.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        return super().do_HEAD()

    def do_GET(self):
        if self.path == "/":
            self.path = "/index.html"
        if self.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        return super().do_GET()

    def send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/api/agent":
            self.send_error(404, "Unknown API route")
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_json({"error": "Invalid content length."}, status=400)
            return

        if content_length > 20000:
            self.send_json({"error": "Question payload is too large."}, status=413)
            return

        try:
            raw_body = self.rfile.read(content_length).decode("utf-8")
            payload = json.loads(raw_body or "{}")
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.send_json({"error": "Request body must be valid JSON."}, status=400)
            return

        session_id = str(payload.get("session_id", "dashboard")).strip() or "dashboard"

        if payload.get("reset"):
            config = AgentConfig.from_env()
            clear_session(session_id, config.chat_history_path)
            self.send_json({"answer": "Chat history cleared.", "session_id": session_id})
            return

        if payload.get("history"):
            config = AgentConfig.from_env()
            messages = get_session_messages(
                session_id,
                config.chat_history_path,
                max_messages=config.max_history_messages,
            )
            self.send_json({"messages": messages, "session_id": session_id})
            return

        try:
            question = str(payload.get("question", "")).strip()
            if not question:
                self.send_json({"error": "Question is required."}, status=400)
                return
            answer, _context = answer_question(question, session_id=session_id)
        except Exception as error:
            self.send_json({"error": str(error)}, status=500)
            return

        self.send_json({"answer": answer, "session_id": session_id})


def main():
    print(f"Dashboard server running at http://{HOST}:{PORT}/")
    print(f"Project root: {ROOT_DIR}")
    print(f"Serving: {SITE_DIR}")
    ThreadingHTTPServer((HOST, PORT), DashboardRequestHandler).serve_forever()


if __name__ == "__main__":
    main()
