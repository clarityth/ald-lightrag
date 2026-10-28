"""Serve the local ProcessLens question UI on loopback only."""

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from lightrag_app.backend import ask

ADDRESS = ("127.0.0.1", 8824)
ASK_LOCK = Lock()
HOSTS = {"127.0.0.1:8824", "localhost:8824"}


def load_examples():
    path = ROOT / "results/examples.json"
    if not path.exists():
        path = ROOT / "demo/example.json"
    examples = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    return examples if isinstance(examples, list) else []


class Handler(BaseHTTPRequestHandler):
    def reply(self, value, status=200, content_type="application/json; charset=utf-8"):
        body = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'; base-uri 'none'; form-action 'self'")
        self.end_headers()
        self.wfile.write(body)

    def allowed_origin(self):
        host = self.headers.get("Host", "")
        origin = self.headers.get("Origin")
        return host in HOSTS and (not origin or origin == f"http://{host}")

    def do_GET(self):
        if not self.allowed_origin():
            return self.reply({"error": "이 로컬 화면에서만 접근할 수 있습니다."}, 403)
        if self.path == "/":
            return self.reply((ROOT / "index.html").read_bytes(), content_type="text/html; charset=utf-8")
        if self.path in {"/comparison.js", "/workspace.js"}:
            return self.reply((ROOT / self.path.lstrip("/")).read_bytes(), content_type="text/javascript; charset=utf-8")
        if self.path == "/api/examples":
            try:
                examples = load_examples()
            except (OSError, ValueError):
                return self.reply({"error": "저장된 결과를 읽을 수 없습니다."}, 500)
            if not isinstance(examples, list):
                return self.reply({"error": "저장된 결과를 읽을 수 없습니다."}, 500)
            return self.reply([{
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
                "chunks": row.get("chunks", []),
                "documents": row.get("documents", []),
                "experiments": row.get("experiments", []),
                "measurements": row.get("measurements", []),
                "paths": row.get("paths", []),
            } for row in examples if isinstance(row, dict)])
        return self.reply({"error": "찾을 수 없습니다."}, 404)

    def do_POST(self):
        if self.path != "/api/ask":
            return self.reply({"error": "찾을 수 없습니다."}, 404)
        if not self.allowed_origin():
            return self.reply({"error": "이 로컬 화면에서만 질문할 수 있습니다."}, 403)
        if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
            return self.reply({"error": "JSON 형식으로 질문해 주세요."}, 415)
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 12000:
                raise ValueError
            payload = json.loads(self.rfile.read(size))
            question = payload.get("question", "")
            if not isinstance(question, str) or not 1 <= len(question.strip()) <= 2000:
                raise ValueError
        except (ValueError, TypeError, AttributeError):
            return self.reply({"error": "질문을 1~2000자로 입력해 주세요."}, 400)
        if not ASK_LOCK.acquire(blocking=False):
            return self.reply({"error": "다른 질문을 처리하고 있습니다. 잠시 후 다시 시도해 주세요."}, 409)
        try:
            return self.reply(ask(question.strip()))
        except Exception:
            return self.reply({"error": "답변 생성에 실패했습니다. 서버 상태를 확인한 뒤 다시 시도해 주세요."}, 502)
        finally:
            ASK_LOCK.release()

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print("ProcessLens: http://127.0.0.1:8824/", flush=True)
    ThreadingHTTPServer(ADDRESS, Handler).serve_forever()
