"""
基于标准库 HTTPServer 的本地网页入口。

用法：
    code/C8/.venv/bin/python web_app.py

然后在浏览器打开：
    http://127.0.0.1:7860
"""

from __future__ import annotations

import html
import logging
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import parse_qs

from main import RecipeRAGSystem


BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "outputs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "web_ui.log"


def configure_file_logging() -> None:
    """把日志统一写到 UTF-8 文件，避免刷控制台。"""
    root_logger = logging.getLogger()
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    root_logger.handlers = [file_handler]
    root_logger.setLevel(logging.INFO)


configure_file_logging()


class LazyRAGSystem:
    """延迟初始化的 RAG 单例。"""

    def __init__(self):
        self._lock = Lock()
        self._system: RecipeRAGSystem | None = None
        self._startup_status: list[str] = []
        self._startup_error: str | None = None

    def get(self) -> tuple[RecipeRAGSystem | None, list[str], str | None]:
        with self._lock:
            if self._system is not None or self._startup_error is not None:
                return self._system, list(self._startup_status), self._startup_error

            status: list[str] = []
            try:
                system = RecipeRAGSystem()
                system.initialize_system(verbose=False, status_callback=status.append)
                system.build_knowledge_base(verbose=False, status_callback=status.append)
                self._system = system
                self._startup_status = status
            except Exception:
                self._startup_status = status
                self._startup_error = traceback.format_exc()
                logging.exception("网页 RAG 初始化失败")

            return self._system, list(self._startup_status), self._startup_error


RAG_SYSTEM = LazyRAGSystem()


def render_page(question: str = "", answer: str = "", status_lines: list[str] | None = None, error: str = "") -> str:
    status_text = "\n".join(status_lines or [])
    escaped_question = html.escape(question)
    escaped_answer = html.escape(answer)
    escaped_status = html.escape(status_text)
    escaped_error = html.escape(error)

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>尝尝咸淡 RAG 网页版</title>
  <style>
    :root {{
      --bg: #f5efe6;
      --card: #fffdf9;
      --ink: #2c241d;
      --muted: #7a6a5d;
      --accent: #b85c38;
      --accent-2: #f0c987;
      --border: #e6d6c6;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top right, rgba(240, 201, 135, 0.45), transparent 28%),
        linear-gradient(180deg, #fbf6ef 0%, var(--bg) 100%);
    }}
    .wrap {{
      max-width: 980px;
      margin: 0 auto;
      padding: 40px 20px 64px;
    }}
    .hero {{
      margin-bottom: 22px;
    }}
    .eyebrow {{
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(184, 92, 56, 0.1);
      color: var(--accent);
      font-size: 13px;
      font-weight: 700;
      letter-spacing: 0.04em;
    }}
    h1 {{
      margin: 14px 0 8px;
      font-size: clamp(30px, 6vw, 52px);
      line-height: 1.05;
    }}
    .sub {{
      margin: 0;
      color: var(--muted);
      font-size: 16px;
      line-height: 1.7;
      max-width: 720px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1.15fr 0.85fr;
      gap: 18px;
      align-items: start;
    }}
    .card {{
      background: rgba(255, 253, 249, 0.88);
      backdrop-filter: blur(8px);
      border: 1px solid var(--border);
      border-radius: 24px;
      box-shadow: 0 18px 40px rgba(78, 53, 31, 0.08);
      padding: 22px;
    }}
    label {{
      display: block;
      font-size: 14px;
      font-weight: 700;
      margin-bottom: 10px;
    }}
    textarea {{
      width: 100%;
      min-height: 140px;
      resize: vertical;
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 16px;
      font: inherit;
      background: #fff;
      color: var(--ink);
    }}
    button {{
      margin-top: 14px;
      border: 0;
      border-radius: 14px;
      padding: 12px 18px;
      font: inherit;
      font-weight: 700;
      color: white;
      background: linear-gradient(135deg, var(--accent), #d97742);
      cursor: pointer;
    }}
    button:hover {{
      filter: brightness(1.03);
    }}
    .section-title {{
      margin: 0 0 10px;
      font-size: 15px;
      font-weight: 800;
      color: var(--accent);
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      line-height: 1.65;
      font-family: ui-monospace, "SFMono-Regular", Menlo, monospace;
      font-size: 13px;
      color: #3d3229;
    }}
    .answer {{
      min-height: 280px;
      background: linear-gradient(180deg, rgba(240, 201, 135, 0.18), rgba(255, 255, 255, 0.75));
    }}
    .status {{
      min-height: 180px;
    }}
    .error {{
      border-color: #d45757;
      background: #fff6f6;
    }}
    .hint {{
      margin-top: 10px;
      font-size: 13px;
      color: var(--muted);
    }}
    @media (max-width: 820px) {{
      .grid {{
        grid-template-columns: 1fr;
      }}
      .wrap {{
        padding-top: 24px;
      }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <div class="eyebrow">LOCAL WEB UI</div>
      <h1>尝尝咸淡 RAG 网页版</h1>
      <p class="sub">不再从控制台读答案。这里直接展示检索状态和最终结果，后台日志会写入 <code>{html.escape(str(LOG_FILE))}</code>。</p>
    </div>

    <div class="grid">
      <div class="card">
        <form method="post" action="/ask">
          <label for="question">输入你的问题</label>
          <textarea id="question" name="question" placeholder="比如：红烧肉怎么做">{escaped_question}</textarea>
          <button type="submit">开始检索</button>
          <div class="hint">建议先试：红烧肉怎么做 / 推荐几个素菜 / 宫保鸡丁需要什么食材</div>
        </form>
      </div>

      <div class="card status">
        <div class="section-title">运行状态</div>
        <pre>{escaped_status or "等待提问..."}</pre>
      </div>
    </div>

    <div class="card answer" style="margin-top: 18px;">
      <div class="section-title">回答结果</div>
      <pre>{escaped_answer or "提交问题后，这里会显示最终答案。"}</pre>
    </div>

    <div class="card error" style="margin-top: 18px; display: {'block' if escaped_error else 'none'};">
      <div class="section-title">错误信息</div>
      <pre>{escaped_error}</pre>
    </div>
  </div>
</body>
</html>"""


class RAGRequestHandler(BaseHTTPRequestHandler):
    def _send_html(self, html_text: str, status: int = HTTPStatus.OK) -> None:
        payload = html_text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        system, startup_status, startup_error = RAG_SYSTEM.get()
        page = render_page(
            status_lines=startup_status if system else startup_status + ["初始化未完成"],
            error=startup_error or "",
        )
        self._send_html(page, HTTPStatus.OK if system else HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:
        if self.path != "/ask":
            self._send_html(render_page(error="未知请求路径"), HTTPStatus.NOT_FOUND)
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length).decode("utf-8", errors="replace")
        form = parse_qs(raw_body)
        question = form.get("question", [""])[0].strip()

        system, startup_status, startup_error = RAG_SYSTEM.get()
        status_lines = list(startup_status)

        if startup_error or system is None:
            page = render_page(
                question=question,
                status_lines=status_lines,
                error=startup_error or "RAG 系统初始化失败",
            )
            self._send_html(page, HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        if not question:
            page = render_page(
                question=question,
                status_lines=status_lines + ["请输入问题后再提交。"],
                error="问题不能为空。",
            )
            self._send_html(page, HTTPStatus.BAD_REQUEST)
            return

        request_status: list[str] = []
        try:
            answer = system.ask_question(
                question,
                stream=False,
                verbose=False,
                status_callback=request_status.append,
            )
            page = render_page(question=question, answer=answer, status_lines=request_status)
            self._send_html(page)
        except Exception:
            err = traceback.format_exc()
            logging.exception("网页问答失败")
            page = render_page(
                question=question,
                status_lines=request_status,
                error=err,
            )
            self._send_html(page, HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, format: str, *args) -> None:
        logging.info("HTTP %s - %s", self.address_string(), format % args)


def main() -> None:
    host = "127.0.0.1"
    port = 7860
    server = ThreadingHTTPServer((host, port), RAGRequestHandler)
    print(f"Web UI running at http://{host}:{port}")
    print(f"Logs will be written to {LOG_FILE}")
    server.serve_forever()


if __name__ == "__main__":
    main()
  


  