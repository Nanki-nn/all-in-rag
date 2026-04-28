"""
C9 图RAG本地网页入口。

用法：
    .venv/bin/python web_app.py

然后在浏览器打开：
    http://127.0.0.1:7861
"""

from __future__ import annotations

import html
import io
import json
import logging
import traceback
import uuid
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread
from time import time
from urllib.parse import parse_qs, urlparse


BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "outputs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "web_ui_c9.log"


def configure_file_logging() -> None:
    root_logger = logging.getLogger()
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    root_logger.handlers = [file_handler]
    root_logger.setLevel(logging.INFO)


configure_file_logging()

from main import AdvancedGraphRAGSystem  # noqa: E402


class LazyGraphRAGSystem:
    """延迟初始化 C9 图RAG 系统，避免首页打开时卡住。"""

    def __init__(self):
        self._lock = Lock()
        self._system: AdvancedGraphRAGSystem | None = None
        self._startup_status: list[str] = []
        self._startup_error: str | None = None

    def get(self, task_id: str | None = None) -> tuple[AdvancedGraphRAGSystem | None, list[str], str | None]:
        with self._lock:
            if self._system is not None or self._startup_error is not None:
                return self._system, list(self._startup_status), self._startup_error

            capture = ObservableBuffer(task_id) if task_id else io.StringIO()
            try:
                with redirect_stdout(capture):
                    system = AdvancedGraphRAGSystem()
                    system.initialize_system()
                    system.build_knowledge_base()
                if isinstance(capture, ObservableBuffer):
                    capture.flush_pending()
                self._system = system
                self._startup_status = capture.getvalue().splitlines()
            except Exception:
                if isinstance(capture, ObservableBuffer):
                    capture.flush_pending()
                self._startup_status = capture.getvalue().splitlines()
                self._startup_error = traceback.format_exc()
                logging.exception("C9 网页版初始化失败")

            return self._system, list(self._startup_status), self._startup_error


GRAPH_RAG = LazyGraphRAGSystem()


@dataclass
class QueryTask:
    question: str
    status: str = "running"
    events: list[dict] = field(default_factory=list)
    answer: str = ""
    error: str = ""
    created_at: float = field(default_factory=time)
    updated_at: float = field(default_factory=time)


class TaskStore:
    def __init__(self):
        self._lock = Lock()
        self._tasks: dict[str, QueryTask] = {}

    def create(self, question: str) -> str:
        task_id = uuid.uuid4().hex
        with self._lock:
            self._tasks[task_id] = QueryTask(question=question)
        return task_id

    def add_event(self, task_id: str, stage: str, message: str, detail: str = "") -> None:
        with self._lock:
            task = self._tasks[task_id]
            task.events.append({
                "time": time(),
                "stage": stage,
                "message": message,
                "detail": detail,
            })
            task.updated_at = time()

    def finish(self, task_id: str, answer: str) -> None:
        with self._lock:
            task = self._tasks[task_id]
            task.status = "done"
            task.answer = answer
            task.updated_at = time()

    def fail(self, task_id: str, error: str) -> None:
        with self._lock:
            task = self._tasks[task_id]
            task.status = "error"
            task.error = error
            task.updated_at = time()

    def snapshot(self, task_id: str) -> dict | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            return {
                "question": task.question,
                "status": task.status,
                "events": list(task.events),
                "answer": task.answer,
                "error": task.error,
            }


TASKS = TaskStore()


class ObservableBuffer(io.StringIO):
    """把核心流程print出来的状态同步记录成前端事件。"""

    def __init__(self, task_id: str):
        super().__init__()
        self.task_id = task_id
        self._pending = ""

    def write(self, text: str) -> int:
        written = super().write(text)
        self._pending += text
        while "\n" in self._pending:
            line, self._pending = self._pending.split("\n", 1)
            line = line.strip()
            if line:
                stage = infer_stage(line)
                TASKS.add_event(self.task_id, stage, line)
        return written

    def flush_pending(self) -> None:
        line = self._pending.strip()
        if line:
            TASKS.add_event(self.task_id, infer_stage(line), line)
        self._pending = ""


def infer_stage(line: str) -> str:
    if "初始化" in line or "知识库" in line:
        return "初始化"
    if "路由" in line or "策略" in line or "复杂度" in line or "关系密集度" in line:
        return "路由分析"
    if "检索" in line or "找到" in line or "文档" in line:
        return "检索召回"
    if "生成" in line or "回答" in line:
        return "答案生成"
    if "耗时" in line or "完成" in line:
        return "完成"
    return "运行"


def run_query_task(task_id: str) -> None:
    question = TASKS.snapshot(task_id)["question"]
    TASKS.add_event(task_id, "接收请求", f"收到问题: {question}")

    try:
        TASKS.add_event(task_id, "初始化", "准备图RAG系统")
        system, startup_status, startup_error = GRAPH_RAG.get(task_id=task_id)
        for line in startup_status:
            TASKS.add_event(task_id, infer_stage(line), line)

        if startup_error or system is None:
            raise RuntimeError(startup_error or "C9 图RAG初始化失败")

        buffer = ObservableBuffer(task_id)
        with redirect_stdout(buffer):
            answer, analysis = system.ask_question_with_routing(question, stream=False, explain_routing=False)
        buffer.flush_pending()

        if analysis is not None:
            TASKS.add_event(task_id, "路由分析", f"使用策略: {analysis.recommended_strategy.value}")
            TASKS.add_event(task_id, "路由分析", f"复杂度: {analysis.query_complexity:.2f}")
            TASKS.add_event(task_id, "路由分析", f"关系密集度: {analysis.relationship_intensity:.2f}")

        TASKS.add_event(task_id, "完成", "回答生成完成")
        TASKS.finish(task_id, answer)
    except Exception:
        err = traceback.format_exc()
        logging.exception("C9 网页任务失败")
        TASKS.fail(task_id, err)


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
  <title>C9 图RAG 烹饪助手</title>
  <style>
    :root {{
      --bg: #f7faf7;
      --panel: #ffffff;
      --ink: #17211d;
      --muted: #66736e;
      --line: #dce7e2;
      --green: #1f8a70;
      --blue: #2d6cdf;
      --red: #d85a47;
      --soft: #eef7f3;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
      color: var(--ink);
      background:
        linear-gradient(135deg, rgba(31, 138, 112, 0.08), transparent 32%),
        linear-gradient(180deg, #fbfdfc 0%, var(--bg) 100%);
    }}
    .wrap {{
      max-width: 1080px;
      margin: 0 auto;
      padding: 36px 20px 56px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 34px;
      line-height: 1.2;
      letter-spacing: 0;
    }}
    .sub {{
      margin: 0 0 24px;
      color: var(--muted);
      line-height: 1.7;
    }}
    .grid {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      gap: 18px;
      align-items: start;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 16px 36px rgba(34, 52, 46, 0.08);
      padding: 20px;
    }}
    label {{
      display: block;
      font-weight: 700;
      margin-bottom: 10px;
    }}
    textarea {{
      width: 100%;
      min-height: 138px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      color: var(--ink);
      font: inherit;
      line-height: 1.6;
      background: #fff;
    }}
    .actions {{
      display: flex;
      align-items: center;
      gap: 12px;
      margin-top: 12px;
      flex-wrap: wrap;
    }}
    button {{
      border: 0;
      border-radius: 8px;
      padding: 11px 16px;
      font: inherit;
      font-weight: 700;
      color: white;
      background: linear-gradient(135deg, var(--green), var(--blue));
      cursor: pointer;
    }}
    button:disabled {{
      opacity: 0.72;
      cursor: wait;
    }}
    .hint {{
      color: var(--muted);
      font-size: 13px;
    }}
    .title {{
      margin: 0 0 10px;
      color: var(--green);
      font-size: 15px;
      font-weight: 800;
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: ui-monospace, "SFMono-Regular", Menlo, monospace;
      font-size: 13px;
      line-height: 1.65;
      color: #22352f;
    }}
    .answer {{
      margin-top: 18px;
      background: linear-gradient(180deg, var(--soft), #fff);
      min-height: 300px;
    }}
    .error {{
      margin-top: 18px;
      border-color: #efb0a5;
      background: #fff7f5;
    }}
    .badge {{
      display: inline-block;
      margin-bottom: 12px;
      padding: 5px 9px;
      border-radius: 999px;
      background: rgba(31, 138, 112, 0.1);
      color: var(--green);
      font-size: 12px;
      font-weight: 800;
    }}
    .timeline {{
      display: grid;
      gap: 10px;
      max-height: 420px;
      overflow: auto;
    }}
    .event {{
      display: grid;
      grid-template-columns: 92px minmax(0, 1fr);
      gap: 10px;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfefd;
    }}
    .event-stage {{
      color: var(--blue);
      font-size: 12px;
      font-weight: 800;
    }}
    .event-message {{
      color: var(--ink);
      font-size: 13px;
      line-height: 1.55;
    }}
    @media (max-width: 860px) {{
      .grid {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 28px; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <span class="badge">C9 Graph RAG</span>
    <h1>图RAG 烹饪助手</h1>
    <p class="sub">网页端复用 C9 主流程：智能路由、图检索、混合检索、本地 Milvus Lite 向量索引和 DeepSeek 生成。</p>

    <div class="grid">
      <div class="panel">
        <form id="ask-form">
          <label for="question">输入问题</label>
          <textarea id="question" name="question" placeholder="比如：红烧肉怎么做">{escaped_question}</textarea>
          <div class="actions">
            <button type="submit" id="submit-btn">开始检索</button>
            <span class="hint" id="loading-text">首次请求会初始化图RAG，可能需要几十秒。</span>
          </div>
        </form>
      </div>

      <div class="panel">
        <div class="title">运行状态</div>
        <div class="timeline" id="timeline">
          <div class="event">
            <div class="event-stage">等待</div>
            <div class="event-message">{escaped_status or "等待提问..."}</div>
          </div>
        </div>
      </div>
    </div>

    <div class="panel answer">
      <div class="title">回答结果</div>
      <pre id="answer-box">{escaped_answer or "提交问题后，这里会显示最终答案。"}</pre>
    </div>

    <div class="panel error" style="display: {'block' if escaped_error else 'none'};">
      <div class="title" style="color: var(--red);">错误信息</div>
      <pre id="error-box">{escaped_error}</pre>
    </div>
  </div>

  <script>
    const form = document.getElementById("ask-form");
    const button = document.getElementById("submit-btn");
    const loadingText = document.getElementById("loading-text");
    const timeline = document.getElementById("timeline");
    const answerBox = document.getElementById("answer-box");
    const errorBox = document.getElementById("error-box");

    function escapeHtml(value) {{
      return value
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }}

    function renderEvents(events) {{
      if (!events.length) {{
        timeline.innerHTML = '<div class="event"><div class="event-stage">等待</div><div class="event-message">准备中...</div></div>';
        return;
      }}
      timeline.innerHTML = events.map((event) => `
        <div class="event">
          <div class="event-stage">${{escapeHtml(event.stage)}}</div>
          <div class="event-message">${{escapeHtml(event.message)}}</div>
        </div>
      `).join("");
      timeline.scrollTop = timeline.scrollHeight;
    }}

    async function pollTask(taskId) {{
      const response = await fetch(`/status?id=${{encodeURIComponent(taskId)}}`);
      const data = await response.json();
      renderEvents(data.events || []);
      if (data.status === "done") {{
        answerBox.textContent = data.answer || "没有生成答案。";
        button.disabled = false;
        button.textContent = "开始检索";
        loadingText.textContent = "已完成。";
        return;
      }}
      if (data.status === "error") {{
        errorBox.textContent = data.error || "未知错误";
        button.disabled = false;
        button.textContent = "开始检索";
        loadingText.textContent = "处理失败，请查看错误信息。";
        return;
      }}
      setTimeout(() => pollTask(taskId), 900);
    }}

    form.addEventListener("submit", async (event) => {{
      event.preventDefault();
      button.disabled = true;
      button.textContent = "检索中...";
      loadingText.textContent = "正在初始化或生成回答，请稍等。";
      answerBox.textContent = "正在生成...";
      errorBox.textContent = "";
      renderEvents([]);

      const formData = new FormData(form);
      const response = await fetch("/ask", {{
        method: "POST",
        body: new URLSearchParams(formData),
      }});
      const data = await response.json();
      if (!response.ok) {{
        errorBox.textContent = data.error || "请求失败";
        button.disabled = false;
        button.textContent = "开始检索";
        return;
      }}
      pollTask(data.task_id);
    }});
  </script>
</body>
</html>"""


class GraphRAGRequestHandler(BaseHTTPRequestHandler):
    def _send_html(self, html_text: str, status: int = HTTPStatus.OK) -> None:
        payload = html_text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, data: dict, status: int = HTTPStatus.OK) -> None:
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/status":
            task_id = parse_qs(parsed.query).get("id", [""])[0]
            snapshot = TASKS.snapshot(task_id)
            if snapshot is None:
                self._send_json({"error": "任务不存在"}, HTTPStatus.NOT_FOUND)
                return
            self._send_json(snapshot)
            return

        self._send_html(render_page())

    def do_POST(self) -> None:
        if self.path != "/ask":
            self._send_json({"error": "未知请求路径"}, HTTPStatus.NOT_FOUND)
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length).decode("utf-8", errors="replace")
        question = parse_qs(raw_body).get("question", [""])[0].strip()

        if not question:
            self._send_json({"error": "问题不能为空。"}, HTTPStatus.BAD_REQUEST)
            return

        task_id = TASKS.create(question)
        Thread(target=run_query_task, args=(task_id,), daemon=True).start()
        self._send_json({"task_id": task_id})

    def log_message(self, format: str, *args) -> None:
        logging.info("HTTP %s - %s", self.address_string(), format % args)


def main() -> None:
    host = "127.0.0.1"
    port = 7861
    server = ThreadingHTTPServer((host, port), GraphRAGRequestHandler)
    print(f"C9 Web UI running at http://{host}:{port}")
    print(f"Logs will be written to {LOG_FILE}")
    server.serve_forever()


if __name__ == "__main__":
    main()
