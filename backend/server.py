from __future__ import annotations
import asyncio
import html
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from persona.logger import setup_logging
from persona.logger import get_logger
from persona.runtime import SimulationRuntime
from persona.history_recorder import HistoryRecorder
from persona.run_archive import archive_runtime_run
from scenarios.registry import get_scenario, list_scenarios
from world.serializer import snapshot

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# 全局运行时状态
# ---------------------------------------------------------------------------

# rt 在 lifespan 中初始化，reset 命令会重建它
rt: SimulationRuntime | None = None
# 历史记录器，随 rt 一起重建
_recorder: HistoryRecorder | None = None
# 当前前端运行场景；默认保持旧 Web 入口的小镇场景。
current_scenario_name = "default_town"
# 当前所有已连接的 WebSocket 客户端，广播时遍历
clients: set[WebSocket] = set()
# 仿真控制状态，running=True 时自动步进循环运行
sim_state: dict = {"running": False, "speed": 1.0}
# 自动步进 asyncio Task 引用，pause/reset 时用于取消
_sim_task: asyncio.Task | None = None
# 最近一次 reset 前归档的实验输出，供前端提示摘要和图表链接。
_last_archived_run: dict | None = None


HISTORY_DIR = Path(__file__).parent / "history"
HISTORY_VIEW_FILES = {
    "config_snapshot.json",
    "experiment_summary.md",
    "opinion_trends.svg",
    "effective_pressure_trends.svg",
    "mediator_peak_trends.svg",
    "polarization_report.md",
    "polarization_metrics.csv",
    "polarization_agent_shift.csv",
}


def _simulation_step_limit() -> int:
    if rt is None:
        return 0
    return max(0, int(rt.config.simulation_step_limit))


def _limit_reached() -> bool:
    limit = _simulation_step_limit()
    return bool(rt is not None and limit > 0 and rt.world.time >= limit)


def _status_payload(*, limit_reached: bool | None = None) -> dict:
    reached = _limit_reached() if limit_reached is None else limit_reached
    return {
        "type": "status",
        "running": sim_state["running"],
        "speed": sim_state["speed"],
        "scenario_name": current_scenario_name,
        "scenarios": list_scenarios(),
        "max_ticks": _simulation_step_limit(),
        "limit_reached": reached,
        "archived_run": _last_archived_run,
    }


# ---------------------------------------------------------------------------
# 仿真初始化（与 main.py 保持一致的智能体配置）
# ---------------------------------------------------------------------------

def _build_runtime(scenario_name: str | None = None) -> SimulationRuntime:
    """按场景名构建仿真运行时。"""
    global _recorder, current_scenario_name
    if _recorder is not None:
        _recorder.close()
    _recorder = HistoryRecorder()

    selected = scenario_name or current_scenario_name
    scenario = get_scenario(selected)
    current_scenario_name = selected
    return scenario.build_runtime(history_recorder=_recorder)


def _finalize_current_run(reason: str) -> dict | None:
    """在重建 runtime 前归档当前已运行 tick 的 CSV、JSONL、摘要和图表。"""

    global _last_archived_run
    archived = archive_runtime_run(rt, _recorder, reason=reason, scenario_name=current_scenario_name)
    if archived is not None:
        _last_archived_run = archived
    return archived


# ---------------------------------------------------------------------------
# WebSocket 广播工具
# ---------------------------------------------------------------------------

async def broadcast(msg: dict) -> None:
    """向所有连接的客户端广播消息，自动清除已断开的连接。"""
    if not clients:
        return
    data = json.dumps(msg, ensure_ascii=False)
    dead: set[WebSocket] = set()
    for ws in list(clients):
        try:
            await ws.send_text(data)
        except Exception:
            # 发送失败说明客户端已断开，标记后统一移除
            dead.add(ws)
    clients.difference_update(dead)


# ---------------------------------------------------------------------------
# 仿真步进逻辑
# ---------------------------------------------------------------------------

async def do_step() -> float:
    """执行 world.astep()，完成后广播最新状态。

    world.astep() 使用 AsyncOpenAI 实现真正异步 I/O，
    LLM 调用期间释放事件循环，不会阻塞 WebSocket 心跳。
    """
    if _limit_reached():
        sim_state["running"] = False
        await broadcast(_status_payload(limit_reached=True))
        return 0.0

    start = time.perf_counter()
    await rt.world.astep()
    world_elapsed = time.perf_counter() - start

    snapshot_start = time.perf_counter()
    state = snapshot(rt.world, rt.platform)
    snapshot_elapsed = time.perf_counter() - snapshot_start

    broadcast_start = time.perf_counter()
    await broadcast({"type": "tick", "state": state})
    broadcast_elapsed = time.perf_counter() - broadcast_start

    if _limit_reached():
        sim_state["running"] = False
        await broadcast(_status_payload(limit_reached=True))

    total_elapsed = time.perf_counter() - start
    logger.info(
        "[Perf] tick=%d total=%.3fs world=%.3fs snapshot=%.3fs broadcast=%.3fs clients=%d",
        rt.world.time,
        total_elapsed,
        world_elapsed,
        snapshot_elapsed,
        broadcast_elapsed,
        len(clients),
    )
    return total_elapsed


async def _sim_loop() -> None:
    """自动步进循环，以 speed 倍率持续执行，直到 running 被置为 False。"""
    while sim_state["running"]:
        if _limit_reached():
            sim_state["running"] = False
            await broadcast(_status_payload(limit_reached=True))
            break
        elapsed = await do_step()
        target_interval = 1.0 / sim_state["speed"]
        sleep_seconds = max(0.0, target_interval - elapsed)
        if sleep_seconds > 0:
            await asyncio.sleep(sleep_seconds)


# ---------------------------------------------------------------------------
# FastAPI 应用
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时初始化运行时，关闭时自动清理（asynccontextmanager）。"""
    global rt
    setup_logging()
    load_dotenv()
    rt = _build_runtime(os.environ.get("SIM_SCENARIO") or "default_town")
    yield


app = FastAPI(lifespan=lifespan)

# 生产模式：Vite 构建产物由 FastAPI 直接托管，无需单独 nginx
FRONTEND_DIST = Path(__file__).parent.parent / "frontend" / "dist"
if FRONTEND_DIST.exists():
    _assets = FRONTEND_DIST / "assets"
    if _assets.exists():
        app.mount("/assets", StaticFiles(directory=str(_assets)), name="assets")


@app.get("/")
async def serve_index():
    """返回 React 应用入口页，未构建时给出提示。"""
    index = FRONTEND_DIST / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({"message": "Frontend not built. Run: cd frontend && npm run build"})


@app.get("/api/history/latest")
async def latest_archived_run():
    """返回最近一次 reset 前归档的实验输出信息。"""

    return {"archived_run": _last_archived_run}


@app.get("/history/{run_name}/{file_name}")
async def serve_history_file(run_name: str, file_name: str):
    """提供 reset 归档后的摘要和图表文件。"""

    if file_name not in HISTORY_VIEW_FILES:
        return JSONResponse({"error": "history file is not exposed"}, status_code=404)
    target = (HISTORY_DIR / run_name / file_name).resolve()
    history_root = HISTORY_DIR.resolve()
    if history_root not in target.parents:
        return JSONResponse({"error": "invalid history path"}, status_code=400)
    if not target.exists():
        return JSONResponse({"error": "history file not found"}, status_code=404)
    return FileResponse(str(target))


def _agent_or_error(agent_id: str):
    if rt is None:
        return None, JSONResponse({"error": "simulation runtime is not initialized"}, status_code=503)
    agent = rt.world.agents.get(agent_id)
    if agent is None:
        return None, JSONResponse({"error": f"agent not found: {agent_id}"}, status_code=404)
    return agent, None


@app.get("/api/agents/{agent_id}/memories")
async def agent_memories(agent_id: str):
    agent, error = _agent_or_error(agent_id)
    if error is not None:
        return error
    memories = rt.mem.list_agent_memories(agent.id)
    structured = {}
    if hasattr(rt.mem, "list_agent_structured_memories"):
        # 保留旧 memories 字段给现有前端，同时追加 SQLite 分表结果供新记忆窗口使用。
        structured = rt.mem.list_agent_structured_memories(agent.id)
    return {
        "agent_id": agent.id,
        "count": len(memories),
        "memories": memories,
        "structured": structured,
    }


@app.get("/api/agents/{agent_id}/trajectory")
async def agent_trajectory(agent_id: str):
    agent, error = _agent_or_error(agent_id)
    if error is not None:
        return error
    trajectory = list(agent.trajectory_buffer)
    return {
        "agent_id": agent.id,
        "count": len(trajectory),
        "trajectory": trajectory,
    }


def _agent_data_page(agent_id: str, data_kind: str) -> HTMLResponse:
    titles = {
        "memories": "智能体记忆",
        "trajectory": "智能体轨迹",
    }
    if data_kind not in titles:
        return HTMLResponse("unknown agent data page", status_code=404)

    safe_title = html.escape(titles[data_kind])
    safe_agent_id = html.escape(agent_id)
    agent_id_json = json.dumps(agent_id, ensure_ascii=False)
    data_kind_json = json.dumps(data_kind, ensure_ascii=False)
    title_json = json.dumps(titles[data_kind], ensure_ascii=False)
    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{safe_title} - {safe_agent_id}</title>
  <style>
    :root {{
      color-scheme: dark;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #111827;
      color: #e5e7eb;
    }}
    body {{ margin: 0; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 22px; }}
    .meta, #status {{ color: #9ca3af; font-size: 13px; }}
    #status {{ margin: 16px 0; }}
    button {{
      margin-top: 12px;
      background: #374151;
      color: #e5e7eb;
      border: 1px solid #4b5563;
      border-radius: 6px;
      padding: 6px 10px;
      cursor: pointer;
    }}
    button:hover {{ background: #4b5563; }}
    pre {{
      background: #1f2937;
      border: 1px solid #374151;
      border-radius: 10px;
      padding: 14px;
      white-space: pre-wrap;
      word-break: break-word;
      line-height: 1.55;
    }}
    .error {{ color: #f87171; }}
  </style>
</head>
<body>
  <h1>{safe_title}</h1>
  <div class="meta">agent_id: {safe_agent_id}</div>
  <button type="button" onclick="location.reload()">刷新</button>
  <div id="status">加载中...</div>
  <pre id="content"></pre>
  <script>
    const agentId = {agent_id_json};
    const dataKind = {data_kind_json};
    const title = {title_json};
    const endpoint = `/api/agents/${{encodeURIComponent(agentId)}}/${{dataKind}}`;
    const statusEl = document.getElementById('status');
    const contentEl = document.getElementById('content');

    fetch(endpoint)
      .then(async (response) => {{
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || response.statusText);
        return payload;
      }})
      .then((payload) => {{
        const rows = dataKind === 'memories' ? payload.memories : payload.trajectory;
        statusEl.textContent = `${{title}}：${{payload.count ?? rows.length}} 条`;
        contentEl.textContent = JSON.stringify(payload, null, 2);
      }})
      .catch((error) => {{
        statusEl.classList.add('error');
        statusEl.textContent = `加载失败：${{error.message}}`;
      }});
  </script>
</body>
</html>"""
    return HTMLResponse(page)


@app.get("/agent/{agent_id}/memories")
async def agent_memories_page(agent_id: str):
    return _agent_data_page(agent_id, "memories")


@app.get("/agent/{agent_id}/trajectory")
async def agent_trajectory_page(agent_id: str):
    return _agent_data_page(agent_id, "trajectory")


# ---------------------------------------------------------------------------
# WebSocket 端点
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    """接受 WebSocket 连接，发送初始状态，持续接收并分发控制命令。"""
    await websocket.accept()
    clients.add(websocket)
    try:
        # 新连接立即推送当前世界快照和运行状态，前端无需等待第一个 tick
        await websocket.send_text(json.dumps({
            "type": "init",
            "state": snapshot(rt.world, rt.platform),
            "running": sim_state["running"],
            "speed": sim_state["speed"],
            "scenario_name": current_scenario_name,
            "scenarios": list_scenarios(),
            "max_ticks": _simulation_step_limit(),
            "limit_reached": _limit_reached(),
            "archived_run": _last_archived_run,
        }, ensure_ascii=False))
        async for text in websocket.iter_text():
            try:
                msg = json.loads(text)
                await _handle_cmd(msg)
            except (json.JSONDecodeError, Exception):
                pass
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(websocket)


async def _handle_cmd(msg: dict) -> None:
    """处理来自前端的控制命令并广播状态变更。"""
    global rt, _sim_task
    cmd = msg.get("cmd")

    if cmd == "step":
        # 仅在暂停时允许手动单步，防止与自动步进并发执行
        if not sim_state["running"] and not _limit_reached():
            await do_step()
        elif _limit_reached():
            await broadcast(_status_payload(limit_reached=True))

    elif cmd == "resume":
        if _limit_reached():
            sim_state["running"] = False
            await broadcast(_status_payload(limit_reached=True))
        elif not sim_state["running"]:
            sim_state["running"] = True
            _sim_task = asyncio.create_task(_sim_loop())
            await broadcast(_status_payload(limit_reached=False))

    elif cmd == "pause":
        sim_state["running"] = False
        if _sim_task:
            _sim_task.cancel()
            _sim_task = None
        await broadcast(_status_payload(limit_reached=_limit_reached()))

    elif cmd == "set_speed":
        # 限制最低速度防止无限快速步进
        sim_state["speed"] = max(0.1, float(msg.get("value", 1.0)))
        await broadcast(_status_payload(limit_reached=_limit_reached()))

    elif cmd == "reset":
        sim_state["running"] = False
        if _sim_task:
            _sim_task.cancel()
            _sim_task = None
        # _build_runtime() 会调用 LLM 初始化，放到线程池避免阻塞事件循环
        loop = asyncio.get_event_loop()
        scenario_name = msg.get("scenario") or current_scenario_name
        if scenario_name not in list_scenarios():
            await broadcast({
                "type": "error",
                "message": f"unknown scenario: {scenario_name}",
                "scenarios": list_scenarios(),
            })
            return
        archived_run = await loop.run_in_executor(None, _finalize_current_run, "reset")
        rt = await loop.run_in_executor(None, _build_runtime, scenario_name)
        await broadcast({
            "type": "init",
            "state": snapshot(rt.world, rt.platform),
            "running": False,
            "speed": sim_state["speed"],
            "scenario_name": current_scenario_name,
            "scenarios": list_scenarios(),
            "max_ticks": _simulation_step_limit(),
            "limit_reached": False,
            "archived_run": archived_run,
        })


# ---------------------------------------------------------------------------
# 入口点
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
