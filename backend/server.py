from __future__ import annotations
import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from default_scenario import build_default_runtime
from persona.logger import setup_logging
from persona.runtime import SimulationRuntime
from persona.history_recorder import HistoryRecorder
from world.serializer import snapshot

# ---------------------------------------------------------------------------
# 全局运行时状态
# ---------------------------------------------------------------------------

# rt 在 lifespan 中初始化，reset 命令会重建它
rt: SimulationRuntime | None = None
# 历史记录器，随 rt 一起重建
_recorder: HistoryRecorder | None = None
# 当前所有已连接的 WebSocket 客户端，广播时遍历
clients: set[WebSocket] = set()
# 仿真控制状态，running=True 时自动步进循环运行
sim_state: dict = {"running": False, "speed": 1.0}
# 自动步进 asyncio Task 引用，pause/reset 时用于取消
_sim_task: asyncio.Task | None = None


# ---------------------------------------------------------------------------
# 仿真初始化（与 main.py 保持一致的智能体配置）
# ---------------------------------------------------------------------------

def _build_runtime() -> SimulationRuntime:
    """构建仿真运行时并初始化默认场景。"""
    global _recorder
    if _recorder is not None:
        _recorder.close()
    _recorder = HistoryRecorder()

    return build_default_runtime(history_recorder=_recorder)


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

async def do_step() -> None:
    """执行 world.astep()，完成后广播最新状态。

    world.astep() 使用 AsyncOpenAI 实现真正异步 I/O，
    LLM 调用期间释放事件循环，不会阻塞 WebSocket 心跳。
    """
    await rt.world.astep()
    await broadcast({"type": "tick", "state": snapshot(rt.world, rt.platform)})


async def _sim_loop() -> None:
    """自动步进循环，以 speed 倍率持续执行，直到 running 被置为 False。"""
    while sim_state["running"]:
        await do_step()
        # 每步之间等待 1/speed 秒，速度越高间隔越短
        await asyncio.sleep(1.0 / sim_state["speed"])


# ---------------------------------------------------------------------------
# FastAPI 应用
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时初始化运行时，关闭时自动清理（asynccontextmanager）。"""
    global rt
    setup_logging()
    load_dotenv()
    rt = _build_runtime()
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
        if not sim_state["running"]:
            await do_step()

    elif cmd == "resume":
        if not sim_state["running"]:
            sim_state["running"] = True
            _sim_task = asyncio.create_task(_sim_loop())
            await broadcast({"type": "status", "running": True, "speed": sim_state["speed"]})

    elif cmd == "pause":
        sim_state["running"] = False
        if _sim_task:
            _sim_task.cancel()
            _sim_task = None
        await broadcast({"type": "status", "running": False, "speed": sim_state["speed"]})

    elif cmd == "set_speed":
        # 限制最低速度防止无限快速步进
        sim_state["speed"] = max(0.1, float(msg.get("value", 1.0)))
        await broadcast({"type": "status", "running": sim_state["running"], "speed": sim_state["speed"]})

    elif cmd == "reset":
        sim_state["running"] = False
        if _sim_task:
            _sim_task.cancel()
            _sim_task = None
        # _build_runtime() 会调用 LLM 初始化，放到线程池避免阻塞事件循环
        loop = asyncio.get_event_loop()
        rt = await loop.run_in_executor(None, _build_runtime)
        await broadcast({
            "type": "init",
            "state": snapshot(rt.world, rt.platform),
            "running": False,
            "speed": sim_state["speed"],
        })


# ---------------------------------------------------------------------------
# 入口点
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
