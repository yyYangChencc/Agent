from __future__ import annotations
import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from persona.logger import setup_logging
from persona.runtime import SimulationRuntime
from world.objects import food, bed, company, food_shop, playground
from world.serializer import snapshot

# ---------------------------------------------------------------------------
# 全局运行时状态
# ---------------------------------------------------------------------------

# rt 在 lifespan 中初始化，reset 命令会重建它
rt: SimulationRuntime | None = None
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
    """构建仿真运行时并初始化5个智能体、3处食物和信任关系。"""
    r = SimulationRuntime.build(conversation_max_rounds=2)
    # 每次 reset 清空向量记忆，保证实验可重复
    r.mem.reset_all()

    # 5个角色各有不同的政治/社交倾向，用于观测意见传播效果
    a = r.create_agent("agent_1", [3, 3],
        role="保守主义者，倾向于节约资源，不喜欢变化", speaking_style="沉稳、措辞谨慎")
    b = r.create_agent("agent_2", [4, 2],
        role="积极探索者，乐于尝试新事物并分享经验", speaking_style="热情、喜欢分享")
    c = r.create_agent("agent_3", [6, 6],
        role="中立观察者，善于倾听各方意见后再表态", speaking_style="理性、措辞中立")
    d = r.create_agent("agent_4", [2, 9],
        role="激进改革派，主张打破现有秩序追求效率", speaking_style="直接、充满激情")
    e = r.create_agent("agent_5", [8, 4],
        role="社区协调员，重视群体和谐与共识", speaking_style="温和、善于调解")

    # opinion 范围 0~1，0.5 为中立；此初始值代表各角色的预设立场
    a.opinion = 0.15; b.opinion = 0.45; c.opinion = 0.50
    d.opinion = 0.85; e.opinion = 0.60

    # offline_trust 超过 friend_trust_threshold 才会触发离线意见同化
    a.offline_trust["agent_2"] = 0.75; a.offline_trust["agent_3"] = 0.65
    b.offline_trust["agent_1"] = 0.75; b.offline_trust["agent_3"] = 0.70
    c.offline_trust["agent_1"] = 0.65; c.offline_trust["agent_2"] = 0.70
    d.offline_trust["agent_5"] = 0.80; e.offline_trust["agent_4"] = 0.80
    c.offline_trust["agent_5"] = 0.62; e.offline_trust["agent_3"] = 0.62

    # online_trust 影响看到对方帖子后意见偏移的权重
    a.online_trust["agent_2"] = 0.55; a.online_trust["agent_3"] = 0.60
    b.online_trust["agent_1"] = 0.50; b.online_trust["agent_4"] = 0.35
    c.online_trust["agent_1"] = 0.55; c.online_trust["agent_4"] = 0.55
    c.online_trust["agent_5"] = 0.60; d.online_trust["agent_5"] = 0.65
    d.online_trust["agent_1"] = 0.30; e.online_trust["agent_4"] = 0.60
    e.online_trust["agent_3"] = 0.65

    # 关注关系决定帖子分发范围（被关注者发帖后关注者能看到）
    a.add_follower("agent_2"); a.add_follower("agent_3")
    b.add_follower("agent_1"); b.add_follower("agent_3"); b.add_follower("agent_5")
    c.add_follower("agent_1"); c.add_follower("agent_4"); c.add_follower("agent_5")
    d.add_follower("agent_5"); d.add_follower("agent_3")
    e.add_follower("agent_4"); e.add_follower("agent_3"); e.add_follower("agent_2")

    # ------------------------------------------------------------------
    # 场景布置（25×25 地图，按功能分区）
    #
    #  住宅区（左上，x=1-4, y=1-12）：5 张床，智能体初始在各自床旁
    #  工作区（右上，x=18-22, y=2-6）：2 家公司
    #  商业区（中左，x=1-4, y=16-22）：2 家食品店
    #  娱乐区（右下，x=18-22, y=18-22）：1 个游乐场
    #  散落食物：地图中部
    # ------------------------------------------------------------------

    # 住宅区：5 张床（x=2，y 间距 2）
    bed("bed_1", [2,  2], r.world)
    bed("bed_2", [2,  4], r.world)
    bed("bed_3", [2,  6], r.world)
    bed("bed_4", [2,  8], r.world)
    bed("bed_5", [2, 10], r.world)

    # 工作区：2 家公司
    company("company_1", [20, 3], r.world, salary=10)
    company("company_2", [22, 6], r.world, salary=15)

    # 商业区：2 家食品店
    food_shop("shop_1", [2, 18], r.world, food_num=20, provide=30, price=5)
    food_shop("shop_2", [4, 21], r.world, food_num=15, provide=20, price=3)

    # 娱乐区：游乐场
    playground("playground_1", [20, 20], r.world, provide=20, price=3)

    # 散落食物（中部区域，数量充足供多轮消耗）
    food("food_1", 3, 20, [10,  8], r.world)
    food("food_2", 3, 20, [12, 14], r.world)
    food("food_3", 3, 20, [ 8, 18], r.world)
    food("food_4", 3, 20, [15,  5], r.world)
    food("food_5", 3, 20, [16, 20], r.world)

    # 初始记忆：告知所有智能体各区域位置
    map_info = (
        "地图信息：住宅区在左上角，床(bed_1~bed_5)位于x=2,y=2/4/6/8/10；"
        "工作区在右上角，company_1在(20,3)、company_2在(22,6)，工作可赚钱；"
        "商业区在左下角，shop_1在(2,18)、shop_2在(4,21)，可购买食物；"
        "娱乐区在右下角，playground_1在(20,20)，可放松；"
        "地图中部(10,8)(12,14)(8,18)(15,5)(16,20)附近有散落食物"
    )
    for aid in ["agent_1", "agent_2", "agent_3", "agent_4", "agent_5"]:
        r.mem.store_agent_memory(aid, map_info, memory_type="system", importance=0.9)

    return r


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
    """在线程池执行 world.step()，完成后广播最新状态。

    world.step() 内部调用 LLM，属于阻塞 I/O，必须放到 executor 中
    避免阻塞事件循环导致 WebSocket 心跳超时。
    """
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, rt.world.step)
    await broadcast({"type": "tick", "state": snapshot(rt.world)})


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
            "state": snapshot(rt.world),
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
            "state": snapshot(rt.world),
            "running": False,
            "speed": sim_state["speed"],
        })


# ---------------------------------------------------------------------------
# 入口点
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
