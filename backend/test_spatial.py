"""
test_spatial.py — 测试 bed/building 进出的空间逻辑，不需要 LLM。

运行方式：
  Terminal 1: python backend/test_spatial.py
  Terminal 2: cd frontend && npm run dev
  浏览器打开 http://localhost:5173

脚本化序列（每步间隔 1.5 秒，可通过前端 step/pause 控制）：
  步骤 0  : 初始状态
  步骤 1  : agent_1 进入 bed_1 睡觉（圆圈消失）
  步骤 2-8: 睡眠 tick（sleep_time=8，relax 分步恢复）
  步骤 9  : agent_1 醒来，出现在 bed_1 旁边的空格
  步骤 10 : agent_2 进入 building_1（圆圈消失，建筑显示占用数）
  步骤 11 : agent_2 离开 building_1（圆圈出现在建筑旁边）
  步骤 12 : agent_3 在建筑内时调用 move，自动出建筑后前往目标
"""
from __future__ import annotations
import asyncio
import json
import sys
import os
import unittest

# 确保 backend/ 在 sys.path 中
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from pathlib import Path
try:
    import uvicorn
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse
    from fastapi.staticfiles import StaticFiles
except ModuleNotFoundError as exc:
    if __name__ != "__main__":
        raise unittest.SkipTest(f"spatial demo web dependency is not installed: {exc.name}")
    raise SystemExit(
        f"Missing dependency: {exc.name}. Install project dependencies with: "
        "python -m pip install -r requirements.txt"
    ) from exc

from persona.logger import setup_logging, get_logger
from persona.config import AgentConfig
from persona.agents.agent import Agent
from world.world import World
from world.objects import bed, building, food_shop
from world.serializer import snapshot
from tools.operator_tools import Operator

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 最小化 stub，让 Agent 可以在不连接 LLM 的情况下实例化
# ---------------------------------------------------------------------------

class _StubMem:
    def smart_retrieve(self, *a, **kw): return []
    def store_agent_memory(self, *a, **kw): pass

class _StubPolicy:
    def decide(self, *a, **kw): return ""

class _StubReflect:
    def step(self, *a, **kw): pass
    prompt = None
    llm = None

class _StubPlatform:
    def give_post(self, *a, **kw): return []
    def execute(self, *a, **kw): return ""
    def add_agent(self, *a, **kw): pass
    def get_agent(self, *a, **kw): return None


# ---------------------------------------------------------------------------
# 构建测试世界
# ---------------------------------------------------------------------------

def _build_world() -> tuple[World, Operator]:
    config = AgentConfig(sleep_time=4, sleep_relax_recover=50.0)
    world = World()
    op = Operator(world)

    stub_mem = _StubMem()
    stub_policy = _StubPolicy()
    stub_reflect = _StubReflect()
    stub_platform = _StubPlatform()

    def make_agent(aid, pos):
        return Agent(
            agent_id=aid,
            position=pos,
            world=world,
            policy=stub_policy,
            mem=stub_mem,
            reflect=stub_reflect,
            platform=stub_platform,
            social_policy=stub_policy,
            config=config,
        )

    # agent_1: 紧邻 bed_1（距离²=1，满足 eat_distance_sq=2）
    make_agent("agent_1", [4, 5])
    # agent_2: 紧邻 building_1
    make_agent("agent_2", [11, 12])
    # agent_3: 先进入 building_2，再用 move 自动出建筑
    make_agent("agent_3", [17, 18])
    # agent_4: 紧邻 shop_1，用于验证进入建筑后立即自动 interact，停留时每 tick 继续自动 interact
    agent_4 = make_agent("agent_4", [19, 5])
    agent_4.satisfaction["money"] = 50

    # 放置物品
    bed("bed_1", [5, 5], world)
    building("building_1", [12, 12], world)
    building("building_2", [18, 18], world)
    food_shop("shop_1", [20, 5], world)

    return world, op


# ---------------------------------------------------------------------------
# 手动 tick（仅处理睡眠状态，不调用 LLM）
# ---------------------------------------------------------------------------

def _manual_tick(world: World) -> None:
    world.time += 1
    world.auto_interact_inside_buildings()
    for agent in world.agents.values():
        if agent.sleeping:
            agent.sleep_ticks_remaining -= 1
            agent.tick_sleep_recovery()
            if agent.sleep_ticks_remaining <= 0:
                bed_obj = world.objects.get(agent.sleeping_on_bed_id)
                agent.wakeup(bed_obj)
        else:
            agent.tick_satisfaction()


# ---------------------------------------------------------------------------
# 脚本化动作序列
# ---------------------------------------------------------------------------

def _build_script(world: World, op: Operator) -> list[tuple[str, callable]]:
    """返回 (描述, 执行函数) 列表，按顺序执行。"""
    script = []

    # 步骤 1: agent_1 睡觉
    script.append(("agent_1 在 bed_1 上睡觉", lambda: op.sleep("agent_1", "bed_1")))

    # 步骤 2-5: 睡眠 tick（sleep_time=4）
    for i in range(4):
        script.append((f"睡眠 tick {i+1}/4", lambda: _manual_tick(world)))

    # 步骤 6: agent_2 进入 building_1
    script.append(("agent_2 进入 building_1", lambda: op.enter_building("agent_2", "building_1")))

    # 步骤 7: agent_2 离开 building_1
    script.append(("agent_2 离开 building_1", lambda: op.exit_building("agent_2")))

    # 步骤 8: agent_3 进入 building_2
    script.append(("agent_3 进入 building_2", lambda: op.enter_building("agent_3", "building_2")))

    # 步骤 9: agent_3 在建筑内调用 move（自动出建筑后前往目标）
    script.append(("agent_3 在建筑内 move → [15, 15]（自动出建筑）",
                   lambda: op.move("agent_3", 15, 15)))

    def _assert_shop_entry_auto_interact():
        agent = world.agents["agent_4"]
        before_satiety = agent.satisfaction["satiety"]
        before_money = agent.satisfaction["money"]
        result = op.enter_building("agent_4", "shop_1")
        assert agent.inside_building_id == "shop_1"
        assert agent.satisfaction["satiety"] > before_satiety
        assert agent.satisfaction["money"] < before_money
        return result

    def _assert_shop_tick_auto_interact():
        agent = world.agents["agent_4"]
        before_satiety = agent.satisfaction["satiety"]
        before_money = agent.satisfaction["money"]
        _manual_tick(world)
        assert agent.inside_building_id == "shop_1"
        assert agent.satisfaction["satiety"] > before_satiety
        assert agent.satisfaction["money"] < before_money
        return "shop_1 自动 interact 已在 tick 内再次触发"

    def _assert_shop_exit_stops_auto_interact():
        agent = world.agents["agent_4"]
        result = op.exit_building("agent_4")
        before_satiety = agent.satisfaction["satiety"]
        before_money = agent.satisfaction["money"]
        _manual_tick(world)
        assert agent.inside_building_id is None
        assert agent.satisfaction["satiety"] <= before_satiety
        assert agent.satisfaction["money"] == before_money
        return result

    # 步骤 10-12: 验证建筑交互流程简化后，食品店不需要 buy，进入和停留自动 interact，离开后停止
    script.append(("agent_4 进入 shop_1 后立即自动 interact", _assert_shop_entry_auto_interact))
    script.append(("agent_4 仍在 shop_1 内，tick 自动 interact", _assert_shop_tick_auto_interact))
    script.append(("agent_4 离开 shop_1 后停止自动 interact", _assert_shop_exit_stops_auto_interact))

    return script


# ---------------------------------------------------------------------------
# FastAPI WebSocket 服务
# ---------------------------------------------------------------------------

world, op = _build_world()
script = _build_script(world, op)
script_idx = 0
clients: set[WebSocket] = set()
sim_state = {"running": False, "speed": 1.0}
_sim_task = None

app = FastAPI()

FRONTEND_DIST = Path(__file__).parent.parent / "frontend" / "dist"
if FRONTEND_DIST.exists():
    _assets = FRONTEND_DIST / "assets"
    if _assets.exists():
        app.mount("/assets", StaticFiles(directory=str(_assets)), name="assets")


@app.get("/")
async def index():
    idx = FRONTEND_DIST / "index.html"
    if idx.exists():
        from fastapi.responses import FileResponse
        return FileResponse(str(idx))
    return JSONResponse({"message": "Frontend not built. Run: cd frontend && npm run build"})


async def broadcast(msg: dict) -> None:
    data = json.dumps(msg, ensure_ascii=False)
    dead = set()
    for ws in list(clients):
        try:
            await ws.send_text(data)
        except Exception:
            dead.add(ws)
    clients.difference_update(dead)


def _do_script_step() -> str:
    """执行下一个脚本步骤，返回描述。若脚本结束则只 tick。"""
    global script_idx
    if script_idx < len(script):
        desc, fn = script[script_idx]
        result = fn()
        script_idx += 1
        if "tick" not in desc:
            world.time += 1
        msg = f"[步骤 {script_idx}/{len(script)}] {desc}"
        if result:
            msg += f" → {result}"
        logger.info(msg)
        return msg
    else:
        _manual_tick(world)
        msg = f"[脚本结束] tick {world.time}"
        logger.info(msg)
        return msg


async def do_step() -> None:
    loop = asyncio.get_event_loop()
    desc = await loop.run_in_executor(None, _do_script_step)
    await broadcast({"type": "tick", "state": snapshot(world), "desc": desc})


async def _sim_loop() -> None:
    while sim_state["running"]:
        await do_step()
        await asyncio.sleep(1.0 / sim_state["speed"])


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    global _sim_task
    await websocket.accept()
    clients.add(websocket)
    try:
        await websocket.send_text(json.dumps({
            "type": "init",
            "state": snapshot(world),
            "running": sim_state["running"],
            "speed": sim_state["speed"],
        }, ensure_ascii=False))
        async for text in websocket.iter_text():
            try:
                msg = json.loads(text)
                cmd = msg.get("cmd")
                if cmd == "step" and not sim_state["running"]:
                    await do_step()
                elif cmd == "resume" and not sim_state["running"]:
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
                    sim_state["speed"] = max(0.1, float(msg.get("value", 1.0)))
                    await broadcast({"type": "status", "running": sim_state["running"], "speed": sim_state["speed"]})
            except Exception:
                pass
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(websocket)


if __name__ == "__main__":
    setup_logging()
    logger.info("测试服务器启动中...")
    for i, (desc, _) in enumerate(script, 1):
        logger.info("  脚本步骤 %d: %s", i, desc)
    logger.info("前端：http://localhost:5173（需另开终端运行 cd frontend && npm run dev）")
    logger.info("注意：运行前请确保 server.py 已停止（两者都占用 :8000）")
    uvicorn.run("test_spatial:app", host="0.0.0.0", port=8000, reload=False)
