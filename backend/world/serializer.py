from __future__ import annotations
from typing import TYPE_CHECKING
from persona.llm.debug_trace import LLMTraceStore
from world.map import serialize_map_design

if TYPE_CHECKING:
    from world.world import World
    from social_sys.platform import SocialPlatform


def _extract_last_think(agent) -> str:
    """从结构化短期记忆中读取最近一次 decision_think 内容。

    未找到时返回空字符串，前端按空字符串判断是否渲染思考区域。
    """
    memory = getattr(agent, "short_term_memory", None)
    if memory is not None:
        entry = memory.latest({"decision_think"})
        if entry is not None:
            return str(entry.content or "").strip()
    return ""


def snapshot(world: "World", platform: "SocialPlatform | None" = None) -> dict:
    """将当前世界状态序列化为可 JSON 传输的字典。

    仅提取前端渲染所需字段，避免将 LLM prompt、ChromaDB 对象等不可序列化
    内容混入广播消息。pos 格式为 [x, y]，与地图坐标系一致。
    """
    trace_store = getattr(world, "llm_trace_store", None)
    display_tick = int(world.time) - 1 if int(world.time) > 0 else None
    if isinstance(trace_store, LLMTraceStore):
        # 快照是统一的安全清理点；完整调用内容仍由按需接口返回。
        trace_store.prune(int(world.time))

    agents = [
        {
            "id": a.id,
            "pos": a.position,          # [row, col]，前端 pos[0]→sy（垂直），pos[1]→sx（水平）
            "emotion": a.emotion,
            "task": a.task,
            "current_focus": a.current_focus,
            "salary": a.salary,                # 工资：公司自动交互时获得的 money 增量
            "satisfaction": dict(a.satisfaction),               # 客观满足度：satiety/relax 为 [0,100]，money 无上限
            "urgency": dict(a.urgency),           # 主观急迫度，通常为 [0,1]
            "satisfaction_threshold": dict(a.satisfaction_threshold),  # 任务完成判定线
            "need_gap": dict(a.need_gap),
            "pressure_memory": dict(a.pressure_memory),
            "load_saturation": dict(a.load_saturation),
            "effective_pressure": dict(a.effective_pressure),
            "last_psychological_assessment": a.last_psychological_assessment,
            "opinion": round(a.opinion, 3),
            "opinion_scores": dict(a.opinion_scores),
            "last_opinion_assessment": a.last_opinion_assessment,
            "last_opinion_voting": a.last_opinion_voting,
            "last_think": _extract_last_think(a),
            "llm_debug": {
                "display_tick": display_tick,
                "call_count": (
                    trace_store.count_for(a.id, display_tick)
                    if isinstance(trace_store, LLMTraceStore) and display_tick is not None
                    else 0
                ),
            },
            "sleeping": a.sleeping,
            "sleep_ticks_remaining": a.sleep_ticks_remaining,
            "personal_bed_id": getattr(a, "personal_bed_id", None),
            "personal_bed_position": getattr(a, "personal_bed_position", None),
            "personal_bed_entrance": getattr(a, "personal_bed_entrance", None),
            "inside_building_id": a.inside_building_id,
        }
        for a in world.agents.values()
    ]
    objects = [
        {
            "id": o.id,
            "pos": o.position,
            "footprint": [list(cell) for cell in getattr(o, "footprint", [o.position])],
            "entrance": getattr(o, "entrance", None),
            "sprite_key": getattr(o, "sprite_key", None),
            "type": type(o).__name__,
            "kind": getattr(o, "kind", "objects"),  # 物品种类标识，前端用于查找描述元数据
            "owner_agent_id": getattr(o, "owner_agent_id", None),
            "free_num": getattr(o, "free_num", None),
            "description": o.get_desc() if hasattr(o, "get_desc") else "",  # 由 objects.get_desc() 动态生成
            # num 为 None 表示对象无数量属性；num <= 0 时前端隐藏图形
            "num": getattr(o, "num", None),
            # occupants: bed 用 occupant_id（单人），building 子类用 occupants 列表
            "occupant_count": len(getattr(o, "occupants", [])) or (1 if getattr(o, "occupant_id", None) else 0),
            "occupants": (
                list(getattr(o, "occupants", None) or [])
                if getattr(o, "occupants", None) is not None
                else ([o.occupant_id] if getattr(o, "occupant_id", None) else [])
            ),
        }
        for o in world.objects.values()
    ]
    posts = []
    if platform is not None:
        posts_lock = getattr(platform, "_posts_lock", None)
        if posts_lock is None:
            post_objects = list(platform.posts)
        else:
            with posts_lock:
                post_objects = list(platform.posts)
        # 复用 Post 的锁内快照，保证前端状态与浏览载荷字段一致。
        for post in post_objects:
            post_data = post.to_dict()
            post_data["opinion_index"] = round(post.opinion_index, 3)
            posts.append(post_data)
    return {
        "time": world.time,
        "scenario_name": getattr(world, "scenario_name", "default_town"),
        "simulation_step_limit": world.agents and next(iter(world.agents.values())).config.simulation_step_limit or 0,
        "map_size": [world.map.width, world.map.height],
        "map_design": serialize_map_design(getattr(world, "map_design", None)),
        "movements": list(getattr(world, "movements", [])),
        "agents": agents,
        "objects": objects,
        "posts": posts,
    }
