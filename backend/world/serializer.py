from __future__ import annotations
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from world.world import World


def _extract_last_think(history: list[str]) -> str:
    """从智能体历史记录中反向搜索，返回最近一次 <Think> 标签的内容。

    history 中每条记录可能包含多轮对话，倒序遍历保证取到最新的思考片段。
    未找到时返回空字符串，前端按空字符串判断是否渲染思考区域。
    """
    for h in reversed(history):
        m = re.search(r"<Think>(.*?)</Think>", h, re.DOTALL)
        if m:
            return m.group(1).strip()
    return ""


def snapshot(world: "World") -> dict:
    """将当前世界状态序列化为可 JSON 传输的字典。

    仅提取前端渲染所需字段，避免将 LLM prompt、ChromaDB 对象等不可序列化
    内容混入广播消息。pos 格式为 [x, y]，与地图坐标系一致。
    """
    agents = [
        {
            "id": a.id,
            "pos": a.position,          # [col, row]，对应画布像素 = pos * CELL
            "role": a.role,
            "emotion": a.emotion,
            "task": a.task,
            "current_focus": a.current_focus,
            "need": dict(a.need),               # 客观需求值，0→1
            "demand": dict(a.demand),           # 主观急迫度，1→0
            "demand_threshold": dict(a.demand_threshold),  # 任务完成判定线
            "opinion": round(a.opinion, 3),
            "last_think": _extract_last_think(a.history),
        }
        for a in world.agents.values()
    ]
    objects = [
        {
            "id": o.id,
            "pos": o.position,
            "type": type(o).__name__,
            # num 为 None 表示对象无数量属性；num <= 0 时前端隐藏图形
            "num": getattr(o, "num", None),
        }
        for o in world.objects.values()
    ]
    return {
        "time": world.time,
        "map_size": [world.map.width, world.map.height],
        "agents": agents,
        "objects": objects,
    }
