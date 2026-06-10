from __future__ import annotations
import re
from typing import TYPE_CHECKING
from world.map import serialize_map_design

if TYPE_CHECKING:
    from world.world import World
    from social_sys.platform.platform import SocialPlatform


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


def snapshot(world: "World", platform: "SocialPlatform | None" = None) -> dict:
    """将当前世界状态序列化为可 JSON 传输的字典。

    仅提取前端渲染所需字段，避免将 LLM prompt、ChromaDB 对象等不可序列化
    内容混入广播消息。pos 格式为 [x, y]，与地图坐标系一致。
    """
    agents = [
        {
            "id": a.id,
            "pos": a.position,          # [row, col]，前端 pos[0]→sy（垂直），pos[1]→sx（水平）
            "role": a.role,
            "emotion": a.emotion,
            "task": a.task,
            "current_focus": a.current_focus,
            "salary": a.salary,                # 工资：公司自动交互时获得的 money 增量
            "satisfaction": dict(a.satisfaction),               # 客观满足度：satiety/relax 为 [0,100]，money 无上限
            "urgency": dict(a.urgency),           # 主观急迫度，通常为 [0,1]
            "satisfaction_threshold": dict(a.satisfaction_threshold),  # 任务完成判定线
            "opinion": round(a.opinion, 3),
            "last_think": _extract_last_think(a.history),
            "sleeping": a.sleeping,
            "sleep_ticks_remaining": a.sleep_ticks_remaining,
            "inside_building_id": a.inside_building_id,
        }
        for a in world.agents.values()
    ]
    objects = [
        {
            "id": o.id,
            "pos": o.position,
            "type": type(o).__name__,
            "kind": getattr(o, "kind", "objects"),  # 物品种类标识，前端用于查找描述元数据
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
        posts = [
            {
                "id": p.id,
                "author_id": p.author_id,
                "content": p.content,
                "time": p.time,
                "likes": p.likes,
                "dislikes": p.dislikes,
                "opinion_index": round(p.opinion_index, 3),
                "comments": [
                    {
                        "id": c.id,
                        "author_id": c.author_id,
                        "content": c.content,
                        "time": c.time,
                    }
                    for c in p.comments_list
                ],
            }
            for p in platform.posts
        ]
    return {
        "time": world.time,
        "map_size": [world.map.width, world.map.height],
        "map_design": serialize_map_design(getattr(world, "map_design", None)),
        "movements": list(getattr(world, "movements", [])),
        "agents": agents,
        "objects": objects,
        "posts": posts,
    }
