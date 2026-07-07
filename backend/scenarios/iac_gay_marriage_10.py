from __future__ import annotations

from copy import deepcopy
from typing import Any

from .builder import build_runtime_from_spec
from .iac_gay_marriage import OPPOSE_INFLUENCERS, SPEC as FULL_SPEC, SUPPORT_INFLUENCERS


SELECTED_AGENT_IDS = [
    "iac_author_323",
    "iac_author_399",
    "iac_author_1491",
    "iac_author_105",
    "iac_author_1438",
    "iac_author_1234",
    "iac_author_817",
    "iac_author_883",
    "iac_author_357",
    "iac_author_99",
]

COMPACT_POSITIONS = [
    [0, 0],
    [0, 1],
    [0, 2],
    [0, 3],
    [0, 4],
    [0, 5],
    [0, 6],
    [0, 7],
    [0, 8],
    [0, 9],
]


def _selected_set() -> set[str]:
    """集中保存 10 人子场景的实体智能体 ID。"""

    return set(SELECTED_AGENT_IDS)


def _influencer_set() -> set[str]:
    """集中保存无实体投放账号 ID。"""

    return set(SUPPORT_INFLUENCERS + OPPOSE_INFLUENCERS)


def _selected_agents() -> list[dict[str, Any]]:
    """从完整 IAC 场景中抽取 10 个实体智能体并重新分配出生点。"""

    by_id = {item["id"]: item for item in FULL_SPEC["agents"]}
    agents: list[dict[str, Any]] = []
    for index, agent_id in enumerate(SELECTED_AGENT_IDS):
        item = deepcopy(by_id[agent_id])
        item["position"] = list(COMPACT_POSITIONS[index])
        item["subset_role"] = _subset_role(float(item.get("initial_opinion", 0.0)))
        agents.append(item)
    return agents


def _subset_role(opinion: float) -> str:
    """按初始观念标注子场景分组，便于配置快照审计。"""

    if opinion > 0.35:
        return "support"
    if opinion < -0.35:
        return "oppose"
    return "neutral"


def _selected_objects() -> list[dict[str, Any]]:
    """保留 10 张床和完整基础设施，避免小场景携带多余床位。"""

    objects: list[dict[str, Any]] = []
    for item in FULL_SPEC["objects"]:
        kind = item.get("kind")
        object_id = str(item.get("id") or "")
        if kind == "bed":
            suffix = object_id.removeprefix("bed_")
            if suffix.isdigit() and int(suffix) <= 10:
                objects.append(deepcopy(item))
            continue
        objects.append(deepcopy(item))
    return objects


def _map_design(objects: list[dict[str, Any]]) -> dict[str, Any]:
    """移除未使用床位的地图对象标注，保持前端展示与场景对象一致。"""

    design = deepcopy(FULL_SPEC["map_design"])
    object_ids = {str(item.get("id") or "") for item in objects}
    regions = design.get("object_regions")
    if isinstance(regions, dict):
        for object_id in list(regions.keys()):
            if object_id.startswith("bed_") and object_id not in object_ids:
                regions.pop(object_id, None)
    return design


def _filter_edges(edges: list[dict[str, Any]], target_field: str) -> list[dict[str, Any]]:
    """过滤完整场景边，只保留 10 个实体智能体与投放账号相关的边。"""

    selected = _selected_set()
    allowed_targets = selected | _influencer_set()
    filtered = []
    for edge in edges:
        source = str(edge.get("source") or edge.get("follower") or "")
        target = str(edge.get(target_field) or "")
        if source in selected and target in allowed_targets:
            filtered.append(deepcopy(edge))
    return filtered


def _follow_edges() -> list[dict[str, str]]:
    """保留数据驱动关注关系，并补充 10 人内部环形关注。"""

    selected = SELECTED_AGENT_IDS
    edges = {
        (str(edge["follower"]), str(edge["author"]))
        for edge in _filter_edges(FULL_SPEC["follow_edges"], "author")
    }
    for index, follower in enumerate(selected):
        edges.add((follower, selected[(index + 1) % len(selected)]))
        edges.add((follower, selected[(index + 2) % len(selected)]))
    return [{"follower": follower, "author": author} for follower, author in sorted(edges)]


def _online_trust(follow_edges: list[dict[str, str]]) -> list[dict[str, Any]]:
    """保留数据驱动线上信任，并给补充关注边设置默认信任。"""

    edges = {
        (str(edge["source"]), str(edge["target"])): deepcopy(edge)
        for edge in _filter_edges(FULL_SPEC["online_trust"], "target")
    }
    for edge in follow_edges:
        key = (edge["follower"], edge["author"])
        if key not in edges:
            edges[key] = {"source": edge["follower"], "target": edge["author"], "value": 0.55}
    return [edges[key] for key in sorted(edges)]


def _offline_trust() -> list[dict[str, Any]]:
    """生成 10 人小场景线下熟人关系，支持沙盒内对话与关系事件。"""

    selected = SELECTED_AGENT_IDS
    edges: set[tuple[str, str, float]] = set()
    for index, source in enumerate(selected):
        for offset in (1, 2):
            target = selected[(index + offset) % len(selected)]
            edges.add((source, target, 0.50))
            edges.add((target, source, 0.50))

    groups: dict[str, list[str]] = {"support": [], "oppose": [], "neutral": []}
    for item in _selected_agents():
        groups[str(item["subset_role"])].append(str(item["id"]))
    for members in groups.values():
        for index, source in enumerate(members):
            if len(members) > 1:
                target = members[(index + 1) % len(members)]
                edges.add((source, target, 0.65))
                edges.add((target, source, 0.65))
    return [{"source": source, "target": target, "value": value} for source, target, value in sorted(edges)]


def _selected_memories() -> list[dict[str, Any]]:
    """只保留 10 个实体智能体的初始化记忆。"""

    selected = _selected_set()
    return [
        deepcopy(item)
        for item in FULL_SPEC["memories"]
        if str(item.get("agent_id") or "") in selected
    ]


def _build_spec() -> dict[str, Any]:
    """从完整 IAC 场景派生 10 人可运行场景。"""

    objects = _selected_objects()
    follow_edges = _follow_edges()
    spec = deepcopy(FULL_SPEC)
    spec["name"] = "iac_gay_marriage_10"
    spec["parent_scenario"] = "iac_gay_marriage"
    spec["selection_rule"] = (
        "按 IAC 初始化种子固定抽取 3 个支持、3 个反对、4 个中立智能体；"
        "支持/反对组优先目标讨论前已评分发言数量更高者，中立组优先初始观念最接近 0 且资料量更高者。"
    )
    spec["agent_ids"] = list(SELECTED_AGENT_IDS)
    spec["agents"] = _selected_agents()
    spec["objects"] = objects
    spec["map_design"] = _map_design(objects)
    spec["offline_trust"] = _offline_trust()
    spec["follow_edges"] = follow_edges
    spec["online_trust"] = _online_trust(follow_edges)
    spec["memories"] = _selected_memories()
    spec["map_memory"] = (
        "IAC gay marriage 10 人子场景从完整 105 人场景派生；"
        "实体智能体缩小为 10 人，投放账号和 52 条线上投放帖子保持不变。"
    )
    return spec


SPEC: dict[str, Any] = _build_spec()


def build_runtime(**kwargs):
    """构建 IAC v2 gay marriage 10 人子场景。"""

    return build_runtime_from_spec(deepcopy(SPEC), **kwargs)
