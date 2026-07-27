from __future__ import annotations

from copy import deepcopy
from typing import Any

from .builder import build_runtime_from_spec
from .iac_gay_marriage import OPPOSE_INFLUENCERS, SPEC as FULL_SPEC, SUPPORT_INFLUENCERS
from .map_designs import polarization_map_design


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
        item.pop("community_id", None)
        item.pop("initial_role", None)
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
    """为 10 人子场景保留独立的 25x25 紧凑设施布局。"""

    beds = []
    for index in range(10):
        row = 1 if index < 5 else 4
        col = 1 + (index % 5) * 2
        beds.append(
            {
                "kind": "bed",
                "id": f"bed_{index + 1}",
                "position": [row, col],
                "footprint": [[row, col], [row, col + 1]],
                "sprite_key": f"home_{index % 6}",
            }
        )
    facilities = [
        {
            "kind": "company",
            "id": "company_1",
            "position": [17, 2],
            "footprint": _square_footprint(17, 2),
            "sprite_key": "company_0",
            "params": {"salary": 8, "relax_cost": 8},
        },
        {
            "kind": "company",
            "id": "company_2",
            "position": [22, 6],
            "footprint": _square_footprint(22, 6),
            "sprite_key": "company_1",
            "params": {"salary": 12, "relax_cost": 12},
        },
        {
            "kind": "food_shop",
            "id": "shop_1",
            "position": [2, 18],
            "footprint": _square_footprint(2, 18),
            "sprite_key": "shop_0",
            "params": {"food_num": 160, "provide": 30, "price": 6},
        },
        {
            "kind": "food_shop",
            "id": "shop_2",
            "position": [3, 21],
            "footprint": _square_footprint(3, 21),
            "sprite_key": "shop_1",
            "params": {"food_num": 160, "provide": 20, "price": 4},
        },
        {
            "kind": "playground",
            "id": "playground_1",
            "position": [17, 18],
            "footprint": _square_footprint(17, 18),
            "sprite_key": "playground_0",
            "params": {"provide": 18, "price": 5},
        },
        {
            "kind": "playground",
            "id": "playground_2",
            "position": [17, 22],
            "footprint": _square_footprint(17, 22),
            "sprite_key": "playground_1",
            "params": {"provide": 18, "price": 5},
        },
        {
            "kind": "playground",
            "id": "playground_3",
            "position": [22, 18],
            "footprint": _square_footprint(22, 18),
            "sprite_key": "playground_2",
            "params": {"provide": 18, "price": 5},
        },
        {
            "kind": "playground",
            "id": "playground_4",
            "position": [22, 22],
            "footprint": _square_footprint(22, 22),
            "sprite_key": "playground_3",
            "params": {"provide": 18, "price": 5},
        },
    ]
    return beds + facilities


def _square_footprint(row: int, col: int) -> list[list[int]]:
    """生成 2x2 建筑占地。"""

    return [[row, col], [row, col + 1], [row + 1, col], [row + 1, col + 1]]


COMPACT_OBJECT_REGIONS = {
    **{
        f"bed_{index + 1}": {
            "region_id": "residential_area",
            "entrance": [1 if index < 5 else 4, 1 + (index % 5) * 2],
        }
        for index in range(10)
    },
    "company_1": {"region_id": "work_area", "entrance": [18, 2]},
    "company_2": {"region_id": "work_area", "entrance": [23, 6]},
    "shop_1": {"region_id": "commercial_area", "entrance": [3, 18]},
    "shop_2": {"region_id": "commercial_area", "entrance": [4, 21]},
    "playground_1": {"region_id": "recreation_area", "entrance": [18, 18]},
    "playground_2": {"region_id": "recreation_area", "entrance": [18, 22]},
    "playground_3": {"region_id": "recreation_area", "entrance": [23, 18]},
    "playground_4": {"region_id": "recreation_area", "entrance": [23, 22]},
}


# 固定随机种子测试页生成的植被坐标，保证实时画面与历史回放一致。
COMPACT_DECORATIONS = [
    {"kind": "tree", "pos": [11, 11], "sprite_key": "tree_1"},
    {"kind": "shrub", "pos": [15, 23], "sprite_key": "shrub_0"},
    {"kind": "shrub", "pos": [9, 21], "sprite_key": "shrub_0"},
    {"kind": "tree", "pos": [13, 2], "sprite_key": "tree_0"},
    {"kind": "shrub", "pos": [9, 20], "sprite_key": "shrub_1"},
    {"kind": "shrub", "pos": [10, 1], "sprite_key": "shrub_0"},
    {"kind": "tree", "pos": [11, 0], "sprite_key": "tree_3"},
    {"kind": "shrub", "pos": [11, 4], "sprite_key": "shrub_1"},
    {"kind": "shrub", "pos": [14, 11], "sprite_key": "shrub_0"},
    {"kind": "tree", "pos": [21, 15], "sprite_key": "tree_3"},
    {"kind": "shrub", "pos": [3, 6], "sprite_key": "shrub_1"},
    {"kind": "shrub", "pos": [18, 15], "sprite_key": "shrub_1"},
    {"kind": "tree", "pos": [3, 4], "sprite_key": "tree_1"},
    {"kind": "shrub", "pos": [10, 19], "sprite_key": "shrub_0"},
    {"kind": "shrub", "pos": [24, 15], "sprite_key": "shrub_0"},
    {"kind": "tree", "pos": [13, 0], "sprite_key": "tree_1"},
    {"kind": "shrub", "pos": [24, 13], "sprite_key": "shrub_1"},
    {"kind": "shrub", "pos": [9, 2], "sprite_key": "shrub_1"},
    {"kind": "tree", "pos": [9, 0], "sprite_key": "tree_1"},
    {"kind": "shrub", "pos": [8, 13], "sprite_key": "shrub_0"},
    {"kind": "shrub", "pos": [10, 14], "sprite_key": "shrub_1"},
    {"kind": "tree", "pos": [19, 15], "sprite_key": "tree_3"},
    {"kind": "shrub", "pos": [13, 10], "sprite_key": "shrub_1"},
    {"kind": "shrub", "pos": [14, 10], "sprite_key": "shrub_1"},
]


def _map_design(objects: list[dict[str, Any]]) -> dict[str, Any]:
    """移除未使用对象和空食物区标注，保持地图与场景对象一致。"""

    design = polarization_map_design()
    object_ids = {str(item["id"]) for item in objects}
    design["version"] = 2
    design["object_regions"] = {
        object_id: deepcopy(COMPACT_OBJECT_REGIONS[object_id])
        for object_id in sorted(object_ids)
    }
    design["tile_sprites"] = {"grass": "grass", "road": "stone_road"}
    design["decorations"] = deepcopy(COMPACT_DECORATIONS)
    regions = design.get("regions")
    if isinstance(regions, list):
        design["regions"] = [
            item for item in regions if item.get("id") != "central_food_area"
        ]
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
    """只保留 10 个实体智能体的论坛画像记忆。"""

    selected = _selected_set()
    return [
        deepcopy(item)
        for item in FULL_SPEC["memories"]
        if str(item.get("agent_id") or "") in selected
        and item.get("task") == "agent_initialization"
    ]


def _build_spec() -> dict[str, Any]:
    """从完整 IAC 场景派生 10 人可运行场景。"""

    objects = _selected_objects()
    follow_edges = _follow_edges()
    spec = deepcopy(FULL_SPEC)
    spec["name"] = "iac_gay_marriage_10"
    spec["parent_scenario"] = "iac_gay_marriage"
    spec["opinion_assessment_mode"] = "llm_as_judge"
    spec["opinion_assessment_interval"] = 5
    # 本场景固定使用已在当前实验机验证通过的本地 FLAN-T5-Large FP16 权重。
    spec["opinion_flan_model_name"] = r"D:\models\flan-t5-large"
    spec["opinion_voting_window_size"] = 10
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
    spec.pop("community_memories", None)
    spec.pop("community_assignment_rule", None)
    spec["facility_memory_scope"] = "global"
    spec["map_memory"] = (
        "IAC gay marriage 10 人子场景从完整 105 人场景派生；"
        "实体智能体缩小为 10 人，投放账号和 52 条线上投放帖子保持不变。"
    )
    return spec


SPEC: dict[str, Any] = _build_spec()


def build_runtime(**kwargs):
    """构建 IAC v2 gay marriage 10 人子场景。"""

    return build_runtime_from_spec(deepcopy(SPEC), **kwargs)
