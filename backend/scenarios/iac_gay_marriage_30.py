from __future__ import annotations

import random
from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any

from .iac_gay_marriage import (
    COMMUNITY_IDS,
    COMMUNITY_ORIGINS,
    OPPOSE_INFLUENCERS,
    SPEC as FULL_SPEC,
    SUPPORT_INFLUENCERS,
)
from .iac_gay_marriage_30_initialization import (
    CONFIRMED_INITIAL_OPINION_CORRECTIONS,
    FULL_AGENT_BY_ID,
    INITIALIZATION_AUDIT,
    ROLE_ORDER,
    SELECTED_AGENT_IDS,
    SELECTED_AGENT_IDS_BY_ROLE,
    SELECTED_AGENT_ID_SET,
    build_agent_initialization_fields,
    build_dataset_initialization_memories,
)


SAME_SIDE_MODE = "same_side_isolated"
CROSS_SIDE_MODE = "cross_side"
BALANCED_MODE = "balanced"
NETWORK_MODES = {SAME_SIDE_MODE, CROSS_SIDE_MODE, BALANCED_MODE}
ENTITY_FOLLOW_COUNT = 5
INFLUENCER_FOLLOW_COUNT = 3
BALANCED_INJECTION_PAIR_COUNT = 12
BALANCED_INJECTION_START_TICK = 2
BALANCED_INJECTION_END_TICK = 90
INJECTION_ITEM_FIELDS = frozenset(
    {
        "author_id",
        "topic",
        "content",
        "opinion_index",
        "is_rumor",
        "source_post_id",
        "source_author_id",
        "source_creation_date",
        "injection_side",
        "confidence",
    }
)
INFLUENCER_ID_SET = set(SUPPORT_INFLUENCERS + OPPOSE_INFLUENCERS)
MAP_SIZE = 100
BED_LOCAL_POSITIONS = (
    (4, 2),
    (4, 7),
    (4, 12),
    (4, 17),
    (4, 22),
    (12, 2),
    (12, 7),
    (12, 12),
    (12, 17),
    (12, 22),
)
FACILITY_LOCAL_POSITIONS = {
    "company": ((32, 4), (40, 14)),
    "food_shop": ((6, 32), (14, 38)),
    "playground": ((36, 36),),
}
SPRITE_PREFIXES = {
    "company": "company",
    "food_shop": "shop",
    "playground": "playground",
}
SPRITE_COUNTS = {"bed": 6, "company": 6, "food_shop": 6, "playground": 4}
DECORATIONS_PER_COMMUNITY = 24


def _ring_targets(members: tuple[str, ...], index: int, offsets: tuple[int, ...]) -> list[str]:
    """按固定顺序轮转目标，避免运行时随机性。"""

    return [members[(index + offset) % len(members)] for offset in offsets]


def _mixed_targets(index: int) -> list[str]:
    """让 10 名 mixed 用户在全组层面形成 25:25 的两方关注。"""

    support = SELECTED_AGENT_IDS_BY_ROLE["support"]
    oppose = SELECTED_AGENT_IDS_BY_ROLE["oppose"]
    if index % 2 == 0:
        return _ring_targets(support, index, (0, 1, 2)) + _ring_targets(oppose, index, (0, 1))
    return _ring_targets(support, index, (0, 1)) + _ring_targets(oppose, index, (0, 1, 2))


def _balanced_targets(role: str, index: int) -> list[str]:
    """用 3:2 与 2:3 交替实现全网严格平均。"""

    support = SELECTED_AGENT_IDS_BY_ROLE["support"]
    oppose = SELECTED_AGENT_IDS_BY_ROLE["oppose"]
    if role == "support":
        if index % 2 == 0:
            return _ring_targets(support, index, (1, 2, 3)) + _ring_targets(oppose, index, (0, 1))
        return _ring_targets(support, index, (1, 2)) + _ring_targets(oppose, index, (0, 1, 2))
    if role == "oppose":
        if index % 2 == 0:
            return _ring_targets(support, index, (0, 1, 2)) + _ring_targets(oppose, index, (1, 2))
        return _ring_targets(support, index, (0, 1)) + _ring_targets(oppose, index, (1, 2, 3))
    return _mixed_targets(index)


def _entity_follow_edges(mode: str) -> list[dict[str, str]]:
    """生成三种实验条件下的 150 条实体关注边。"""

    if mode not in NETWORK_MODES:
        raise ValueError(f"unsupported IAC 30 network mode: {mode}")
    support = SELECTED_AGENT_IDS_BY_ROLE["support"]
    oppose = SELECTED_AGENT_IDS_BY_ROLE["oppose"]
    mixed = SELECTED_AGENT_IDS_BY_ROLE["mixed"]
    edges: set[tuple[str, str]] = set()

    for index in range(10):
        if mode == SAME_SIDE_MODE:
            targets_by_role = {
                "support": _ring_targets(support, index, (1, 2, 3, 4, 5)),
                "oppose": _ring_targets(oppose, index, (1, 2, 3, 4, 5)),
                "mixed": _mixed_targets(index),
            }
        elif mode == CROSS_SIDE_MODE:
            targets_by_role = {
                "support": _ring_targets(oppose, index, (0, 1, 2, 3, 4)),
                "oppose": _ring_targets(support, index, (0, 1, 2, 3, 4)),
                "mixed": _mixed_targets(index),
            }
        else:
            targets_by_role = {
                role: _balanced_targets(role, index)
                for role in ROLE_ORDER
            }

        followers = {"support": support[index], "oppose": oppose[index], "mixed": mixed[index]}
        for role in ROLE_ORDER:
            follower = followers[role]
            for author in targets_by_role[role]:
                edges.add((follower, author))

    _validate_entity_follow_edges(edges)
    return [
        {"follower": follower, "author": author}
        for follower, author in sorted(edges)
    ]


def _validate_entity_follow_edges(edges: set[tuple[str, str]]) -> None:
    """在场景导入时校验实体关注实验的硬约束。"""

    if len(edges) != len(SELECTED_AGENT_IDS) * ENTITY_FOLLOW_COUNT:
        raise ValueError("IAC 30 entity follow edge count mismatch")
    counts = Counter(follower for follower, _ in edges)
    if set(counts) != SELECTED_AGENT_ID_SET or set(counts.values()) != {ENTITY_FOLLOW_COUNT}:
        raise ValueError("IAC 30 entity follow outdegree mismatch")
    if any(follower == author for follower, author in edges):
        raise ValueError("IAC 30 entity follow edges contain a self edge")
    side_targets = set(SELECTED_AGENT_IDS_BY_ROLE["support"] + SELECTED_AGENT_IDS_BY_ROLE["oppose"])
    if any(author not in side_targets for _, author in edges):
        raise ValueError("IAC 30 entity follow target is outside support/oppose groups")
    support_incoming = sum(author in SELECTED_AGENT_IDS_BY_ROLE["support"] for _, author in edges)
    oppose_incoming = sum(author in SELECTED_AGENT_IDS_BY_ROLE["oppose"] for _, author in edges)
    if (support_incoming, oppose_incoming) != (75, 75):
        raise ValueError("IAC 30 entity follow sides are not globally balanced")


def _influencer_targets(mode: str, role: str, rng: random.Random) -> tuple[str, ...]:
    """按处理组和立场返回三个投放账号。"""

    if mode not in NETWORK_MODES:
        raise ValueError(f"unsupported IAC 30 network mode: {mode}")
    if role not in ROLE_ORDER:
        raise ValueError(f"unsupported IAC 30 role: {role}")
    if mode == SAME_SIDE_MODE and role == "support":
        return tuple(SUPPORT_INFLUENCERS)
    if mode == SAME_SIDE_MODE and role == "oppose":
        return tuple(OPPOSE_INFLUENCERS)
    if mode == CROSS_SIDE_MODE and role == "support":
        return tuple(OPPOSE_INFLUENCERS)
    if mode == CROSS_SIDE_MODE and role == "oppose":
        return tuple(SUPPORT_INFLUENCERS)
    all_influencers = tuple(SUPPORT_INFLUENCERS + OPPOSE_INFLUENCERS)
    return tuple(sorted(rng.sample(all_influencers, INFLUENCER_FOLLOW_COUNT)))


def _influencer_follow_edges(mode: str) -> list[dict[str, str]]:
    """使用实验种子的独立随机状态生成 90 条投放账号关注边。"""

    rng = random.Random()
    # 复制全局种子状态，避免关注抽样改变后续仿真的随机序列。
    rng.setstate(random.getstate())
    edges: set[tuple[str, str]] = set()
    for role in ROLE_ORDER:
        for follower in SELECTED_AGENT_IDS_BY_ROLE[role]:
            for author in _influencer_targets(mode, role, rng):
                edges.add((follower, author))
    _validate_influencer_follow_edges(mode, edges)
    return [
        {"follower": follower, "author": author}
        for follower, author in sorted(edges)
    ]


def _validate_influencer_follow_edges(mode: str, edges: set[tuple[str, str]]) -> None:
    """校验投放账号关注数以及隔离组和对调组的阵营方向。"""

    expected_count = len(SELECTED_AGENT_IDS) * INFLUENCER_FOLLOW_COUNT
    if len(edges) != expected_count:
        raise ValueError("IAC 30 influencer follow edge count mismatch")
    counts = Counter(follower for follower, _ in edges)
    if set(counts) != SELECTED_AGENT_ID_SET or set(counts.values()) != {INFLUENCER_FOLLOW_COUNT}:
        raise ValueError("IAC 30 influencer follow outdegree mismatch")
    if any(author not in INFLUENCER_ID_SET for _, author in edges):
        raise ValueError("IAC 30 influencer follow target is outside influencer accounts")

    support_agents = set(SELECTED_AGENT_IDS_BY_ROLE["support"])
    oppose_agents = set(SELECTED_AGENT_IDS_BY_ROLE["oppose"])
    support_influencers = set(SUPPORT_INFLUENCERS)
    oppose_influencers = set(OPPOSE_INFLUENCERS)
    for follower, author in edges:
        if mode == SAME_SIDE_MODE:
            if follower in support_agents and author not in support_influencers:
                raise ValueError("IAC 30 same-side support agent follows an opposing influencer")
            if follower in oppose_agents and author not in oppose_influencers:
                raise ValueError("IAC 30 same-side oppose agent follows a supporting influencer")
        elif mode == CROSS_SIDE_MODE:
            if follower in support_agents and author not in oppose_influencers:
                raise ValueError("IAC 30 cross-side support agent follows a supporting influencer")
            if follower in oppose_agents and author not in support_influencers:
                raise ValueError("IAC 30 cross-side oppose agent follows an opposing influencer")


def _follow_edges(mode: str) -> list[dict[str, str]]:
    """合并实体关注边与按实验模式生成的投放账号关注边。"""

    edges = {
        (edge["follower"], edge["author"])
        for edge in _entity_follow_edges(mode)
    }
    edges.update(
        (edge["follower"], edge["author"])
        for edge in _influencer_follow_edges(mode)
    )
    return [
        {"follower": follower, "author": author}
        for follower, author in sorted(edges)
    ]


def _online_trust() -> list[dict[str, Any]]:
    """为三种处理组建立完全相同的线上信任初值。"""

    allowed_targets = SELECTED_AGENT_ID_SET | INFLUENCER_ID_SET
    edges = {
        (edge["source"], edge["target"]): deepcopy(edge)
        for edge in FULL_SPEC["online_trust"]
        if edge["source"] in SELECTED_AGENT_ID_SET and edge["target"] in allowed_targets
    }
    # 三种实体图的并集和全部投放账号都预先获得相同默认信任值。
    possible_edges = {
        (edge["follower"], edge["author"])
        for mode in sorted(NETWORK_MODES)
        for edge in _entity_follow_edges(mode)
    }
    possible_edges.update(
        (agent_id, influencer_id)
        for agent_id in SELECTED_AGENT_IDS
        for influencer_id in sorted(INFLUENCER_ID_SET)
    )
    for key in sorted(possible_edges):
        if key not in edges:
            edges[key] = {"source": key[0], "target": key[1], "value": 0.55}
    return [edges[key] for key in sorted(edges)]


def _offline_trust() -> list[dict[str, Any]]:
    """只按 30 人名单过滤，绝不按社区删除线下信任。"""

    return [
        deepcopy(edge)
        for edge in FULL_SPEC["offline_trust"]
        if edge["source"] in SELECTED_AGENT_ID_SET and edge["target"] in SELECTED_AGENT_ID_SET
    ]


def _community_assignments(mode: str) -> dict[str, str]:
    """所有处理组保留同一批用户在完整场景中的原社区。"""

    if mode not in NETWORK_MODES:
        raise ValueError(f"unsupported IAC 30 network mode: {mode}")
    return {
        agent_id: FULL_AGENT_BY_ID[agent_id]["community_id"]
        for agent_id in SELECTED_AGENT_IDS
    }


def _personal_beds(
    assignments: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """在每个社区内为智能体一一绑定现有床位。"""

    beds_by_community: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in FULL_SPEC["objects"]:
        if item["kind"] == "bed":
            beds_by_community[item["community_id"]].append(item)

    agents_by_community: dict[str, list[str]] = defaultdict(list)
    for agent_id in SELECTED_AGENT_IDS:
        agents_by_community[assignments[agent_id]].append(agent_id)

    beds: list[dict[str, Any]] = []
    bed_by_agent: dict[str, str] = {}
    for community_id in COMMUNITY_IDS:
        agent_ids = agents_by_community[community_id]
        available_beds = beds_by_community[community_id]
        if len(available_beds) < len(agent_ids):
            raise ValueError(f"community has insufficient personal beds: {community_id}")
        row_offset, col_offset = COMMUNITY_ORIGINS[community_id]
        for slot_index, (agent_id, source_bed) in enumerate(zip(agent_ids, available_beds)):
            item = deepcopy(source_bed)
            local_row, local_col = BED_LOCAL_POSITIONS[slot_index]
            row = row_offset + local_row
            col = col_offset + local_col
            item["position"] = [row, col]
            item["footprint"] = [[row, col], [row, col + 1]]
            item["sprite_key"] = f"home_{len(beds) % SPRITE_COUNTS['bed']}"
            params = dict(item.get("params") or {})
            params["owner_agent_id"] = agent_id
            item["params"] = params
            beds.append(item)
            bed_by_agent[agent_id] = item["id"]

    if set(bed_by_agent) != SELECTED_AGENT_ID_SET or len(set(bed_by_agent.values())) != 30:
        raise ValueError("IAC 30 personal bed assignment mismatch")
    return beds, bed_by_agent


def _square_footprint(row: int, col: int) -> list[list[int]]:
    """生成 2x2 设施占地。"""

    return [[row, col], [row, col + 1], [row + 1, col], [row + 1, col + 1]]


def _planned_non_bed_objects() -> list[dict[str, Any]]:
    """把四个社区的公共设施放入各自功能区。"""

    objects = [deepcopy(item) for item in FULL_SPEC["objects"] if item["kind"] != "bed"]
    sprite_indices = {kind: 0 for kind in FACILITY_LOCAL_POSITIONS}
    for community_id in COMMUNITY_IDS:
        row_offset, col_offset = COMMUNITY_ORIGINS[community_id]
        for kind, local_positions in FACILITY_LOCAL_POSITIONS.items():
            community_objects = [
                item
                for item in objects
                if item["community_id"] == community_id and item["kind"] == kind
            ]
            if len(community_objects) != len(local_positions):
                raise ValueError(f"IAC 30 facility count mismatch: {community_id}/{kind}")
            for item, (local_row, local_col) in zip(community_objects, local_positions):
                row = row_offset + local_row
                col = col_offset + local_col
                item["position"] = [row, col]
                item["footprint"] = _square_footprint(row, col)
                prefix = SPRITE_PREFIXES[kind]
                item["sprite_key"] = f"{prefix}_{sprite_indices[kind] % SPRITE_COUNTS[kind]}"
                sprite_indices[kind] += 1
    return objects


def _large_map_terrain() -> list[dict[str, Any]]:
    """为每个社区生成住宅、商业、工作和娱乐四个功能地块。"""

    terrain = [
        {
            "id": "iac_30_grass_base",
            "name": "草地底色",
            "kind": "grass",
            "bounds": [0, 0, MAP_SIZE - 1, MAP_SIZE - 1],
            "color": "#7fb069",
            "alpha": 1.0,
        }
    ]
    zone_specs = (
        ("residential", "住宅区", "yard", [0, 0, 23, 23], "#315a3b", 0.32),
        ("commercial", "商业区", "plaza", [0, 25, 23, 47], "#8a6f4d", 0.42),
        ("work", "工作区", "office_ground", [25, 0, 47, 23], "#49667f", 0.38),
        ("recreation", "娱乐区", "park", [25, 25, 47, 47], "#4f825c", 0.42),
    )
    for community_id in COMMUNITY_IDS:
        row_offset, col_offset = COMMUNITY_ORIGINS[community_id]
        for zone_id, name, kind, bounds, color, alpha in zone_specs:
            row_start, col_start, row_end, col_end = bounds
            terrain.append(
                {
                    "id": f"{community_id}_{zone_id}_ground",
                    "name": f"{community_id} {name}",
                    "kind": kind,
                    "bounds": [
                        row_offset + row_start,
                        col_offset + col_start,
                        row_offset + row_end,
                        col_offset + col_end,
                    ],
                    "color": color,
                    "alpha": alpha,
                }
            )
    return terrain


def _object_entrance(item: dict[str, Any]) -> list[int]:
    """把入口放在建筑占地外侧，供出生、寻路和交互使用。"""

    row, col = item["position"]
    if item["kind"] == "bed":
        return [row + 1, col]
    return [row, col - 1]


def _terrain_kind_at(terrain: list[dict[str, Any]], row: int, col: int) -> str:
    """按前端相同的逆序覆盖规则读取格子地形。"""

    for item in reversed(terrain):
        row_start, col_start, row_end, col_end = item["bounds"]
        if row_start <= row <= row_end and col_start <= col <= col_end:
            return str(item["kind"])
    return "grass"


def _large_map_decorations(
    design: dict[str, Any],
    objects: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """使用独立固定随机源生成四社区一致的植被密度。"""

    reserved: set[tuple[int, int]] = set()
    for item in objects:
        for row, col in item["footprint"]:
            for row_offset in (-1, 0, 1):
                for col_offset in (-1, 0, 1):
                    cell = (row + row_offset, col + col_offset)
                    if 0 <= cell[0] < MAP_SIZE and 0 <= cell[1] < MAP_SIZE:
                        reserved.add(cell)
        reserved.add(tuple(design["object_regions"][item["id"]]["entrance"]))
    reserved.update(tuple(region["label_pos"]) for region in design["regions"])
    road_cells = {
        tuple(cell)
        for road in design["roads"]
        for cell in road["cells"]
    }
    available = [
        [row, col]
        for row in range(MAP_SIZE)
        for col in range(MAP_SIZE)
        if (row, col) not in reserved
        and (row, col) not in road_cells
        and _terrain_kind_at(design["terrain"], row, col) in {"grass", "yard"}
    ]
    rng = random.Random(20260727)
    rng.shuffle(available)
    decoration_count = DECORATIONS_PER_COMMUNITY * len(COMMUNITY_IDS)
    if len(available) < decoration_count:
        raise ValueError("IAC 30 map has insufficient decoration cells")
    decorations = []
    for index, pos in enumerate(available[:decoration_count]):
        kind = "tree" if index % 3 == 0 else "shrub"
        sprite_count = 4 if kind == "tree" else 2
        decorations.append(
            {
                "kind": kind,
                "pos": pos,
                "sprite_key": f"{kind}_{rng.randrange(sprite_count)}",
            }
        )
    return decorations


def _map_design(objects: list[dict[str, Any]]) -> dict[str, Any]:
    """构建 30 人场景共享的 100x100 四社区地图。"""

    source_design = FULL_SPEC["map_design"]
    design = {
        "version": 2,
        "map_size": [MAP_SIZE, MAP_SIZE],
        "position_format": source_design["position_format"],
        "bounds_format": source_design["bounds_format"],
        "terrain": _large_map_terrain(),
        "regions": deepcopy(source_design["regions"]),
        "roads": deepcopy(source_design["roads"]),
        "object_regions": {
            item["id"]: {
                "region_id": item["community_id"],
                "entrance": _object_entrance(item),
            }
            for item in objects
        },
        "tile_sprites": {"grass": "grass", "road": "stone_road"},
    }
    design["decorations"] = _large_map_decorations(design, objects)
    return design


def _build_agents(
    assignments: dict[str, str],
    bed_by_agent: dict[str, str],
    map_design: dict[str, Any],
) -> list[dict[str, Any]]:
    """使用共享数据集初始化结果构建三组完全相同的 30 名用户。"""

    agents: list[dict[str, Any]] = []
    positions: set[tuple[int, int]] = set()
    for role in ROLE_ORDER:
        for agent_id in SELECTED_AGENT_IDS_BY_ROLE[role]:
            item = deepcopy(FULL_AGENT_BY_ID[agent_id])
            bed_id = bed_by_agent[agent_id]
            entrance = map_design["object_regions"][bed_id]["entrance"]
            position = (entrance[0], entrance[1])
            if position in positions:
                raise ValueError(f"duplicate IAC 30 spawn position: {position}")
            positions.add(position)
            item["position"] = [position[0], position[1]]
            item["community_id"] = assignments[agent_id]
            item["initial_role"] = role
            item.update(build_agent_initialization_fields(agent_id, role))
            item["personal_bed_id"] = bed_id
            item["community_movement_restricted"] = False
            agents.append(item)
    if len(agents) != 30 or {item["id"] for item in agents} != SELECTED_AGENT_ID_SET:
        raise ValueError("IAC 30 controlled agent set mismatch")
    return agents


def _memories(
    agents: list[dict[str, Any]],
    bed_by_agent: dict[str, str],
    objects: list[dict[str, Any]],
    map_design: dict[str, Any],
) -> list[dict[str, Any]]:
    """注入真实数据集初始化、社区与专属床记忆。"""

    memories: list[dict[str, Any]] = []
    object_by_id = {item["id"]: item for item in objects}
    agent_by_id = {item["id"]: item for item in agents}
    for agent_id in SELECTED_AGENT_IDS:
        agent = agent_by_id[agent_id]
        bed_id = bed_by_agent[agent_id]
        bed_item = object_by_id[bed_id]
        region = map_design["object_regions"][bed_id]
        community_id = agent["community_id"]
        memories.extend(build_dataset_initialization_memories(agent_id, str(agent["initial_role"])))
        memories.append(
            {
                "agent_id": agent_id,
                "content": (
                    f"专属床铺记忆：bed_id={bed_id}，owner_agent_id={agent_id}，"
                    f"position={bed_item['position']}，entrance={region['entrance']}，"
                    f"community_id={community_id}。你只能在这张专属床铺上调用 sleep。"
                ),
                "memory_type": "semantic",
                "task": "personal_bed_assignment",
                "source_type": "scenario_personal_bed",
                "provenance": "scenario_initialization",
                "object_id": f"personal_bed:{bed_id}",
                "episode_id": f"personal_bed:{bed_id}",
                "importance": 1.0,
                "confidence": 1.0,
            }
        )
        memories.append(
            {
                "agent_id": agent_id,
                "content": (
                    f"社区出生记忆：community_id={community_id}，initial_role={agent['initial_role']}。"
                    "社区只表示开局分区和本地设施知识；允许跨社区移动，信任关系不按社区删除。"
                ),
                "memory_type": "semantic",
                "task": "community_orientation",
                "source_type": "scenario_community_orientation",
                "provenance": "scenario_initialization",
                "object_id": f"community_role:{community_id}:{agent['initial_role']}",
                "episode_id": f"community_role:{community_id}:{agent['initial_role']}:{agent_id}",
                "importance": 0.9,
                "confidence": 1.0,
            }
        )
    opinion_counts = Counter(
        item["agent_id"]
        for item in memories
        if item["task"] == "dataset_initial_opinion"
    )
    if set(opinion_counts) != SELECTED_AGENT_ID_SET or set(opinion_counts.values()) != {1}:
        raise ValueError("IAC 30 dataset initial opinion memory count mismatch")
    return memories


def _full_injection_rows() -> list[dict[str, Any]]:
    """按原排期顺序展开完整 IAC 投放，并校验精确字段结构。"""

    rows: list[dict[str, Any]] = []
    for tick in sorted(FULL_SPEC["influencer_schedule"]):
        for item in FULL_SPEC["influencer_schedule"][tick]:
            if set(item) != INJECTION_ITEM_FIELDS:
                raise ValueError(f"IAC injection item fields changed at source tick: {tick}")
            rows.append(deepcopy(item))
    return rows


def _injection_match_metrics(item: dict[str, Any]) -> tuple[float, float, float]:
    """返回字符长度、立场强度和评分置信度三个匹配指标。"""

    return (
        float(len(str(item["content"]))),
        abs(float(item["opinion_index"])),
        float(item["confidence"]),
    )


def _paired_injection_rows() -> tuple[list[tuple[dict[str, Any], dict[str, Any], float]], tuple[float, ...]]:
    """为每条反对投放贪心匹配距离最小且尚未使用的支持投放。"""

    rows = _full_injection_rows()
    support_rows = [item for item in rows if item["injection_side"] == "support"]
    oppose_rows = [item for item in rows if item["injection_side"] == "oppose"]
    if len(support_rows) != 40 or len(oppose_rows) != BALANCED_INJECTION_PAIR_COUNT:
        raise ValueError("IAC full injection side counts changed from support=40 and oppose=12")
    if any(float(item["opinion_index"]) <= 0 for item in support_rows):
        raise ValueError("IAC support injection contains a non-positive opinion_index")
    if any(float(item["opinion_index"]) >= 0 for item in oppose_rows):
        raise ValueError("IAC oppose injection contains a non-negative opinion_index")

    all_metrics = [_injection_match_metrics(item) for item in rows]
    spans = tuple(
        max(values) - min(values)
        for values in zip(*all_metrics)
    )

    def distance(left: dict[str, Any], right: dict[str, Any]) -> float:
        left_metrics = _injection_match_metrics(left)
        right_metrics = _injection_match_metrics(right)
        return sum(
            abs(left_value - right_value) / span if span > 0 else 0.0
            for left_value, right_value, span in zip(left_metrics, right_metrics, spans)
        )

    available_support = list(support_rows)
    pairs: list[tuple[dict[str, Any], dict[str, Any], float]] = []
    for oppose_item in oppose_rows:
        support_item = min(
            available_support,
            key=lambda item: (
                distance(item, oppose_item),
                str(item["source_creation_date"]),
                str(item["source_post_id"]),
            ),
        )
        pair_distance = distance(support_item, oppose_item)
        pairs.append((support_item, oppose_item, pair_distance))
        available_support.remove(support_item)
    return pairs, spans


def _paired_injection_ticks() -> tuple[int, ...]:
    """在 tick 2 至 90 之间生成 12 个含首尾的等间隔投放时点。"""

    ticks = tuple(
        BALANCED_INJECTION_START_TICK
        + round(
            index
            * (BALANCED_INJECTION_END_TICK - BALANCED_INJECTION_START_TICK)
            / (BALANCED_INJECTION_PAIR_COUNT - 1)
        )
        for index in range(BALANCED_INJECTION_PAIR_COUNT)
    )
    if len(set(ticks)) != BALANCED_INJECTION_PAIR_COUNT:
        raise ValueError("IAC paired injection ticks are not unique")
    return ticks


def _paired_influencer_schedule() -> tuple[
    dict[int, list[dict[str, Any]]],
    list[dict[str, Any]],
    tuple[float, ...],
]:
    """构建数量、时间和匹配指标均受控的双侧投放排期。"""

    pairs, spans = _paired_injection_rows()
    ticks = _paired_injection_ticks()
    schedule: dict[int, list[dict[str, Any]]] = {}
    pair_records: list[dict[str, Any]] = []
    for pair_index, (support_source, oppose_source, distance) in enumerate(pairs):
        tick = ticks[pair_index]
        support_item = deepcopy(support_source)
        oppose_item = deepcopy(oppose_source)
        support_item["author_id"] = SUPPORT_INFLUENCERS[pair_index % len(SUPPORT_INFLUENCERS)]
        oppose_item["author_id"] = OPPOSE_INFLUENCERS[pair_index % len(OPPOSE_INFLUENCERS)]
        # 交替创建顺序，避免同一阵营始终在同 tick 内先发布。
        schedule[tick] = (
            [support_item, oppose_item]
            if pair_index % 2 == 0
            else [oppose_item, support_item]
        )
        support_metrics = _injection_match_metrics(support_item)
        oppose_metrics = _injection_match_metrics(oppose_item)
        pair_records.append(
            {
                "pair_index": pair_index,
                "tick": tick,
                "support_source_post_id": support_item["source_post_id"],
                "oppose_source_post_id": oppose_item["source_post_id"],
                "support_content_length": int(support_metrics[0]),
                "oppose_content_length": int(oppose_metrics[0]),
                "support_opinion_strength": support_metrics[1],
                "oppose_opinion_strength": oppose_metrics[1],
                "support_confidence": support_metrics[2],
                "oppose_confidence": oppose_metrics[2],
                "normalized_distance": distance,
            }
        )
    _validate_paired_influencer_schedule(schedule, pair_records)
    return schedule, pair_records, spans


def _validate_paired_influencer_schedule(
    schedule: dict[int, list[dict[str, Any]]],
    pair_records: list[dict[str, Any]],
) -> None:
    """校验 12 对投放的数量、时点、阵营、来源和账号配额。"""

    ticks = _paired_injection_ticks()
    if tuple(schedule) != ticks or len(pair_records) != BALANCED_INJECTION_PAIR_COUNT:
        raise ValueError("IAC paired injection schedule ticks or pair records mismatch")
    rows = [item for tick in ticks for item in schedule[tick]]
    if len(rows) != BALANCED_INJECTION_PAIR_COUNT * 2:
        raise ValueError("IAC paired injection total count mismatch")
    if any(set(item) != INJECTION_ITEM_FIELDS for item in rows):
        raise ValueError("IAC paired injection item fields mismatch")
    if len({item["source_post_id"] for item in rows}) != len(rows):
        raise ValueError("IAC paired injection source_post_id is duplicated")

    side_counts = Counter(item["injection_side"] for item in rows)
    if side_counts != {"support": 12, "oppose": 12}:
        raise ValueError("IAC paired injection side count mismatch")
    author_counts = Counter(item["author_id"] for item in rows)
    expected_authors = set(SUPPORT_INFLUENCERS + OPPOSE_INFLUENCERS)
    if set(author_counts) != expected_authors or set(author_counts.values()) != {4}:
        raise ValueError("IAC paired injection author quota mismatch")

    for pair_index, tick in enumerate(ticks):
        items = schedule[tick]
        if len(items) != 2 or {item["injection_side"] for item in items} != {"support", "oppose"}:
            raise ValueError(f"IAC paired injection sides mismatch at tick: {tick}")
        expected_first_side = "support" if pair_index % 2 == 0 else "oppose"
        if items[0]["injection_side"] != expected_first_side:
            raise ValueError(f"IAC paired injection creation order mismatch at tick: {tick}")
        support_item = next(item for item in items if item["injection_side"] == "support")
        oppose_item = next(item for item in items if item["injection_side"] == "oppose")
        if support_item["author_id"] not in SUPPORT_INFLUENCERS:
            raise ValueError(f"IAC support injection uses an opposing account at tick: {tick}")
        if oppose_item["author_id"] not in OPPOSE_INFLUENCERS:
            raise ValueError(f"IAC oppose injection uses a supporting account at tick: {tick}")
        if float(support_item["opinion_index"]) <= 0 or float(oppose_item["opinion_index"]) >= 0:
            raise ValueError(f"IAC paired injection opinion sign mismatch at tick: {tick}")
        record = pair_records[pair_index]
        if (
            record["tick"] != tick
            or record["support_source_post_id"] != support_item["source_post_id"]
            or record["oppose_source_post_id"] != oppose_item["source_post_id"]
        ):
            raise ValueError(f"IAC paired injection audit record mismatch at tick: {tick}")


def _community_assignment_rule(mode: str) -> str:
    """返回写入配置快照的精确社区规则。"""

    if mode not in NETWORK_MODES:
        raise ValueError(f"unsupported IAC 30 network mode: {mode}")
    return (
        "三个实体关注处理组均保留 30 名用户在 iac_gay_marriage 完整场景中的原 community_id；"
        "不限制跨社区移动，不删除跨社区信任。"
    )


def _topology_rule(mode: str) -> str:
    """返回写入配置快照的精确关注规则。"""

    if mode == SAME_SIDE_MODE:
        return (
            "每人关注 5 名实体用户；support 只关注 support，oppose 只关注 oppose，"
            "mixed 按索引奇偶交替关注 3+2 或 2+3 名两方用户。"
        )
    if mode == CROSS_SIDE_MODE:
        return (
            "每人关注 5 名实体用户；support 只关注 oppose，oppose 只关注 support，"
            "mixed 按索引奇偶交替关注 3+2 或 2+3 名两方用户。"
        )
    return (
        "每人关注 5 名实体用户；各立场组内一半采用 3 support + 2 oppose，"
        "另一半采用 2 support + 3 oppose，使全网两方各获得 75 条实体入边。"
    )


def _influencer_follow_rule(mode: str) -> str:
    """返回写入配置快照的投放账号关注规则。"""

    if mode == SAME_SIDE_MODE:
        return (
            "每人关注 3 个投放账号；support 关注全部 3 个 support 投放账号，"
            "oppose 关注全部 3 个 oppose 投放账号，mixed 使用实验种子从 6 个账号中随机抽取 3 个。"
        )
    if mode == CROSS_SIDE_MODE:
        return (
            "每人关注 3 个投放账号；support 关注全部 3 个 oppose 投放账号，"
            "oppose 关注全部 3 个 support 投放账号，mixed 使用实验种子从 6 个账号中随机抽取 3 个。"
        )
    if mode == BALANCED_MODE:
        return "每人使用实验种子从全部 6 个投放账号中随机抽取 3 个。"
    raise ValueError(f"unsupported IAC 30 network mode: {mode}")


def build_spec(name: str, mode: str) -> dict[str, Any]:
    """构建共享同一批 30 人的 IAC 极化实验场景。"""

    if not isinstance(name, str) or not name:
        raise ValueError("IAC 30 scenario name must be a non-empty string")
    if mode not in NETWORK_MODES:
        raise ValueError(f"unsupported IAC 30 network mode: {mode}")

    assignments = _community_assignments(mode)
    beds, bed_by_agent = _personal_beds(assignments)
    non_bed_objects = _planned_non_bed_objects()
    objects = beds + non_bed_objects
    map_design = _map_design(objects)
    agents = _build_agents(assignments, bed_by_agent, map_design)
    follow_edges = _follow_edges(mode)
    influencer_schedule, injection_pair_records, injection_metric_spans = _paired_influencer_schedule()

    spec = deepcopy(FULL_SPEC)
    spec["name"] = name
    spec["parent_scenario"] = "iac_gay_marriage"
    spec["opinion_assessment_mode"] = "llm_as_judge"
    spec["opinion_assessment_interval"] = 5
    # 30 人实验沿用已在当前实验机验证的本地权重目录。
    spec["opinion_flan_model_name"] = r"D:\models\flan-t5-large"
    spec["opinion_voting_window_size"] = 10
    spec["selection_rule"] = str(INITIALIZATION_AUDIT["selection_rule"])
    spec["initial_opinion_corrections"] = deepcopy(CONFIRMED_INITIAL_OPINION_CORRECTIONS)
    spec["agent_profile_rule"] = (
        "只依据当前同性婚姻讨论中初始化时点前保留的原帖构造画像；"
        "保留结构化 raw_author、可追溯表达样本、论证摘要、社交统计和逐帖话题记忆。"
    )
    spec["dataset_initial_opinion_memory_task"] = "dataset_initial_opinion"
    spec["dataset_initial_opinion_memory_rule"] = (
        "支持与反对用户的初始观念由当前讨论最后评分帖子支撑；"
        "不确定用户在初始化前没有任何 IAC v2 发帖，初始观念固定为 0。"
    )
    spec["initialization_audit"] = deepcopy(INITIALIZATION_AUDIT)
    spec["history_scope_rule"] = str(INITIALIZATION_AUDIT["history_scope_rule"])
    spec["source_topic_post_count"] = int(INITIALIZATION_AUDIT["source_topic_post_count"])
    spec["retained_topic_post_count"] = int(INITIALIZATION_AUDIT["retained_topic_post_count"])
    spec["max_history_posts_per_agent"] = int(INITIALIZATION_AUDIT["max_history_posts_per_agent"])
    spec["network_mode"] = mode
    spec["controlled_variable"] = "entity_and_influencer_follow_edges"
    spec["topology_rule"] = _topology_rule(mode)
    spec["influencer_follow_rule"] = _influencer_follow_rule(mode)
    spec["online_trust_rule"] = "三个处理组共享三种实体图并集及全部投放账号的相同初始线上信任。"
    spec["entity_follow_count"] = ENTITY_FOLLOW_COUNT
    spec["influencer_follow_count"] = INFLUENCER_FOLLOW_COUNT
    spec["influencer_follow_randomization"] = "复制实验种子状态后独立抽样，不推进仿真全局随机状态。"
    spec["entity_follow_balance_scope"] = "global_support_75_oppose_75"
    spec["agent_ids"] = list(SELECTED_AGENT_IDS)
    spec["agents"] = agents
    spec["objects"] = objects
    spec["map_design"] = map_design
    spec["community_assignment_rule"] = _community_assignment_rule(mode)
    spec["community_movement_restricted"] = False
    spec["cross_community_trust_preserved"] = True
    spec["offline_trust"] = _offline_trust()
    spec["follow_edges"] = follow_edges
    spec["online_trust"] = _online_trust()
    spec["memories"] = _memories(agents, bed_by_agent, objects, map_design)
    spec["influencer_schedule"] = influencer_schedule
    spec["injection_balance_rule"] = (
        "保留全部 12 条 oppose 投放，并从 40 条 support 投放中一对一匹配 12 条；"
        "每对在同一 tick 发布，12 个 tick 在 2 至 90 之间含首尾等间隔分布。"
    )
    spec["injection_pair_count"] = BALANCED_INJECTION_PAIR_COUNT
    spec["injection_pair_ticks"] = list(_paired_injection_ticks())
    spec["injection_matching_method"] = (
        "按原 oppose 排期顺序，为每条 oppose 贪心选择尚未使用且归一化距离最小的 support；"
        "距离为字符长度、abs(opinion_index)、confidence 三项归一化绝对差之和。"
    )
    spec["injection_matching_metric_spans"] = {
        "content_character_length": injection_metric_spans[0],
        "absolute_opinion_index": injection_metric_spans[1],
        "confidence": injection_metric_spans[2],
    }
    spec["injection_pair_records"] = injection_pair_records
    spec["injection_creation_order_rule"] = "同 tick 内按 pair_index 奇偶交替先创建 support 或 oppose。"
    spec["injection_account_quota_rule"] = "六个投放账号各发布 4 条；每方合计 12 条。"
    spec["facility_memory_scope"] = "community"
    spec.pop("map_memory", None)
    return spec
