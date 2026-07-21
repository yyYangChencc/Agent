from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any

from persona.opinion.scale import clamp_opinion

from .iac_gay_marriage import (
    COMMUNITY_IDS,
    OPPOSE_INFLUENCERS,
    SEED_AGENTS,
    SPEC as FULL_SPEC,
    SUPPORT_INFLUENCERS,
)


SAME_SIDE_MODE = "same_side_isolated"
CROSS_SIDE_MODE = "cross_side"
BALANCED_MODE = "balanced"
NETWORK_MODES = {SAME_SIDE_MODE, CROSS_SIDE_MODE, BALANCED_MODE}
ROLE_ORDER = ("support", "oppose", "mixed")
ENTITY_FOLLOW_COUNT = 5
ROLE_COMMUNITIES = {
    "support": "community_1",
    "oppose": "community_2",
    "mixed": "community_3",
}


def _role_for_opinion(opinion: float) -> str:
    """沿用完整 IAC 场景阈值划分三类初始立场。"""

    score = clamp_opinion(float(opinion))
    if score > 0.35:
        return "support"
    if score < -0.35:
        return "oppose"
    return "mixed"


def _selection_key(row: dict[str, Any], role: str) -> tuple[Any, ...]:
    """按已确认的数据字段生成稳定抽取顺序。"""

    agent_id = row["agent_id"]
    if not isinstance(agent_id, str) or not agent_id:
        raise ValueError("agents[].agent_id must be a non-empty string")
    scored_count = row["pre_t_scored_post_count"]
    history_count = row["history_post_count_before_t"]
    if isinstance(scored_count, bool) or not isinstance(scored_count, int):
        raise ValueError(f"pre_t_scored_post_count must be an integer: {agent_id}")
    if isinstance(history_count, bool) or not isinstance(history_count, int):
        raise ValueError(f"history_post_count_before_t must be an integer: {agent_id}")
    base = (-scored_count, -history_count, agent_id)
    if role == "mixed":
        return (abs(clamp_opinion(float(row["initial_opinion"]))), *base)
    return base


def _select_agent_ids_by_role() -> dict[str, tuple[str, ...]]:
    """从 IAC 初始化种子中固定抽取每类立场各 10 人。"""

    grouped: dict[str, list[dict[str, Any]]] = {role: [] for role in ROLE_ORDER}
    for row in SEED_AGENTS:
        role = _role_for_opinion(row["initial_opinion"])
        grouped[role].append(row)

    selected: dict[str, tuple[str, ...]] = {}
    for role in ROLE_ORDER:
        ordered = sorted(grouped[role], key=lambda row: _selection_key(row, role))
        if len(ordered) < 10:
            raise ValueError(f"IAC seed has fewer than 10 agents for role: {role}")
        selected[role] = tuple(row["agent_id"] for row in ordered[:10])
    return selected


SELECTED_AGENT_IDS_BY_ROLE = _select_agent_ids_by_role()
SELECTED_AGENT_IDS = tuple(
    agent_id
    for role in ROLE_ORDER
    for agent_id in SELECTED_AGENT_IDS_BY_ROLE[role]
)
SELECTED_AGENT_ID_SET = set(SELECTED_AGENT_IDS)
INFLUENCER_ID_SET = set(SUPPORT_INFLUENCERS + OPPOSE_INFLUENCERS)
FULL_AGENT_BY_ID = {item["id"]: item for item in FULL_SPEC["agents"]}


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


def _influencer_targets(role: str, mode: str) -> tuple[str, ...]:
    """按实体网络条件设置投放账号关注关系。"""

    support = tuple(SUPPORT_INFLUENCERS)
    oppose = tuple(OPPOSE_INFLUENCERS)
    if role == "mixed" or mode == BALANCED_MODE:
        return support + oppose
    if mode == SAME_SIDE_MODE:
        return support if role == "support" else oppose
    if mode == CROSS_SIDE_MODE:
        return oppose if role == "support" else support
    raise ValueError(f"unsupported IAC 30 network mode: {mode}")


def _follow_edges(mode: str) -> list[dict[str, str]]:
    """合并 5 条实体边与单独统计的投放账号边。"""

    edges = {
        (edge["follower"], edge["author"])
        for edge in _entity_follow_edges(mode)
    }
    for role in ROLE_ORDER:
        for follower in SELECTED_AGENT_IDS_BY_ROLE[role]:
            for author in _influencer_targets(role, mode):
                edges.add((follower, author))
    return [
        {"follower": follower, "author": author}
        for follower, author in sorted(edges)
    ]


def _online_trust(follow_edges: list[dict[str, str]]) -> list[dict[str, Any]]:
    """保留已存在的线上信任，并为新增实体关注边补充默认值。"""

    allowed_targets = SELECTED_AGENT_ID_SET | INFLUENCER_ID_SET
    edges = {
        (edge["source"], edge["target"]): deepcopy(edge)
        for edge in FULL_SPEC["online_trust"]
        if edge["source"] in SELECTED_AGENT_ID_SET and edge["target"] in allowed_targets
    }
    for edge in follow_edges:
        key = (edge["follower"], edge["author"])
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
    """情景 1 按立场分区，其余情景保留完整场景原社区。"""

    assignments: dict[str, str] = {}
    for role in ROLE_ORDER:
        for agent_id in SELECTED_AGENT_IDS_BY_ROLE[role]:
            if mode == SAME_SIDE_MODE:
                assignments[agent_id] = ROLE_COMMUNITIES[role]
            else:
                assignments[agent_id] = FULL_AGENT_BY_ID[agent_id]["community_id"]
    return assignments


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
        for agent_id, source_bed in zip(agent_ids, available_beds):
            item = deepcopy(source_bed)
            params = dict(item.get("params") or {})
            params["owner_agent_id"] = agent_id
            item["params"] = params
            beds.append(item)
            bed_by_agent[agent_id] = item["id"]

    if set(bed_by_agent) != SELECTED_AGENT_ID_SET or len(set(bed_by_agent.values())) != 30:
        raise ValueError("IAC 30 personal bed assignment mismatch")
    return beds, bed_by_agent


def _map_design(objects: list[dict[str, Any]]) -> dict[str, Any]:
    """复用完整地图并移除未实例化床位的区域条目。"""

    design = deepcopy(FULL_SPEC["map_design"])
    object_ids = {item["id"] for item in objects}
    design["object_regions"] = {
        object_id: deepcopy(region)
        for object_id, region in design["object_regions"].items()
        if object_id in object_ids
    }
    return design


def _build_agents(
    assignments: dict[str, str],
    bed_by_agent: dict[str, str],
    map_design: dict[str, Any],
) -> list[dict[str, Any]]:
    """复制 30 名数据集用户并出生在各自床铺入口。"""

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
            item["personal_bed_id"] = bed_id
            item["community_movement_restricted"] = False
            agents.append(item)
    return agents


def _memories(
    agents: list[dict[str, Any]],
    bed_by_agent: dict[str, str],
    objects: list[dict[str, Any]],
    map_design: dict[str, Any],
) -> list[dict[str, Any]]:
    """保留个人画像，并注入精确的社区与专属床记忆。"""

    memories = [
        deepcopy(item)
        for item in FULL_SPEC["memories"]
        if item["agent_id"] in SELECTED_AGENT_ID_SET and item.get("task") == "agent_initialization"
    ]
    object_by_id = {item["id"]: item for item in objects}
    agent_by_id = {item["id"]: item for item in agents}
    for agent_id in SELECTED_AGENT_IDS:
        agent = agent_by_id[agent_id]
        bed_id = bed_by_agent[agent_id]
        bed_item = object_by_id[bed_id]
        region = map_design["object_regions"][bed_id]
        community_id = agent["community_id"]
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
                "object_id": f"personal_bed:{bed_id}",
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
                "object_id": f"community_role:{community_id}:{agent['initial_role']}",
                "importance": 0.9,
                "confidence": 1.0,
            }
        )
    return memories


def _community_assignment_rule(mode: str) -> str:
    """返回写入配置快照的精确社区规则。"""

    if mode == SAME_SIDE_MODE:
        return (
            "support 固定出生于 community_1，oppose 固定出生于 community_2，"
            "mixed 固定出生于 community_3；不限制跨社区移动，不删除跨社区信任。"
        )
    return "保留 30 名用户在 iac_gay_marriage 完整场景中的原 community_id；不限制跨社区移动。"


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


def build_spec(name: str, mode: str) -> dict[str, Any]:
    """构建共享同一批 30 人的 IAC 极化实验场景。"""

    if not isinstance(name, str) or not name:
        raise ValueError("IAC 30 scenario name must be a non-empty string")
    if mode not in NETWORK_MODES:
        raise ValueError(f"unsupported IAC 30 network mode: {mode}")

    assignments = _community_assignments(mode)
    beds, bed_by_agent = _personal_beds(assignments)
    non_bed_objects = [deepcopy(item) for item in FULL_SPEC["objects"] if item["kind"] != "bed"]
    objects = beds + non_bed_objects
    map_design = _map_design(objects)
    agents = _build_agents(assignments, bed_by_agent, map_design)
    follow_edges = _follow_edges(mode)

    spec = deepcopy(FULL_SPEC)
    spec["name"] = name
    spec["parent_scenario"] = "iac_gay_marriage"
    spec["opinion_assessment_mode"] = "llm_as_judge"
    spec["opinion_assessment_interval"] = 5
    # 30 人实验沿用已在当前实验机验证的本地权重目录。
    spec["opinion_flan_model_name"] = r"D:\models\flan-t5-large"
    spec["opinion_voting_window_size"] = 10
    spec["selection_rule"] = (
        "从 IAC 初始化种子固定抽取 10 support、10 oppose、10 mixed；"
        "support/oppose 按 pre_t_scored_post_count 降序、history_post_count_before_t 降序、agent_id 升序；"
        "mixed 先按 abs(initial_opinion) 升序，再使用相同资料字段排序。"
    )
    spec["network_mode"] = mode
    spec["topology_rule"] = _topology_rule(mode)
    spec["entity_follow_count"] = ENTITY_FOLLOW_COUNT
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
    spec["online_trust"] = _online_trust(follow_edges)
    spec["memories"] = _memories(agents, bed_by_agent, objects, map_design)
    spec["facility_memory_scope"] = "community"
    spec.pop("map_memory", None)
    return spec
