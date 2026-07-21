from __future__ import annotations

import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

from persona.opinion.scale import clamp_opinion

from .builder import build_runtime_from_spec
from .map_designs import IAC_COMMUNITIES, iac_community_map_design


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "iac_v2" / "llm_pipeline" / "gay_marriage_1q"
SEED_PATH = DATA_DIR / "agent_initialization_seed_1q.json"
METADATA_PATH = DATA_DIR / "selected_discussion_metadata.json"
INJECTION_PATH = DATA_DIR / "injection_posts_direct.jsonl"

SUPPORT_INFLUENCERS = ["iac_gm_support_1", "iac_gm_support_2", "iac_gm_support_3"]
OPPOSE_INFLUENCERS = ["iac_gm_oppose_1", "iac_gm_oppose_2", "iac_gm_oppose_3"]
COMMUNITY_IDS = tuple(item["id"] for item in IAC_COMMUNITIES)
COMMUNITY_ORIGINS = {
    "community_1": (0, 0),
    "community_2": (0, 52),
    "community_3": (52, 0),
    "community_4": (52, 52),
}
ROLE_COMMUNITY_OFFSETS = {"support": 0, "oppose": 3, "mixed": 2}


def _read_json(path: Path) -> dict[str, Any]:
    """读取场景依赖的 JSON 文件，缺失时直接暴露精确路径。"""

    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取投放帖子 JSONL，每行保留原始字段。"""

    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


METADATA = _read_json(METADATA_PATH)
SEED_DATA = _read_json(SEED_PATH)
SEED_AGENTS = list(SEED_DATA["agents"])
INJECTION_ROWS = _read_jsonl(INJECTION_PATH)
TOPIC = str(METADATA["discussion_title"])
DISCUSSION_ID = str(METADATA["discussion_id"])
DATASET = str(METADATA["dataset"])
QUARTER_TIME = str(METADATA["quarter_time"])
STANCE_DEFINITION = str(METADATA["stance_definition"])


def _safe_float(value: Any, default: float = 0.0) -> float:
    """把数据集数值字段转为 float，失败时使用明确默认值。"""

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    """把 author_id 等字段转为 int，用于稳定排序和资源扰动。"""

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _objects_and_regions() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """为四个社区生成完全对称的设施与入口。"""

    objects: list[dict[str, Any]] = []
    object_regions: dict[str, dict[str, Any]] = {}
    bed_index = company_index = shop_index = playground_index = 0
    bed_positions = [(4, 4), (4, 12), (4, 20), (4, 28), (4, 36),
                     (10, 4), (10, 12), (10, 20), (10, 28), (10, 36)]
    company_specs = [((34, 8), 8, 8), ((40, 16), 12, 12)]
    shop_specs = [((8, 36), 100, 30, 6), ((16, 40), 100, 20, 4)]
    playground_position = (38, 38)

    for community_id in COMMUNITY_IDS:
        row_offset, col_offset = COMMUNITY_ORIGINS[community_id]
        for local_row, local_col in bed_positions:
            bed_index += 1
            object_id = f"bed_{bed_index}"
            position = [row_offset + local_row, col_offset + local_col]
            objects.append({"kind": "bed", "id": object_id, "position": position, "community_id": community_id})
            object_regions[object_id] = {
                "region_id": community_id,
                "entrance": [position[0], position[1] + 1],
            }
        for (local_row, local_col), salary, relax_cost in company_specs:
            company_index += 1
            object_id = f"company_{company_index}"
            position = [row_offset + local_row, col_offset + local_col]
            objects.append(
                {
                    "kind": "company",
                    "id": object_id,
                    "position": position,
                    "community_id": community_id,
                    "params": {"salary": salary, "relax_cost": relax_cost},
                }
            )
            object_regions[object_id] = {
                "region_id": community_id,
                "entrance": [position[0], position[1] + 1],
            }
        for (local_row, local_col), food_num, provide, price in shop_specs:
            shop_index += 1
            object_id = f"shop_{shop_index}"
            position = [row_offset + local_row, col_offset + local_col]
            objects.append(
                {
                    "kind": "food_shop",
                    "id": object_id,
                    "position": position,
                    "community_id": community_id,
                    "params": {"food_num": food_num, "provide": provide, "price": price},
                }
            )
            object_regions[object_id] = {
                "region_id": community_id,
                "entrance": [position[0], position[1] - 1],
            }
        playground_index += 1
        object_id = f"playground_{playground_index}"
        position = [row_offset + playground_position[0], col_offset + playground_position[1]]
        objects.append(
            {
                "kind": "playground",
                "id": object_id,
                "position": position,
                "community_id": community_id,
                "params": {"provide": 18, "price": 5},
            }
        )
        object_regions[object_id] = {
            "region_id": community_id,
            "entrance": [position[0], position[1] - 1],
        }
    return objects, object_regions


OBJECTS, OBJECT_REGIONS = _objects_and_regions()


def _map_design() -> dict[str, Any]:
    """返回完整场景专用的 100x100 四社区地图。"""

    return iac_community_map_design(OBJECT_REGIONS)


def _occupied_object_positions() -> set[tuple[int, int]]:
    """收集物体格点，避免初始化实体智能体时被物体覆盖。"""

    return {tuple(item["position"]) for item in OBJECTS}


def _initial_role(opinion: float) -> str:
    """沿用场景现有阈值，把初始观念映射为三个阵营。"""

    if opinion > 0.35:
        return "support"
    if opinion < -0.35:
        return "oppose"
    return "mixed"


def _agent_assignments() -> dict[str, dict[str, str]]:
    """按阵营分别轮转到四社区，保证总人数和阵营人数均衡。"""

    role_indexes: dict[str, int] = defaultdict(int)
    assignments: dict[str, dict[str, str]] = {}
    for row in SEED_AGENTS:
        agent_id = str(row["agent_id"])
        opinion = clamp_opinion(_safe_float(row.get("initial_opinion")))
        role = _initial_role(opinion)
        community_index = (role_indexes[role] + ROLE_COMMUNITY_OFFSETS[role]) % len(COMMUNITY_IDS)
        assignments[agent_id] = {
            "community_id": COMMUNITY_IDS[community_index],
            "initial_role": role,
        }
        role_indexes[role] += 1
    return assignments


AGENT_ASSIGNMENTS = _agent_assignments()


def _community_agent_positions() -> dict[str, list[list[int]]]:
    """在每个社区内生成分散且不占用设施的固定出生点。"""

    occupied = _occupied_object_positions()
    positions: dict[str, list[list[int]]] = {}
    for community_id in COMMUNITY_IDS:
        row_offset, col_offset = COMMUNITY_ORIGINS[community_id]
        cells = []
        for local_row in range(2, 43, 8):
            for local_col in range(2, 43, 8):
                position = (row_offset + local_row, col_offset + local_col)
                if position not in occupied:
                    cells.append([position[0], position[1]])
        positions[community_id] = cells
    return positions


AGENT_POSITIONS_BY_COMMUNITY = _community_agent_positions()


def _speaking_style(row: dict[str, Any]) -> str:
    """把用户历史总结合并成智能体说话风格。"""

    parts = [
        f"发言风格：{row.get('speaking_style') or ''}",
        f"论证风格：{row.get('argument_style') or ''}",
        f"社交风格：{row.get('social_style') or ''}",
    ]
    return "\n".join(part for part in parts if part.strip())


def _initial_money(row: dict[str, Any]) -> float:
    """用 source_author_id 做稳定扰动，避免所有智能体资源完全相同。"""

    return 12.0 + float(_safe_int(row.get("source_author_id")) % 7)


def _build_agents() -> list[dict[str, Any]]:
    """把 IAC 初始化种子转成场景实体智能体。"""

    agents: list[dict[str, Any]] = []
    community_position_indexes: dict[str, int] = defaultdict(int)
    for row in SEED_AGENTS:
        agent_id = str(row["agent_id"])
        assignment = AGENT_ASSIGNMENTS[agent_id]
        community_id = assignment["community_id"]
        position_index = community_position_indexes[community_id]
        available_positions = AGENT_POSITIONS_BY_COMMUNITY[community_id]
        if position_index >= len(available_positions):
            raise ValueError(f"community has insufficient spawn positions: {community_id}")
        initial_opinion = clamp_opinion(_safe_float(row.get("initial_opinion")))
        agents.append(
            {
                "id": agent_id,
                "position": list(available_positions[position_index]),
                "community_id": community_id,
                "initial_role": assignment["initial_role"],
                "speaking_style": _speaking_style(row),
                "initial_opinion": initial_opinion,
                "initial_money": _initial_money(row),
                "source_author_id": str(row.get("source_author_id") or ""),
                "discussion_title": str(row.get("discussion_title") or ""),
                "discussion_id": str(row.get("discussion_id") or ""),
                "dataset": str(row.get("dataset") or ""),
                "initialization_time": str(row.get("initialization_time") or ""),
            }
        )
        community_position_indexes[community_id] += 1
    return agents


AGENTS = _build_agents()
AGENT_IDS = [item["id"] for item in AGENTS]


def _agent_opinion(agent_id: str) -> float:
    """读取场景智能体初始观念。"""

    for item in AGENTS:
        if item["id"] == agent_id:
            return float(item["initial_opinion"])
    return 0.0


def _influencers() -> list[dict[str, Any]]:
    """注册无地图实体的投放账号。"""

    return [
        {"id": "iac_gm_support_1", "camp": "support", "name": "IAC Gay Marriage Support 1", "stance": 0.95},
        {"id": "iac_gm_support_2", "camp": "support", "name": "IAC Gay Marriage Support 2", "stance": 0.90},
        {"id": "iac_gm_support_3", "camp": "support", "name": "IAC Gay Marriage Support 3", "stance": 0.85},
        {"id": "iac_gm_oppose_1", "camp": "oppose", "name": "IAC Gay Marriage Oppose 1", "stance": -0.95},
        {"id": "iac_gm_oppose_2", "camp": "oppose", "name": "IAC Gay Marriage Oppose 2", "stance": -0.90},
        {"id": "iac_gm_oppose_3", "camp": "oppose", "name": "IAC Gay Marriage Oppose 3", "stance": -0.85},
    ]


def _influencer_ids_for_agent(opinion: float) -> list[str]:
    """按初始立场设置选择性关注。"""

    if opinion > 0.35:
        return SUPPORT_INFLUENCERS + [OPPOSE_INFLUENCERS[0]]
    if opinion < -0.35:
        return OPPOSE_INFLUENCERS + [SUPPORT_INFLUENCERS[0]]
    return SUPPORT_INFLUENCERS + OPPOSE_INFLUENCERS


def _follow_edges() -> list[dict[str, str]]:
    """生成线上关注关系，包含投放账号和少量普通用户互关。"""

    edges: set[tuple[str, str]] = set()
    for item in AGENTS:
        follower = item["id"]
        opinion = float(item["initial_opinion"])
        for author in _influencer_ids_for_agent(opinion):
            edges.add((follower, author))

    for index, follower in enumerate(AGENT_IDS):
        for offset in (1, 2):
            edges.add((follower, AGENT_IDS[(index + offset) % len(AGENT_IDS)]))

    groups: dict[str, list[str]] = defaultdict(list)
    for item in AGENTS:
        opinion = float(item["initial_opinion"])
        if opinion > 0.35:
            groups["support"].append(item["id"])
        elif opinion < -0.35:
            groups["oppose"].append(item["id"])
        else:
            groups["mixed"].append(item["id"])
    for members in groups.values():
        for index, follower in enumerate(members):
            if len(members) > 1:
                edges.add((follower, members[(index + 1) % len(members)]))

    return [{"follower": follower, "author": author} for follower, author in sorted(edges)]


def _trust_for_influencer(agent_opinion: float, influencer_id: str) -> float:
    """按立场接近度初始化投放账号线上信任。"""

    is_support = influencer_id in SUPPORT_INFLUENCERS
    if agent_opinion > 0.35:
        return 0.75 if is_support else 0.30
    if agent_opinion < -0.35:
        return 0.30 if is_support else 0.75
    return 0.55


def _online_trust() -> list[dict[str, Any]]:
    """生成线上信任边，投放账号与普通用户都保留可审计权重。"""

    edges: list[dict[str, Any]] = []
    influencer_ids = SUPPORT_INFLUENCERS + OPPOSE_INFLUENCERS
    for item in AGENTS:
        source = item["id"]
        opinion = float(item["initial_opinion"])
        for influencer_id in influencer_ids:
            edges.append({"source": source, "target": influencer_id, "value": _trust_for_influencer(opinion, influencer_id)})
    for edge in _follow_edges():
        if edge["author"] in influencer_ids:
            continue
        edges.append({"source": edge["follower"], "target": edge["author"], "value": 0.55})
    return edges


def _offline_trust() -> list[dict[str, Any]]:
    """生成线下熟人关系，避免 105 人完全互相隔离。"""

    edges: set[tuple[str, str, float]] = set()
    for index, source in enumerate(AGENT_IDS):
        for offset in (1, 2):
            target = AGENT_IDS[(index + offset) % len(AGENT_IDS)]
            edges.add((source, target, 0.45))
            edges.add((target, source, 0.45))

    groups: dict[str, list[str]] = defaultdict(list)
    for item in AGENTS:
        opinion = float(item["initial_opinion"])
        if opinion > 0.35:
            groups["support"].append(item["id"])
        elif opinion < -0.35:
            groups["oppose"].append(item["id"])
        else:
            groups["mixed"].append(item["id"])
    for members in groups.values():
        for index, source in enumerate(members):
            if len(members) > 1:
                target = members[(index + 1) % len(members)]
                edges.add((source, target, 0.60))
                edges.add((target, source, 0.60))

    return [{"source": source, "target": target, "value": value} for source, target, value in sorted(edges)]


def _official_news_schedule() -> dict[int, dict[str, Any]]:
    """在第 1 tick 投放中性议题说明，使所有智能体获得系统关注主题。"""

    return {
        1: {
            "topic": TOPIC,
            "title": f"IAC v2 discussion {DISCUSSION_ID}: {TOPIC}",
            "content": (
                f"The FourForums discussion asks participants to state whether they are for or against gay marriage. "
                f"The simulation starts at the first-quarter time {QUARTER_TIME}; later injected posts reproduce selected "
                f"supporting and opposing arguments without declaring an official ground truth."
            ),
            "opinion_index": 0.0,
        }
    }


def _influencer_for_row(row: dict[str, Any], side_index: dict[str, int]) -> str:
    """按投放阵营轮换无实体账号。"""

    side = str(row["injection_side"])
    if side == "support":
        accounts = SUPPORT_INFLUENCERS
    elif side == "oppose":
        accounts = OPPOSE_INFLUENCERS
    else:
        raise ValueError(f"unsupported injection_side: {side}")
    index = side_index[side]
    side_index[side] += 1
    return accounts[index % len(accounts)]


def _influencer_schedule() -> dict[int, list[dict[str, Any]]]:
    """把筛选后的 IAC 帖子分布到 100 tick 实验窗口内。"""

    rows = sorted(INJECTION_ROWS, key=lambda row: str(row.get("creation_date") or ""))
    schedule: dict[int, list[dict[str, Any]]] = defaultdict(list)
    side_index: dict[str, int] = defaultdict(int)
    max_index = max(1, len(rows) - 1)
    for index, row in enumerate(rows):
        tick = 2 + round(index * 88 / max_index)
        schedule[tick].append(
            {
                "author_id": _influencer_for_row(row, side_index),
                "topic": TOPIC,
                "content": str(row["text"]),
                "opinion_index": clamp_opinion(_safe_float(row.get("stance_score"))),
                "is_rumor": False,
                "source_post_id": str(row.get("post_id") or ""),
                "source_author_id": str(row.get("author_id") or ""),
                "source_creation_date": str(row.get("creation_date") or ""),
                "injection_side": str(row.get("injection_side") or ""),
                "confidence": _safe_float(row.get("confidence"), 0.0),
            }
        )
    return dict(sorted(schedule.items()))


def _memory_content(row: dict[str, Any]) -> str:
    """把初始化管线输出压缩为一条结构化初始记忆。"""

    memory_lines = [str(item) for item in (row.get("memory_seeds") or [])]
    return "\n".join(
        [
            f"IAC v2 用户初始化：agent_id={row.get('agent_id')}, source_author_id={row.get('source_author_id')}",
            f"初始化时间：{row.get('initialization_time')}",
            f"目标讨论：{row.get('discussion_title')} / discussion_id={row.get('discussion_id')}",
            f"初始观念：{row.get('initial_opinion')}，来源：{row.get('initial_opinion_source')}",
            f"历史发言数量：{row.get('history_post_count_before_t')}，目标讨论前置评分数量：{row.get('pre_t_scored_post_count')}",
            f"风格摘要：{_speaking_style(row)}",
            "历史记忆种子：",
            *[f"- {line}" for line in memory_lines],
        ]
    )


def _community_memories() -> dict[str, str]:
    """为每个社区生成只描述本地范围的地图记忆。"""

    memories: dict[str, str] = {}
    for community in IAC_COMMUNITIES:
        community_id = str(community["id"])
        memories[community_id] = (
            f"本社区地图记忆：community_id={community_id}，名称={community['name']}，"
            f"边界={community['bounds']}，坐标格式为 [row, col]。"
            "你的开局地理知识只覆盖本社区；社区内具备居住、工作、食品购买和娱乐设施。"
            "未在本社区地图记忆或设施记忆中出现的位置与设施均属于未知信息。"
        )
    return memories


def _memories() -> list[dict[str, Any]]:
    """注入个人论坛画像和本社区角色记忆。"""

    memories: list[dict[str, Any]] = []
    for row in SEED_AGENTS:
        agent_id = str(row["agent_id"])
        assignment = AGENT_ASSIGNMENTS[agent_id]
        confidence = _safe_float(row.get("profile_confidence"), 0.7)
        memories.append(
            {
                "agent_id": agent_id,
                "content": _memory_content(row),
                "memory_type": "semantic",
                "task": "agent_initialization",
                "object_id": f"iac_profile_{row.get('source_author_id')}",
                "importance": 0.9,
                "confidence": confidence,
            }
        )
        memories.append(
            {
                "agent_id": agent_id,
                "content": (
                    f"本地角色记忆：community_id={assignment['community_id']}，"
                    f"initial_role={assignment['initial_role']}。"
                    "你从本社区开始生活，地理行动应优先使用本社区地图和设施记忆。"
                ),
                "memory_type": "semantic",
                "task": "community_orientation",
                "object_id": f"community_role:{assignment['community_id']}:{assignment['initial_role']}",
                "importance": 0.9,
                "confidence": 1.0,
            }
        )
    return memories


SPEC: dict[str, Any] = {
    "name": "iac_gay_marriage",
    "agent_ids": AGENT_IDS,
    "default_opinion_topic": TOPIC,
    "dataset": DATASET,
    "discussion_id": DISCUSSION_ID,
    "discussion_title": TOPIC,
    "quarter_time": QUARTER_TIME,
    "stance_definition": STANCE_DEFINITION,
    "source_paths": {
        "agent_initialization_seed": str(SEED_PATH),
        "selected_discussion_metadata": str(METADATA_PATH),
        "injection_posts_direct": str(INJECTION_PATH),
    },
    "map_design": _map_design(),
    "default_initial_satisfaction": {"satiety": 55.0, "relax": 55.0},
    "agents": AGENTS,
    "offline_trust": _offline_trust(),
    "online_trust": _online_trust(),
    "follow_edges": _follow_edges(),
    "objects": OBJECTS,
    "community_assignment_rule": (
        "按 initial_opinion 划分 support、oppose、mixed 三个初始阵营，"
        "各阵营分别轮转分配到 community_1 至 community_4。"
    ),
    "community_memories": _community_memories(),
    "facility_memory_scope": "community",
    "official_news_schedule": _official_news_schedule(),
    "influencers": _influencers(),
    "influencer_schedule": _influencer_schedule(),
    "memories": _memories(),
}


def build_runtime(**kwargs):
    """构建 IAC v2 gay marriage 数据集场景。"""

    return build_runtime_from_spec(deepcopy(SPEC), **kwargs)
