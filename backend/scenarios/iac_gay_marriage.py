from __future__ import annotations

import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

from persona.opinion.scale import clamp_opinion

from .builder import build_runtime_from_spec
from .map_designs import polarization_map_design


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "iac_v2" / "llm_pipeline" / "gay_marriage_1q"
SEED_PATH = DATA_DIR / "agent_initialization_seed_1q.json"
METADATA_PATH = DATA_DIR / "selected_discussion_metadata.json"
INJECTION_PATH = DATA_DIR / "injection_posts_direct.jsonl"

SUPPORT_INFLUENCERS = ["iac_gm_support_1", "iac_gm_support_2", "iac_gm_support_3"]
OPPOSE_INFLUENCERS = ["iac_gm_oppose_1", "iac_gm_oppose_2", "iac_gm_oppose_3"]


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


def _objects() -> list[dict[str, Any]]:
    """提供足够的基础设施，使数据集场景可以进入完整沙盒循环。"""

    beds = [
        {"kind": "bed", "id": f"bed_{index + 1}", "position": [1 + index // 5, 1 + (index % 5) * 2]}
        for index in range(20)
    ]
    facilities = [
        {"kind": "company", "id": "company_1", "position": [20, 3], "params": {"salary": 8, "relax_cost": 8}},
        {"kind": "company", "id": "company_2", "position": [22, 6], "params": {"salary": 12, "relax_cost": 12}},
        {"kind": "food_shop", "id": "shop_1", "position": [2, 18], "params": {"food_num": 160, "provide": 30, "price": 6}},
        {"kind": "food_shop", "id": "shop_2", "position": [4, 21], "params": {"food_num": 160, "provide": 20, "price": 4}},
        {"kind": "playground", "id": "playground_1", "position": [20, 20], "params": {"provide": 18, "price": 5}},
        {"kind": "food", "id": "food_1", "position": [10, 8], "params": {"num": 20, "provide": 20}},
        {"kind": "food", "id": "food_2", "position": [12, 14], "params": {"num": 20, "provide": 20}},
        {"kind": "food", "id": "food_3", "position": [8, 18], "params": {"num": 20, "provide": 20}},
        {"kind": "food", "id": "food_4", "position": [15, 5], "params": {"num": 20, "provide": 20}},
        {"kind": "food", "id": "food_5", "position": [16, 20], "params": {"num": 20, "provide": 20}},
    ]
    return beds + facilities


OBJECTS = _objects()


def _map_design() -> dict[str, Any]:
    """扩展基础地图的床位标注，前端和设施记忆可直接读取。"""

    design = polarization_map_design()
    object_regions = design.setdefault("object_regions", {})
    for item in OBJECTS:
        if item["kind"] != "bed":
            continue
        row, col = item["position"]
        object_regions[item["id"]] = {
            "region_id": "residential_area",
            "entrance": [row, min(24, col + 1)],
        }
    return design


def _occupied_object_positions() -> set[tuple[int, int]]:
    """收集物体格点，避免初始化实体智能体时被物体覆盖。"""

    return {tuple(item["position"]) for item in OBJECTS}


def _available_agent_positions() -> list[list[int]]:
    """生成 25x25 地图内可用出生点，顺序固定以保证复现。"""

    occupied = _occupied_object_positions()
    positions: list[list[int]] = []
    for row in range(25):
        for col in range(25):
            if (row, col) not in occupied:
                positions.append([row, col])
    return positions


AGENT_POSITIONS = _available_agent_positions()


def _agent_position(index: int) -> list[int]:
    """按数据顺序分配唯一出生点。"""

    if index >= len(AGENT_POSITIONS):
        raise ValueError("IAC gay marriage scenario has more agents than free map cells")
    return list(AGENT_POSITIONS[index])


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
    for index, row in enumerate(SEED_AGENTS):
        initial_opinion = clamp_opinion(_safe_float(row.get("initial_opinion")))
        agents.append(
            {
                "id": str(row["agent_id"]),
                "position": _agent_position(index),
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


def _memories() -> list[dict[str, Any]]:
    """给每个实体智能体注入由历史发言总结出的初始记忆。"""

    memories: list[dict[str, Any]] = []
    for row in SEED_AGENTS:
        confidence = _safe_float(row.get("profile_confidence"), 0.7)
        memories.append(
            {
                "agent_id": str(row["agent_id"]),
                "content": _memory_content(row),
                "memory_type": "semantic",
                "task": "agent_initialization",
                "object_id": f"iac_profile_{row.get('source_author_id')}",
                "importance": 0.9,
                "confidence": confidence,
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
    "map_memory": (
        "IAC gay marriage 场景使用 25x25 沙盒地图；实体智能体来自 IAC v2 初始化种子，"
        "线上投放者没有地图实体，只通过社交平台按时间线投放帖子。"
    ),
    "official_news_schedule": _official_news_schedule(),
    "influencers": _influencers(),
    "influencer_schedule": _influencer_schedule(),
    "memories": _memories(),
}


def build_runtime(**kwargs):
    """构建 IAC v2 gay marriage 数据集场景。"""

    return build_runtime_from_spec(deepcopy(SPEC), **kwargs)
