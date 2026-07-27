from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from persona.opinion.scale import clamp_opinion

from .iac_gay_marriage import SEED_AGENTS, SPEC as FULL_SPEC


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IAC_SOURCE_DB_PATH = PROJECT_ROOT / "data" / "processed" / "iac_v2" / "iac_v2.sqlite"
STANCE_OBSERVATIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "iac_v2"
    / "llm_pipeline"
    / "gay_marriage_1q"
    / "discussion_title_fourforums_For_or_Against_Gay_Marriage"
    / "post_stance_observations.jsonl"
)

SOURCE_DATASET = "fourforums"
TARGET_DISCUSSION_ID = "1613"
TARGET_DISCUSSION_TITLE = "For or Against Gay Marriage"
TARGET_TOPIC = "gay marriage"
INITIALIZATION_TIME = "2006-06-04 17:08:00"
INITIALIZATION_BOUNDARY = datetime.fromisoformat(INITIALIZATION_TIME)
ROLE_ORDER = ("support", "oppose", "mixed")
MAX_HISTORY_POSTS_PER_AGENT = 100
HISTORY_TEXT_PART_MAX_UTF8_BYTES = 800
EXPECTED_SOURCE_TOPIC_POST_COUNT = 390
EXPECTED_RETAINED_TOPIC_POST_COUNT = 334
NO_HISTORY_INITIAL_OPINION_SOURCE = "no_pre_t_post_in_selected_discussion"

RAW_AUTHOR_FIELDS = (
    "username",
    "gender",
    "age",
    "marital_status",
    "political_party",
    "country",
    "religion",
    "education",
)

CONFIRMED_INITIAL_OPINION_CORRECTIONS = {
    "iac_author_358": {
        "expected_initial_opinion": -1.0,
        "expected_mean_pre_t_opinion": 0.0,
        "expected_last_pre_t_post_id": "95",
        "corrected_initial_opinion": 1.0,
        "corrected_mean_pre_t_opinion": 1.0,
        "evidence_post_ids": ["80", "95"],
        "reason": "帖子 80 和 95 均主张同性伴侣应获得婚姻权利；帖子 95 的反讽语境曾被反向评分。",
    }
}


def _role_for_opinion(opinion: float) -> str:
    """按实验阈值把初始观念映射到三个分组。"""

    score = clamp_opinion(float(opinion))
    if score > 0.35:
        return "support"
    if score < -0.35:
        return "oppose"
    return "mixed"


def _corrected_seed_agents() -> list[dict[str, Any]]:
    """只应用已有原帖证据确认的初始观念修正。"""

    rows = deepcopy(SEED_AGENTS)
    by_id = {str(row["agent_id"]): row for row in rows}
    for agent_id, correction in CONFIRMED_INITIAL_OPINION_CORRECTIONS.items():
        row = by_id.get(agent_id)
        if row is None:
            raise ValueError(f"confirmed IAC initial opinion correction agent is missing: {agent_id}")
        if float(row["initial_opinion"]) != correction["expected_initial_opinion"]:
            raise ValueError(f"confirmed IAC initial opinion correction source score changed: {agent_id}")
        if float(row["mean_pre_t_opinion"]) != correction["expected_mean_pre_t_opinion"]:
            raise ValueError(f"confirmed IAC initial opinion correction mean score changed: {agent_id}")
        if str(row["last_pre_t_post_id"]) != correction["expected_last_pre_t_post_id"]:
            raise ValueError(f"confirmed IAC initial opinion correction source post changed: {agent_id}")
        row["initial_opinion"] = correction["corrected_initial_opinion"]
        row["mean_pre_t_opinion"] = correction["corrected_mean_pre_t_opinion"]
        row["initial_opinion_source"] = "confirmed_source_post_correction"
        row["initial_opinion_correction"] = deepcopy(correction)
    return rows


CORRECTED_SEED_AGENTS = _corrected_seed_agents()
FULL_AGENT_BY_ID = {str(item["id"]): item for item in FULL_SPEC["agents"]}
BED_CAPACITY_BY_COMMUNITY = Counter(
    str(item["community_id"])
    for item in FULL_SPEC["objects"]
    if item["kind"] == "bed"
)


def _validated_history_count(row: dict[str, Any]) -> int:
    """读取初始化前全部话题发帖数，并拒绝布尔值等错误类型。"""

    value = row["history_post_count_before_t"]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"history_post_count_before_t must be an integer: {row['agent_id']}")
    return value


def _selection_key(row: dict[str, Any], role: str) -> tuple[Any, ...]:
    """使用初始化种子的精确字段生成稳定选择顺序。"""

    agent_id = row["agent_id"]
    if not isinstance(agent_id, str) or not agent_id:
        raise ValueError("agents[].agent_id must be a non-empty string")
    scored_count = row["pre_t_scored_post_count"]
    if isinstance(scored_count, bool) or not isinstance(scored_count, int):
        raise ValueError(f"pre_t_scored_post_count must be an integer: {agent_id}")
    history_count = _validated_history_count(row)
    if role == "mixed":
        return (agent_id,)
    return (-scored_count, -history_count, agent_id)


def _is_strict_no_history_seed(row: dict[str, Any]) -> bool:
    """不确定组必须在初始化时点前没有任何 IAC v2 发帖。"""

    return (
        _validated_history_count(row) == 0
        and clamp_opinion(float(row["initial_opinion"])) == 0.0
        and row["initial_opinion_source"] == NO_HISTORY_INITIAL_OPINION_SOURCE
        and row["pre_t_scored_post_count"] == 0
    )


def _select_agent_ids_by_role() -> dict[str, tuple[str, ...]]:
    """固定抽取十名支持者、十名反对者和十名无历史用户。"""

    grouped: dict[str, list[dict[str, Any]]] = {role: [] for role in ROLE_ORDER}
    for row in CORRECTED_SEED_AGENTS:
        if _is_strict_no_history_seed(row):
            grouped["mixed"].append(row)
            continue
        role = _role_for_opinion(row["initial_opinion"])
        if role in {"support", "oppose"}:
            grouped[role].append(row)

    selected: dict[str, tuple[str, ...]] = {}
    for role in ("support", "oppose"):
        ordered = sorted(grouped[role], key=lambda row: _selection_key(row, role))
        if len(ordered) < 10:
            raise ValueError(f"IAC seed has fewer than 10 agents for role: {role}")
        selected[role] = tuple(str(row["agent_id"]) for row in ordered[:10])

    community_counts = Counter(
        str(FULL_AGENT_BY_ID[agent_id]["community_id"])
        for role in ("support", "oppose")
        for agent_id in selected[role]
    )
    mixed_ids: list[str] = []
    for row in sorted(grouped["mixed"], key=lambda item: _selection_key(item, "mixed")):
        agent_id = str(row["agent_id"])
        community_id = str(FULL_AGENT_BY_ID[agent_id]["community_id"])
        if community_counts[community_id] >= BED_CAPACITY_BY_COMMUNITY[community_id]:
            continue
        mixed_ids.append(agent_id)
        community_counts[community_id] += 1
        if len(mixed_ids) == 10:
            break
    if len(mixed_ids) != 10:
        raise ValueError("IAC seed has fewer than 10 strict no-history agents within community bed capacities")
    selected["mixed"] = tuple(mixed_ids)
    return selected


SELECTED_AGENT_IDS_BY_ROLE = _select_agent_ids_by_role()
SELECTED_AGENT_IDS = tuple(
    agent_id
    for role in ROLE_ORDER
    for agent_id in SELECTED_AGENT_IDS_BY_ROLE[role]
)
SELECTED_AGENT_ID_SET = set(SELECTED_AGENT_IDS)
SELECTED_SEED_BY_ID = {
    str(row["agent_id"]): row
    for row in CORRECTED_SEED_AGENTS
    if str(row["agent_id"]) in SELECTED_AGENT_ID_SET
}

if set(SELECTED_SEED_BY_ID) != SELECTED_AGENT_ID_SET:
    raise ValueError("IAC 30 selected seed rows are incomplete")
if "iac_author_358" in SELECTED_AGENT_IDS_BY_ROLE["oppose"]:
    raise ValueError("confirmed IAC initial opinion correction did not leave the opposing group")
if any(
    not _is_strict_no_history_seed(SELECTED_SEED_BY_ID[agent_id])
    for agent_id in SELECTED_AGENT_IDS_BY_ROLE["mixed"]
):
    raise ValueError("IAC 30 mixed group contains an agent with pre-initialization history")


def _parse_source_time(value: Any, *, field: str) -> datetime:
    """使用数据文件中的 ISO 时间格式解析并校验时间字段。"""

    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty ISO datetime string")
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"invalid {field}: {value}") from exc


def _numeric_string_key(value: Any) -> tuple[int, int | str]:
    """数字标识符按整数排序，其他已存在格式按原字符串排序。"""

    text = str(value)
    if text.isdigit():
        return (0, int(text))
    return (1, text)


def _load_source_rows() -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """从 SQLite 精确读取选中用户和当前讨论的初始化前帖子。"""

    if not IAC_SOURCE_DB_PATH.is_file():
        raise FileNotFoundError(f"IAC source database does not exist: {IAC_SOURCE_DB_PATH}")
    source_author_ids = {
        str(SELECTED_SEED_BY_ID[agent_id]["source_author_id"])
        for agent_id in SELECTED_AGENT_IDS
    }
    connection = sqlite3.connect(IAC_SOURCE_DB_PATH)
    connection.row_factory = sqlite3.Row
    try:
        author_rows = connection.execute(
            "SELECT dataset, author_id, username, gender, age, marital_status, "
            "political_party, country, religion, education FROM raw_author WHERE dataset = ?",
            (SOURCE_DATASET,),
        ).fetchall()
        post_rows = connection.execute(
            "SELECT p.dataset, p.discussion_id, p.post_id, p.author_id, a.username, "
            "p.creation_date, p.parent_post_id, d.title AS discussion_title, t.text "
            "FROM raw_post AS p "
            "JOIN raw_text AS t ON t.dataset = p.dataset AND t.text_id = p.text_id "
            "JOIN raw_discussion AS d ON d.dataset = p.dataset AND d.discussion_id = p.discussion_id "
            "LEFT JOIN raw_author AS a ON a.dataset = p.dataset AND a.author_id = p.author_id "
            "WHERE p.dataset = ? AND p.discussion_id = ?",
            (SOURCE_DATASET, TARGET_DISCUSSION_ID),
        ).fetchall()
    finally:
        connection.close()

    raw_author_by_source_id: dict[str, dict[str, Any]] = {}
    for row in author_rows:
        source_author_id = str(row["author_id"])
        if source_author_id not in source_author_ids:
            continue
        raw_author_by_source_id[source_author_id] = {
            field: row[field]
            for field in RAW_AUTHOR_FIELDS
        }
    if set(raw_author_by_source_id) != source_author_ids:
        missing = sorted(source_author_ids - set(raw_author_by_source_id), key=_numeric_string_key)
        raise ValueError(f"selected IAC raw_author rows are missing: {missing}")

    posts_by_agent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    agent_id_by_source_id = {
        str(seed["source_author_id"]): agent_id
        for agent_id, seed in SELECTED_SEED_BY_ID.items()
    }
    for row in post_rows:
        source_author_id = str(row["author_id"])
        agent_id = agent_id_by_source_id.get(source_author_id)
        if agent_id is None:
            continue
        creation_time = _parse_source_time(row["creation_date"], field="raw_post.creation_date")
        if creation_time >= INITIALIZATION_BOUNDARY:
            continue
        discussion_title = str(row["discussion_title"] or "")
        if discussion_title != TARGET_DISCUSSION_TITLE:
            raise ValueError(f"IAC discussion title changed for discussion_id={TARGET_DISCUSSION_ID}")
        text = row["text"]
        if not isinstance(text, str):
            raise ValueError(f"raw_text.text must be a string: post_id={row['post_id']}")
        posts_by_agent[agent_id].append(
            {
                "dataset": str(row["dataset"]),
                "discussion_id": str(row["discussion_id"]),
                "post_id": str(row["post_id"]),
                "author_id": source_author_id,
                "username": str(row["username"] or ""),
                "creation_date": str(row["creation_date"]),
                "parent_post_id": None if row["parent_post_id"] is None else str(row["parent_post_id"]),
                "discussion_title": discussion_title,
                "topic": TARGET_TOPIC,
                "text": text,
            }
        )

    for posts in posts_by_agent.values():
        posts.sort(
            key=lambda post: (
                _parse_source_time(post["creation_date"], field="raw_post.creation_date"),
                _numeric_string_key(post["discussion_id"]),
                _numeric_string_key(post["post_id"]),
            )
        )
    for agent_id in SELECTED_AGENT_IDS:
        posts_by_agent.setdefault(agent_id, [])
    return raw_author_by_source_id, dict(posts_by_agent)


RAW_AUTHOR_BY_SOURCE_ID, ALL_TOPIC_POSTS_BY_AGENT = _load_source_rows()


def _evenly_spaced_values(values: list[Any], count: int) -> list[Any]:
    """从完整时间跨度确定性抽取指定数量并保留首尾覆盖。"""

    if count <= 0 or not values:
        return []
    if count >= len(values):
        return list(values)
    if count == 1:
        return [values[-1]]
    indexes = [round(index * (len(values) - 1) / (count - 1)) for index in range(count)]
    if len(set(indexes)) != count:
        raise ValueError("evenly spaced history indexes are not unique")
    return [values[index] for index in indexes]


def _retain_topic_posts(agent_id: str, posts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """每人最多保留一百条，并强制保留首帖、末帖和初始观念证据帖。"""

    if len(posts) <= MAX_HISTORY_POSTS_PER_AGENT:
        return deepcopy(posts)
    seed = SELECTED_SEED_BY_ID[agent_id]
    required_post_ids = {
        str(posts[0]["post_id"]),
        str(posts[-1]["post_id"]),
        str(seed["last_pre_t_post_id"]),
    }
    posts_by_id = {str(post["post_id"]): post for post in posts}
    missing_required = sorted(required_post_ids - set(posts_by_id), key=_numeric_string_key)
    if missing_required:
        raise ValueError(f"initial opinion evidence posts are missing for {agent_id}: {missing_required}")
    optional_posts = [post for post in posts if str(post["post_id"]) not in required_post_ids]
    optional_count = MAX_HISTORY_POSTS_PER_AGENT - len(required_post_ids)
    selected_ids = set(required_post_ids)
    selected_ids.update(str(post["post_id"]) for post in _evenly_spaced_values(optional_posts, optional_count))
    retained = [deepcopy(post) for post in posts if str(post["post_id"]) in selected_ids]
    if len(retained) != MAX_HISTORY_POSTS_PER_AGENT:
        raise ValueError(f"retained topic post count mismatch for agent: {agent_id}")
    return retained


RETAINED_TOPIC_POSTS_BY_AGENT = {
    agent_id: _retain_topic_posts(agent_id, ALL_TOPIC_POSTS_BY_AGENT[agent_id])
    for agent_id in SELECTED_AGENT_IDS
}


def _load_stance_observations() -> dict[tuple[str, str], dict[str, Any]]:
    """读取当前讨论已有的逐帖立场评估，并校验其与原帖一致。"""

    if not STANCE_OBSERVATIONS_PATH.is_file():
        raise FileNotFoundError(f"IAC stance observations do not exist: {STANCE_OBSERVATIONS_PATH}")
    source_post_by_key = {
        (str(post["author_id"]), str(post["post_id"])): post
        for posts in ALL_TOPIC_POSTS_BY_AGENT.values()
        for post in posts
    }
    observations: dict[tuple[str, str], dict[str, Any]] = {}
    with STANCE_OBSERVATIONS_PATH.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"stance observation must be an object at line {line_number}")
            if row.get("dataset") != SOURCE_DATASET or str(row.get("discussion_id")) != TARGET_DISCUSSION_ID:
                continue
            key = (str(row.get("author_id")), str(row.get("post_id")))
            source_post = source_post_by_key.get(key)
            if source_post is None:
                continue
            if str(row.get("text")) != source_post["text"]:
                raise ValueError(f"stance observation text differs from raw_text: post_id={key[1]}")
            observations[key] = row
    return observations


STANCE_BY_SOURCE_POST = _load_stance_observations()


def _topic_memory_seeds(posts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """从当前讨论的逐帖评估中选取最多十二条可追溯立场摘要。"""

    rows: list[dict[str, Any]] = []
    for post in posts:
        observation = STANCE_BY_SOURCE_POST.get((str(post["author_id"]), str(post["post_id"])))
        if observation is None or not str(observation.get("reason") or "").strip():
            continue
        rows.append(
            {
                "post_id": str(post["post_id"]),
                "creation_date": str(post["creation_date"]),
                "stance_score": float(observation["stance_score"]),
                "confidence": float(observation["confidence"]),
                "reason": str(observation["reason"]),
            }
        )
    return _evenly_spaced_values(rows, min(12, len(rows)))


def _excerpt(text: str, limit: int = 220) -> str:
    """保留原文开头作为表达样本，并明确标记截断。"""

    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _build_topic_profile(agent_id: str) -> dict[str, Any]:
    """只依据当前讨论原帖构造可审计的静态用户画像。"""

    seed = SELECTED_SEED_BY_ID[agent_id]
    source_author_id = str(seed["source_author_id"])
    posts = RETAINED_TOPIC_POSTS_BY_AGENT[agent_id]
    raw_author = deepcopy(RAW_AUTHOR_BY_SOURCE_ID[source_author_id])
    if not posts:
        return {
            "profile_method": "strict_no_history_neutral_profile",
            "source_scope": "no_iac_v2_posts_before_initialization_time",
            "source_author_id": source_author_id,
            "raw_author": raw_author,
            "speaking_style": "初始化时点前没有可用发言，无法可靠概括语言风格。",
            "argument_style": "初始化时点前没有可用发言，无法可靠概括论证方式。",
            "social_style": "初始化时点前没有可用发言，无法可靠概括互动方式。",
            "topic_interests": [],
            "memory_seeds": [],
            "initialization_notes": [
                "不得使用未来帖子或其他用户资料推断该智能体的表达习惯和立场。",
                "使用中性默认表达，后续只由仿真中的实际经历塑造。",
            ],
            "profile_confidence": 0.0,
            "source_topic_post_count": 0,
            "retained_topic_post_count": 0,
            "reply_post_count": 0,
            "representative_excerpts": [],
        }

    representative_posts = _evenly_spaced_values(posts, min(3, len(posts)))
    representative_excerpts = [
        {
            "post_id": str(post["post_id"]),
            "creation_date": str(post["creation_date"]),
            "text": _excerpt(str(post["text"])),
        }
        for post in representative_posts
    ]
    reply_count = sum(post["parent_post_id"] is not None for post in posts)
    average_text_length = round(sum(len(str(post["text"])) for post in posts) / len(posts), 2)
    memory_seeds = _topic_memory_seeds(posts)
    excerpt_text = "；".join(
        f"post_id={item['post_id']}：{item['text']}"
        for item in representative_excerpts
    )
    reason_text = "；".join(
        f"post_id={item['post_id']}：{item['reason']}"
        for item in _evenly_spaced_values(memory_seeds, min(3, len(memory_seeds)))
    )
    # 该分值只表示当前话题可用证据量，不表示对现实人格判断的正确率。
    evidence_confidence = round(min(1.0, len(posts) / 20.0), 2)
    return {
        "profile_method": "target_discussion_evidence_profile",
        "source_scope": f"{SOURCE_DATASET}/{TARGET_DISCUSSION_ID}/before/{INITIALIZATION_TIME}",
        "source_author_id": source_author_id,
        "raw_author": raw_author,
        "speaking_style": (
            "只依据初始化前当前同性婚姻讨论中的本人原帖保持英文论坛表达，不导入其他话题风格。"
            f"可追溯表达样本：{excerpt_text}"
        ),
        "argument_style": (
            "论证内容只沿用当前讨论中可追溯的原帖理由。"
            + (f"逐帖立场评估摘要：{reason_text}" if reason_text else "没有可用逐帖理由摘要。")
        ),
        "social_style": (
            f"保留的当前讨论发言共 {len(posts)} 条，其中回复型发言 {reply_count} 条，"
            f"平均正文长度 {average_text_length} 个字符；互动方式以原帖证据为准。"
        ),
        "topic_interests": [TARGET_TOPIC],
        "memory_seeds": memory_seeds,
        "initialization_notes": [
            "只模仿当前讨论中能够从原帖直接观察到的表达，不进行现实人格诊断。",
            "初始化前本人发帖是历史表达证据，不是仿真期间刚看到的平台信息。",
            "其他话题帖子和初始化时点之后的帖子均未进入画像。",
        ],
        "profile_confidence": evidence_confidence,
        "source_topic_post_count": len(ALL_TOPIC_POSTS_BY_AGENT[agent_id]),
        "retained_topic_post_count": len(posts),
        "reply_post_count": reply_count,
        "average_text_length": average_text_length,
        "representative_excerpts": representative_excerpts,
    }


TOPIC_PROFILE_BY_AGENT = {
    agent_id: _build_topic_profile(agent_id)
    for agent_id in SELECTED_AGENT_IDS
}


def build_agent_initialization_fields(agent_id: str, role: str) -> dict[str, Any]:
    """返回场景构建器写入智能体的完整数据集初始化字段。"""

    if agent_id not in SELECTED_AGENT_ID_SET:
        raise ValueError(f"agent is outside the selected IAC 30 set: {agent_id}")
    if role not in ROLE_ORDER:
        raise ValueError(f"unsupported IAC 30 role: {role}")
    seed = SELECTED_SEED_BY_ID[agent_id]
    if _role_for_opinion(float(seed["initial_opinion"])) != role:
        raise ValueError(f"IAC 30 initial opinion and role mismatch: {agent_id}")
    if role == "mixed" and not _is_strict_no_history_seed(seed):
        raise ValueError(f"IAC 30 mixed agent is not a strict no-history user: {agent_id}")
    profile = deepcopy(TOPIC_PROFILE_BY_AGENT[agent_id])
    return {
        "initial_opinion": clamp_opinion(float(seed["initial_opinion"])),
        "initial_opinion_source": str(seed["initial_opinion_source"]),
        "initialization_time": INITIALIZATION_TIME,
        "source_author_id": str(seed["source_author_id"]),
        "discussion_title": TARGET_DISCUSSION_TITLE,
        "discussion_id": TARGET_DISCUSSION_ID,
        "dataset": SOURCE_DATASET,
        "speaking_style": str(profile["speaking_style"]),
        "dataset_user_profile": profile,
    }


def _split_utf8_text(text: str, max_bytes: int = HISTORY_TEXT_PART_MAX_UTF8_BYTES) -> list[str]:
    """按 UTF-8 字节上限无损切分正文，不丢弃空格和换行。"""

    if max_bytes <= 0:
        raise ValueError("history text part byte limit must be greater than zero")
    if not text:
        return [""]
    parts: list[str] = []
    current: list[str] = []
    current_bytes = 0
    for character in text:
        character_bytes = len(character.encode("utf-8"))
        if current and current_bytes + character_bytes > max_bytes:
            parts.append("".join(current))
            current = []
            current_bytes = 0
        current.append(character)
        current_bytes += character_bytes
    if current:
        parts.append("".join(current))
    if "".join(parts) != text:
        raise ValueError("history text parts do not reconstruct the source post")
    return parts


def _profile_memory(agent_id: str) -> dict[str, Any]:
    """把结构化画像格式化为一条可召回初始化记忆。"""

    profile = TOPIC_PROFILE_BY_AGENT[agent_id]
    raw_author = profile["raw_author"]
    raw_author_text = "，".join(
        f"{field}={raw_author[field]}"
        for field in RAW_AUTHOR_FIELDS
        if raw_author.get(field) not in {None, ""}
    ) or "无非空 raw_author 属性"
    content = (
        f"数据集用户初始化画像：agent_id={agent_id}，source_author_id={profile['source_author_id']}。"
        f"资料：{raw_author_text}。"
        f"发言风格：{profile['speaking_style']}"
        f"论证约束：{profile['argument_style']}"
        f"社交表达：{profile['social_style']}"
    )
    return {
        "agent_id": agent_id,
        "content": content,
        "memory_type": "semantic",
        "task": "dataset_user_initialization",
        "source_type": "dataset_user_profile",
        "provenance": "pre_simulation_dataset_profile",
        "object_id": f"dataset_user_profile:{agent_id}",
        "episode_id": f"dataset_user_profile:{agent_id}",
        "importance": 1.0,
        "confidence": float(profile["profile_confidence"]),
        "require_embedding": True,
    }


def _initial_opinion_memory(agent_id: str, role: str) -> dict[str, Any]:
    """生成由当前讨论最后评分帖子支撑的初始观念记忆。"""

    seed = SELECTED_SEED_BY_ID[agent_id]
    score = clamp_opinion(float(seed["initial_opinion"]))
    if role == "mixed":
        content = (
            f"数据集初始观念：topic={TARGET_TOPIC}，initial_role=mixed，initial_opinion=0.0。"
            "该用户在初始化时点前没有任何 IAC v2 发帖，不能从历史推断支持或反对立场。"
        )
        confidence = 1.0
        evidence_post_id = ""
    else:
        evidence_post_id = str(seed["last_pre_t_post_id"])
        evidence_post = next(
            (post for post in ALL_TOPIC_POSTS_BY_AGENT[agent_id] if post["post_id"] == evidence_post_id),
            None,
        )
        if evidence_post is None:
            raise ValueError(f"initial opinion evidence post is missing: {agent_id}/{evidence_post_id}")
        observation = STANCE_BY_SOURCE_POST.get((str(evidence_post["author_id"]), evidence_post_id))
        reason = str(observation.get("reason") or "") if observation else ""
        confidence = float(observation.get("confidence", 1.0)) if observation else 1.0
        content = (
            f"数据集初始观念：topic={TARGET_TOPIC}，initial_role={role}，initial_opinion={score}，"
            f"evidence_post_id={evidence_post_id}，creation_date={evidence_post['creation_date']}。"
            f"逐帖评估摘要：{reason or '没有可用评估摘要'}。"
            f"本人原文节选：{_excerpt(evidence_post['text'], 600)}"
        )
    return {
        "agent_id": agent_id,
        "content": content,
        "memory_type": "semantic",
        "task": "dataset_initial_opinion",
        "source_type": "dataset_initial_opinion",
        "provenance": "pre_simulation_self_post" if evidence_post_id else "pre_simulation_no_history",
        "object_id": f"dataset_initial_opinion:{agent_id}",
        "episode_id": f"dataset_initial_opinion:{agent_id}",
        "source_post_id": evidence_post_id,
        "importance": 1.0,
        "confidence": confidence,
        "require_embedding": True,
    }


def _topic_seed_memories(agent_id: str) -> list[dict[str, Any]]:
    """将当前讨论的立场摘要逐条写入，避免合并后无法精确召回。"""

    records: list[dict[str, Any]] = []
    for item in TOPIC_PROFILE_BY_AGENT[agent_id]["memory_seeds"]:
        post_id = str(item["post_id"])
        records.append(
            {
                "agent_id": agent_id,
                "content": (
                    f"初始化前当前话题发言摘要：post_id={post_id}，"
                    f"creation_date={item['creation_date']}，stance_score={item['stance_score']}。"
                    f"逐帖评估摘要：{item['reason']}"
                ),
                "memory_type": "semantic",
                "task": "dataset_topic_memory_seed",
                "source_type": "dataset_stance_summary",
                "provenance": "pre_simulation_self_post_assessment",
                "object_id": f"dataset_stance_summary:{SOURCE_DATASET}:{TARGET_DISCUSSION_ID}:{post_id}",
                "episode_id": f"dataset_stance_summary:{SOURCE_DATASET}:{TARGET_DISCUSSION_ID}:{post_id}",
                "source_post_id": post_id,
                "source_creation_date": str(item["creation_date"]),
                "importance": 0.85,
                "confidence": float(item["confidence"]),
                "require_embedding": True,
            }
        )
    return records


def _topic_post_memories(agent_id: str) -> list[dict[str, Any]]:
    """把选中的当前讨论原帖无损分段为可召回语义记忆。"""

    posts = RETAINED_TOPIC_POSTS_BY_AGENT[agent_id]
    if not posts:
        return [
            {
                "agent_id": agent_id,
                "content": "初始化前发帖历史为空：该用户在初始化时点前没有任何 IAC v2 发帖。",
                "memory_type": "semantic",
                "task": "dataset_post_history_empty",
                "source_type": "dataset_post_history_empty",
                "provenance": "pre_simulation_no_history",
                "object_id": f"dataset_post_history_empty:{agent_id}",
                "episode_id": f"dataset_post_history_empty:{agent_id}",
                "importance": 1.0,
                "confidence": 1.0,
                "require_embedding": True,
            }
        ]

    records: list[dict[str, Any]] = []
    for post in posts:
        text_parts = _split_utf8_text(str(post["text"]))
        if "".join(text_parts) != post["text"]:
            raise ValueError(f"history post reconstruction failed: post_id={post['post_id']}")
        for part_index, text_part in enumerate(text_parts, start=1):
            post_id = str(post["post_id"])
            episode_id = (
                f"dataset_post:{post['dataset']}:{post['discussion_id']}:{post_id}:"
                f"part:{part_index}:{len(text_parts)}"
            )
            records.append(
                {
                    "agent_id": agent_id,
                    "content": (
                        "[初始化前本人发帖]\n"
                        f"discussion_title={post['discussion_title']}\n"
                        f"creation_date={post['creation_date']}\n"
                        f"post_id={post_id}\n"
                        f"text_part={part_index}/{len(text_parts)}\n"
                        f"正文：{text_part}"
                    ),
                    "memory_type": "semantic",
                    "task": "dataset_post_history",
                    "source_type": "dataset_post_history",
                    "provenance": "pre_simulation_self_post",
                    "object_id": episode_id,
                    "episode_id": episode_id,
                    "dataset": str(post["dataset"]),
                    "discussion_id": str(post["discussion_id"]),
                    "source_post_id": post_id,
                    "source_author_id": str(post["author_id"]),
                    "source_creation_date": str(post["creation_date"]),
                    "parent_post_id": "" if post["parent_post_id"] is None else str(post["parent_post_id"]),
                    "discussion_title": str(post["discussion_title"]),
                    "topic": TARGET_TOPIC,
                    "text_part": part_index,
                    "text_part_count": len(text_parts),
                    "importance": 0.8,
                    "confidence": 1.0,
                    "require_embedding": True,
                }
            )
    return records


def build_dataset_initialization_memories(agent_id: str, role: str) -> list[dict[str, Any]]:
    """返回单个智能体全部数据集画像、观念、摘要和历史记忆。"""

    records = [_profile_memory(agent_id), _initial_opinion_memory(agent_id, role)]
    records.extend(_topic_seed_memories(agent_id))
    records.extend(_topic_post_memories(agent_id))
    return records


SOURCE_TOPIC_POST_COUNT_BY_ROLE = {
    role: sum(len(ALL_TOPIC_POSTS_BY_AGENT[agent_id]) for agent_id in SELECTED_AGENT_IDS_BY_ROLE[role])
    for role in ROLE_ORDER
}
RETAINED_TOPIC_POST_COUNT_BY_ROLE = {
    role: sum(len(RETAINED_TOPIC_POSTS_BY_AGENT[agent_id]) for agent_id in SELECTED_AGENT_IDS_BY_ROLE[role])
    for role in ROLE_ORDER
}
TOTAL_SOURCE_TOPIC_POST_COUNT = sum(SOURCE_TOPIC_POST_COUNT_BY_ROLE.values())
TOTAL_RETAINED_TOPIC_POST_COUNT = sum(RETAINED_TOPIC_POST_COUNT_BY_ROLE.values())

if TOTAL_SOURCE_TOPIC_POST_COUNT != EXPECTED_SOURCE_TOPIC_POST_COUNT:
    raise ValueError(
        f"IAC 30 source topic post count changed: {TOTAL_SOURCE_TOPIC_POST_COUNT} "
        f"!= {EXPECTED_SOURCE_TOPIC_POST_COUNT}"
    )
if TOTAL_RETAINED_TOPIC_POST_COUNT != EXPECTED_RETAINED_TOPIC_POST_COUNT:
    raise ValueError(
        f"IAC 30 retained topic post count changed: {TOTAL_RETAINED_TOPIC_POST_COUNT} "
        f"!= {EXPECTED_RETAINED_TOPIC_POST_COUNT}"
    )
if SOURCE_TOPIC_POST_COUNT_BY_ROLE != {"support": 346, "oppose": 44, "mixed": 0}:
    raise ValueError(f"IAC 30 source topic post role counts changed: {SOURCE_TOPIC_POST_COUNT_BY_ROLE}")
if RETAINED_TOPIC_POST_COUNT_BY_ROLE != {"support": 290, "oppose": 44, "mixed": 0}:
    raise ValueError(f"IAC 30 retained topic post role counts changed: {RETAINED_TOPIC_POST_COUNT_BY_ROLE}")


INITIALIZATION_AUDIT = {
    "source_database": str(IAC_SOURCE_DB_PATH),
    "stance_observations": str(STANCE_OBSERVATIONS_PATH),
    "dataset": SOURCE_DATASET,
    "discussion_id": TARGET_DISCUSSION_ID,
    "discussion_title": TARGET_DISCUSSION_TITLE,
    "topic": TARGET_TOPIC,
    "initialization_time": INITIALIZATION_TIME,
    "agent_ids_by_role": {
        role: list(SELECTED_AGENT_IDS_BY_ROLE[role])
        for role in ROLE_ORDER
    },
    "source_topic_post_count_by_role": deepcopy(SOURCE_TOPIC_POST_COUNT_BY_ROLE),
    "retained_topic_post_count_by_role": deepcopy(RETAINED_TOPIC_POST_COUNT_BY_ROLE),
    "source_topic_post_count": TOTAL_SOURCE_TOPIC_POST_COUNT,
    "retained_topic_post_count": TOTAL_RETAINED_TOPIC_POST_COUNT,
    "max_history_posts_per_agent": MAX_HISTORY_POSTS_PER_AGENT,
    "history_text_part_max_utf8_bytes": HISTORY_TEXT_PART_MAX_UTF8_BYTES,
    "selection_rule": (
        "support/oppose 按 pre_t_scored_post_count 降序、history_post_count_before_t 降序、agent_id 升序；"
        "mixed 必须满足 history_post_count_before_t=0，并在原社区床位容量内按 agent_id 升序选择。"
    ),
    "history_scope_rule": (
        f"只保留 dataset={SOURCE_DATASET}、discussion_id={TARGET_DISCUSSION_ID} 且 creation_date "
        f"早于 {INITIALIZATION_TIME} 的帖子；每人最多 {MAX_HISTORY_POSTS_PER_AGENT} 条。"
    ),
}

