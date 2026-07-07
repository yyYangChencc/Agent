from __future__ import annotations

from typing import Any

from persona.opinion.scale import JIANG_PING_TOPIC
from .builder import build_runtime_from_spec
from .map_designs import polarization_map_design


AGENT_IDS = [f"agent_{index}" for index in range(1, 11)]

SUPPORT_INFLUENCERS = ["jp_support_1", "jp_support_2", "jp_support_3"]
OPPOSE_INFLUENCERS = ["jp_oppose_1", "jp_oppose_2", "jp_oppose_3"]


def _post(author_id: str, content: str, opinion_index: float) -> dict[str, Any]:
    """生成固定投放帖子，保持极化实验可复现。"""

    return {
        "author_id": author_id,
        "topic": JIANG_PING_TOPIC,
        "content": content,
        "opinion_index": opinion_index,
        "is_rumor": True,
    }


OFFICIAL_NEWS_SCHEDULE = {
    1: {
        "topic": JIANG_PING_TOPIC,
        "title": "数学竞赛预赛排名引发公共关注",
        "content": (
            "一名来自中职学校的参赛者在数学竞赛预赛中取得高排名，引发公众对学习经历、教育路径和竞赛难度的讨论。"
            "目前公开信息仍以媒体报道和平台讨论为主。"
        ),
        "opinion_index": 0.20,
    },
    14: {
        "topic": JIANG_PING_TOPIC,
        "title": "围绕证明材料和报道节奏的讨论增加",
        "content": (
            "随着关注度升高，支持者强调个体努力与教育机会，怀疑者要求更多可核验证据。"
            "公开讨论开始从单一励志叙事转向多方争议。"
        ),
        "opinion_index": 0.00,
    },
    31: {
        "topic": JIANG_PING_TOPIC,
        "title": "赛事结果等待期拉长舆论分歧",
        "content": (
            "在进一步信息尚未充分公开的阶段，社交平台中出现更多互相矛盾的解释。"
            "一部分讨论关注个人遭遇，另一部分讨论竞赛透明度和媒体责任。"
        ),
        "opinion_index": -0.05,
    },
    51: {
        "topic": JIANG_PING_TOPIC,
        "title": "公共讨论转向平台情绪与群体分化",
        "content": (
            "围绕姜萍事件的讨论继续扩散，不同群体对同一信息作出明显不同解释。"
            "部分评论呼吁减少人身攻击，更多关注证据和程序。"
        ),
        "opinion_index": 0.00,
    },
    71: {
        "topic": JIANG_PING_TOPIC,
        "title": "争议持续但缺少决定性公开材料",
        "content": (
            "事件相关讨论仍在持续，支持和质疑两类叙事均在平台传播。"
            "当前公开材料不足以让所有群体形成一致判断。"
        ),
        "opinion_index": -0.05,
    },
}


INFLUENCERS = [
    {"id": "jp_support_1", "camp": "support", "name": "姜萍守护者A", "stance": 0.95},
    {"id": "jp_support_2", "camp": "support", "name": "中专奇迹声援号", "stance": 0.90},
    {"id": "jp_support_3", "camp": "support", "name": "反网暴观察", "stance": 0.85},
    {"id": "jp_oppose_1", "camp": "oppose", "name": "竞赛公信力质疑者A", "stance": -0.95},
    {"id": "jp_oppose_2", "camp": "oppose", "name": "反造神记录", "stance": -0.90},
    {"id": "jp_oppose_3", "camp": "oppose", "name": "证据优先讨论号", "stance": -0.85},
]


INFLUENCER_SCHEDULE = {
    8: [
        _post("jp_support_1", "这个故事说明普通路径里也可能出现强者，不该一开始就用恶意压过努力。", 0.95),
        _post("jp_oppose_1", "排名反差越大越应该公开证明材料，不能只靠情绪和励志标题带节奏。", -0.95),
    ],
    15: [
        _post("jp_support_2", "很多人不是在求证，而是在围攻一个年轻学生；这种舆论本身就不公平。", 0.85),
        _post("jp_oppose_2", "媒体把不完整信息包装成传奇，公众有权质疑这种造神叙事。", -0.90),
    ],
    24: [
        _post("jp_support_3", "没有最终材料前，反复用阴谋论攻击个人，是对普通学生最坏的示范。", 0.80),
        _post("jp_oppose_3", "同样没有最终材料前，也不能把质疑者说成网暴；证据链才是重点。", -0.80),
    ],
    36: [
        _post("jp_support_1", "越是争议大，越要保护当事人免受羞辱式审判；支持她不是拒绝证据。", 0.90),
        _post("jp_oppose_1", "把程序问题说成对个人恶意，是在转移焦点；竞赛透明度必须优先。", -0.92),
    ],
    48: [
        _post("jp_support_2", "如果她真的靠自学走到这一步，质疑浪潮会伤害很多普通学生的可能性。", 0.82),
        _post("jp_oppose_2", "如果平台继续只推励志故事，不推证明材料，舆论只会越来越失真。", -0.88),
    ],
    60: [
        _post("jp_support_3", "现在最需要的是克制和等待，而不是把一个学生当作流量靶子。", 0.78),
        _post("jp_oppose_3", "等待不等于沉默，越缺材料越要追问流程、责任和媒体边界。", -0.82),
    ],
    72: [
        _post("jp_support_1", "我仍然相信她至少代表了一种被忽视的努力，不能把所有不确定都解释成骗局。", 0.88),
        _post("jp_oppose_1", "越到后期越不能靠信念支撑叙事，公开、可复核、可追责才是底线。", -0.96),
    ],
    88: [
        _post("jp_support_2", "即使讨论程序，也不该把所有压力压到姜萍本人身上。", 0.76),
        _post("jp_oppose_2", "这个事件最该警惕的是媒体造神机制，而不是把怀疑者描述成冷血。", -0.86),
    ],
}


def _cluster_trust() -> list[dict[str, Any]]:
    """给支持组、反对组和混合组设置线下熟人关系。"""

    edges = []
    clusters = [
        ["agent_1", "agent_2", "agent_3", "agent_4"],
        ["agent_5", "agent_6", "agent_7", "agent_8"],
        ["agent_9", "agent_10"],
    ]
    for cluster in clusters:
        for source in cluster:
            for target in cluster:
                if source != target:
                    edges.append({"source": source, "target": target, "value": 0.65})
    for source, target in [("agent_4", "agent_9"), ("agent_9", "agent_4"), ("agent_8", "agent_10"), ("agent_10", "agent_8")]:
        edges.append({"source": source, "target": target, "value": 0.45})
    return edges


def _online_trust() -> list[dict[str, Any]]:
    """初始化对投放者和普通用户的线上信任。"""

    edges = []
    for agent_id in ["agent_1", "agent_2", "agent_3", "agent_4"]:
        for influencer in SUPPORT_INFLUENCERS:
            edges.append({"source": agent_id, "target": influencer, "value": 0.75})
        for influencer in OPPOSE_INFLUENCERS:
            edges.append({"source": agent_id, "target": influencer, "value": 0.25})
    for agent_id in ["agent_5", "agent_6", "agent_7", "agent_8"]:
        for influencer in OPPOSE_INFLUENCERS:
            edges.append({"source": agent_id, "target": influencer, "value": 0.75})
        for influencer in SUPPORT_INFLUENCERS:
            edges.append({"source": agent_id, "target": influencer, "value": 0.25})
    for agent_id in ["agent_9", "agent_10"]:
        for influencer in SUPPORT_INFLUENCERS + OPPOSE_INFLUENCERS:
            edges.append({"source": agent_id, "target": influencer, "value": 0.50})
    return edges


def _follow_edges() -> list[dict[str, str]]:
    """设置关注关系，形成支持组、反对组和跨阵营暴露组。"""

    edges: list[dict[str, str]] = []
    for follower in ["agent_1", "agent_2", "agent_3", "agent_4"]:
        for author in SUPPORT_INFLUENCERS:
            edges.append({"follower": follower, "author": author})
    for follower in ["agent_5", "agent_6", "agent_7", "agent_8"]:
        for author in OPPOSE_INFLUENCERS:
            edges.append({"follower": follower, "author": author})
    for follower in ["agent_9", "agent_10"]:
        for author in SUPPORT_INFLUENCERS + OPPOSE_INFLUENCERS:
            edges.append({"follower": follower, "author": author})
    ordinary_pairs = [
        ("agent_1", "agent_2"), ("agent_2", "agent_3"), ("agent_3", "agent_4"), ("agent_4", "agent_1"),
        ("agent_5", "agent_6"), ("agent_6", "agent_7"), ("agent_7", "agent_8"), ("agent_8", "agent_5"),
        ("agent_9", "agent_1"), ("agent_9", "agent_5"), ("agent_10", "agent_4"), ("agent_10", "agent_8"),
    ]
    for follower, author in ordinary_pairs:
        edges.append({"follower": follower, "author": author})
    return edges


def _objects() -> list[dict[str, Any]]:
    """放置 10 人场景需要的资源对象。"""

    return [
        {"kind": "bed", "id": "bed_1", "position": [2, 2]},
        {"kind": "bed", "id": "bed_2", "position": [2, 4]},
        {"kind": "bed", "id": "bed_3", "position": [2, 6]},
        {"kind": "bed", "id": "bed_4", "position": [2, 8]},
        {"kind": "bed", "id": "bed_5", "position": [2, 10]},
        {"kind": "bed", "id": "bed_6", "position": [4, 2]},
        {"kind": "bed", "id": "bed_7", "position": [4, 4]},
        {"kind": "bed", "id": "bed_8", "position": [4, 6]},
        {"kind": "bed", "id": "bed_9", "position": [4, 8]},
        {"kind": "bed", "id": "bed_10", "position": [4, 10]},
        {"kind": "company", "id": "company_1", "position": [20, 3], "params": {"salary": 8, "relax_cost": 8}},
        {"kind": "company", "id": "company_2", "position": [22, 6], "params": {"salary": 12, "relax_cost": 12}},
        {"kind": "food_shop", "id": "shop_1", "position": [2, 18], "params": {"food_num": 35, "provide": 30, "price": 6}},
        {"kind": "food_shop", "id": "shop_2", "position": [4, 21], "params": {"food_num": 30, "provide": 20, "price": 4}},
        {"kind": "playground", "id": "playground_1", "position": [20, 20], "params": {"provide": 18, "price": 5}},
        {"kind": "food", "id": "food_1", "position": [10, 8], "params": {"num": 4, "provide": 20}},
        {"kind": "food", "id": "food_2", "position": [12, 14], "params": {"num": 4, "provide": 20}},
        {"kind": "food", "id": "food_3", "position": [8, 18], "params": {"num": 4, "provide": 20}},
        {"kind": "food", "id": "food_4", "position": [15, 5], "params": {"num": 4, "provide": 20}},
        {"kind": "food", "id": "food_5", "position": [16, 20], "params": {"num": 4, "provide": 20}},
    ]


SPEC: dict[str, Any] = {
    "name": "jiang_ping_polarization",
    "agent_ids": AGENT_IDS,
    "map_design": polarization_map_design(),
    "default_initial_satisfaction": {"satiety": 55.0, "relax": 55.0},
    "agents": [
        {"id": "agent_1", "position": [3, 2], "speaking_style": "谨慎但容易被同伴影响", "initial_money": 8.0},
        {"id": "agent_2", "position": [3, 4], "speaking_style": "热情、愿意声援弱者", "initial_money": 12.0},
        {"id": "agent_3", "position": [4, 6], "speaking_style": "表达直接、容易被情绪感染", "initial_money": 10.0},
        {"id": "agent_4", "position": [5, 8], "speaking_style": "重视公平、倾向保护当事人", "initial_money": 16.0},
        {"id": "agent_5", "position": [6, 3], "speaking_style": "理性、强调证据链", "initial_money": 9.0},
        {"id": "agent_6", "position": [7, 5], "speaking_style": "怀疑媒体叙事、措辞尖锐", "initial_money": 14.0},
        {"id": "agent_7", "position": [8, 7], "speaking_style": "关注制度公信力", "initial_money": 18.0},
        {"id": "agent_8", "position": [9, 9], "speaking_style": "批判流量叙事、容易反感造神", "initial_money": 11.0},
        {"id": "agent_9", "position": [10, 4], "speaking_style": "中立、会比较不同证据", "initial_money": 20.0},
        {"id": "agent_10", "position": [11, 6], "speaking_style": "温和、倾向调和冲突", "initial_money": 15.0},
    ],
    "offline_trust": _cluster_trust(),
    "online_trust": _online_trust(),
    "follow_edges": _follow_edges(),
    "objects": _objects(),
    "map_memory": "极化实验地图包含住宅区、商业区、中心食物区、工作区和娱乐区；坐标格式为[row, col]。",
    "official_news_schedule": OFFICIAL_NEWS_SCHEDULE,
    "influencers": INFLUENCERS,
    "influencer_schedule": INFLUENCER_SCHEDULE,
}


def build_runtime(**kwargs):
    """构建姜萍事件舆论极化场景。"""

    return build_runtime_from_spec(SPEC, **kwargs)
