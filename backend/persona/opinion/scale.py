from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

OPINION_MIN = -1.0
OPINION_NEUTRAL = 0.0
OPINION_MAX = 1.0

VOTING_ROLE_SUPPORT = "support"
VOTING_ROLE_OPPOSE = "oppose"
VOTING_ROLE_UNKNOWN = "unknown"
VOTING_POLARIZATION_ROLES = {
    VOTING_ROLE_SUPPORT,
    VOTING_ROLE_OPPOSE,
    VOTING_ROLE_UNKNOWN,
}
VOTING_STANCE_INVALID = "invalid"

# 立场分类使用固定、可解释的保守规则，避免简单最高票放大分裂结果。
VOTING_MIN_SUCCESS_RATE = 0.80
VOTING_MIN_DIRECTION_SHARE = 0.60
VOTING_MIN_DIRECTION_MARGIN = 0.20
VOTING_MIN_UNKNOWN_SHARE = 0.50


def classify_voting_stance(
    *,
    requested_voters: int,
    successful_votes: int,
    choice_counts: Mapping[str, Any],
    option_roles: Mapping[str, str],
) -> dict[str, Any]:
    """把单个智能体的一轮投票转换为离散立场和一致性指标。"""

    requested = max(0, int(requested_voters or 0))
    successful = max(0, int(successful_votes or 0))
    role_counts = {
        VOTING_ROLE_SUPPORT: 0,
        VOTING_ROLE_OPPOSE: 0,
        VOTING_ROLE_UNKNOWN: 0,
    }
    for option, raw_count in choice_counts.items():
        role = option_roles.get(option)
        if role not in role_counts:
            continue
        try:
            count = int(raw_count or 0)
        except (TypeError, ValueError):
            count = 0
        role_counts[role] += max(0, count)

    divisor = float(successful) if successful else 1.0
    support_share = role_counts[VOTING_ROLE_SUPPORT] / divisor if successful else 0.0
    oppose_share = role_counts[VOTING_ROLE_OPPOSE] / divisor if successful else 0.0
    unknown_share = role_counts[VOTING_ROLE_UNKNOWN] / divisor if successful else 0.0
    success_rate = successful / requested if requested else 0.0
    agreement = max(support_share, oppose_share, unknown_share)
    direction_margin = abs(support_share - oppose_share)

    if requested <= 0 or success_rate + 1e-12 < VOTING_MIN_SUCCESS_RATE:
        stance = VOTING_STANCE_INVALID
        valid = False
    elif unknown_share + 1e-12 >= VOTING_MIN_UNKNOWN_SHARE:
        stance = VOTING_ROLE_UNKNOWN
        valid = True
    elif (
        support_share + 1e-12 >= VOTING_MIN_DIRECTION_SHARE
        and support_share - oppose_share + 1e-12 >= VOTING_MIN_DIRECTION_MARGIN
    ):
        stance = VOTING_ROLE_SUPPORT
        valid = True
    elif (
        oppose_share + 1e-12 >= VOTING_MIN_DIRECTION_SHARE
        and oppose_share - support_share + 1e-12 >= VOTING_MIN_DIRECTION_MARGIN
    ):
        stance = VOTING_ROLE_OPPOSE
        valid = True
    else:
        stance = VOTING_ROLE_UNKNOWN
        valid = True

    return {
        "stance": stance,
        "stance_valid": valid,
        "stance_agreement": agreement,
        "stance_direction_margin": direction_margin,
        "stance_success_rate": success_rate,
    }

JIANG_PING_TOPIC = "姜萍事件"
GAY_MARRIAGE_TOPIC = "For or Against Gay Marriage"


@dataclass(frozen=True)
class OpinionTopicDefinition:
    """定义一个可评测的新闻观念主题。

    direction_prompt 明确 score 的正负含义，避免把观念分数误解为对单条帖子
    或 system 账号的相信度。新增新闻主题时，只需要新增一个主题定义并注册。
    """

    topic: str
    narrative: str
    direction_prompt: str
    scale: list[dict[str, object]]
    voting_options: list[str]
    voting_option_roles: dict[str, str]
    positive_keywords: list[str]
    negative_keywords: list[str]


JIANG_PING_OPINION_SCALE: list[dict[str, object]] = [
    {
        "range": [-1.0, -0.75],
        "label": "激烈反对",
        "description": "对姜萍事件正面叙事强烈反对或强烈质疑，可能质疑成绩真实性、媒体造神、竞赛公信力或教育叙事。",
    },
    {
        "range": [-0.75, -0.35],
        "label": "明显质疑或反对",
        "description": "明显质疑或反对姜萍事件中的正面叙事，批评主要集中在事件本身。",
    },
    {
        "range": [-0.35, -0.10],
        "label": "轻度怀疑",
        "description": "对姜萍事件正面叙事轻度怀疑，更倾向于等待更充分证据后再接受支持性说法。",
    },
    {
        "range": [-0.10, 0.10],
        "label": "中立或未知",
        "description": "不关心、没听说过，或暂时不愿表态。",
    },
    {
        "range": [0.10, 0.35],
        "label": "轻度同情或谨慎支持",
        "description": "对姜萍本人或事件正面叙事轻度同情、谨慎支持，同时承认事件仍存在不确定性。",
    },
    {
        "range": [0.35, 0.75],
        "label": "明显支持",
        "description": "明显支持姜萍事件正面叙事，接受励志解释或对姜萍本人表达明确支持。",
    },
    {
        "range": [0.75, 1.0],
        "label": "完全赞成",
        "description": "强烈支持姜萍事件正面叙事，高度相信励志解释，明显同情或支持姜萍。",
    },
]

JIANG_PING_TOPIC_DEFINITION = OpinionTopicDefinition(
    topic=JIANG_PING_TOPIC,
    narrative="姜萍事件正面叙事",
    direction_prompt=(
        "-1 表示对姜萍事件正面叙事强烈反对/强烈质疑，可能包括质疑成绩真实性、"
        "反对媒体造神、质疑竞赛公信力等；0 表示中立/未知/观望；"
        "1 表示对姜萍事件正面叙事强烈支持，可能包括相信励志叙事、同情/支持姜萍、接受正面解释。"
    ),
    scale=JIANG_PING_OPINION_SCALE,
    voting_options=[
        "该用户支持姜萍",
        "该用户不支持姜萍",
        "该用户立场未知",
    ],
    voting_option_roles={
        "该用户支持姜萍": VOTING_ROLE_SUPPORT,
        "该用户不支持姜萍": VOTING_ROLE_OPPOSE,
        "该用户立场未知": VOTING_ROLE_UNKNOWN,
    },
    positive_keywords=[
        "支持",
        "赞同",
        "同情",
        "鼓励",
        "励志",
        "突出个体",
        "正面叙事",
        "谨慎支持",
        "接受正面解释",
        "相信励志叙事",
        "姜萍",
    ],
    negative_keywords=[
        "反对",
        "质疑",
        "怀疑",
        "违规",
        "造神",
        "公信力",
        "不透明",
        "包装典型",
        "流量叙事",
        "过度帮助",
        "成绩真实性",
        "竞赛公信力",
    ],
)

GAY_MARRIAGE_TOPIC_DEFINITION = OpinionTopicDefinition(
    topic=GAY_MARRIAGE_TOPIC,
    narrative="Gay marriage or same-sex marriage rights should be supported.",
    direction_prompt=(
        "-1 means the user opposes gay marriage or argues against same-sex marriage rights; "
        "0 means the user's stance is unknown; "
        "1 means the user supports gay marriage or same-sex marriage rights."
    ),
    scale=[],
    voting_options=[
        "This user supports gay marriage or same-sex marriage rights",
        "This user opposes gay marriage or argues against same-sex marriage rights",
        "This user's stance is unknown",
    ],
    voting_option_roles={
        "This user supports gay marriage or same-sex marriage rights": VOTING_ROLE_SUPPORT,
        "This user opposes gay marriage or argues against same-sex marriage rights": VOTING_ROLE_OPPOSE,
        "This user's stance is unknown": VOTING_ROLE_UNKNOWN,
    },
    positive_keywords=["support gay marriage", "same-sex marriage rights", "marriage equality"],
    negative_keywords=["oppose gay marriage", "against same-sex marriage", "gay marriage ban"],
)

# 主题注册表是新增新闻主题的统一入口，评测、prompt 和规则兜底都从这里读取语义。
OPINION_TOPIC_DEFINITIONS: dict[str, OpinionTopicDefinition] = {
    JIANG_PING_TOPIC_DEFINITION.topic: JIANG_PING_TOPIC_DEFINITION,
    GAY_MARRIAGE_TOPIC_DEFINITION.topic: GAY_MARRIAGE_TOPIC_DEFINITION,
}


def clamp_opinion(value: float) -> float:
    return round(max(OPINION_MIN, min(OPINION_MAX, float(value))), 4)


def get_opinion_topic_definition(topic: str) -> OpinionTopicDefinition:
    """按主题名获取观念主题定义，未知主题返回通用定义以保持系统可运行。"""

    definition = OPINION_TOPIC_DEFINITIONS.get(topic)
    if definition is not None:
        return definition
    return OpinionTopicDefinition(
        topic=topic,
        narrative=f"{topic}的正面叙事" if topic else "系统新闻主题的正面叙事",
        direction_prompt=(
            f"-1 表示强烈反对/质疑{topic or '该主题'}的正面叙事；"
            "0 表示中立/未知/观望；"
            f"1 表示强烈支持/接受{topic or '该主题'}的正面叙事。"
        ),
        scale=[],
        voting_options=[],
        voting_option_roles={},
        positive_keywords=["支持", "赞同", "认可", "正面", "谨慎支持", "接受"],
        negative_keywords=["反对", "质疑", "怀疑", "负面", "不透明", "不认可"],
    )
