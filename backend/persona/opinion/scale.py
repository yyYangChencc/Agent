from __future__ import annotations

from dataclasses import dataclass

OPINION_MIN = -1.0
OPINION_NEUTRAL = 0.0
OPINION_MAX = 1.0

JIANG_PING_TOPIC = "姜萍事件"


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

# 主题注册表是新增新闻主题的统一入口，评测、prompt 和规则兜底都从这里读取语义。
OPINION_TOPIC_DEFINITIONS: dict[str, OpinionTopicDefinition] = {
    JIANG_PING_TOPIC_DEFINITION.topic: JIANG_PING_TOPIC_DEFINITION,
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
        positive_keywords=["支持", "赞同", "认可", "正面", "谨慎支持", "接受"],
        negative_keywords=["反对", "质疑", "怀疑", "负面", "不透明", "不认可"],
    )
