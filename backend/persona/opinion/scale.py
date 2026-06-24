from __future__ import annotations

OPINION_MIN = -1.0
OPINION_NEUTRAL = 0.0
OPINION_MAX = 1.0

JIANG_PING_TOPIC = "姜萍事件"

JIANG_PING_OPINION_SCALE: list[dict[str, object]] = [
    {
        "range": [-1.0, -0.75],
        "label": "激烈反对",
        "description": "激烈反对姜萍，并可能上升到反对媒体造神、竞赛公信力或教育叙事层面。",
    },
    {
        "range": [-0.75, -0.35],
        "label": "明显质疑或反对",
        "description": "明显质疑或反对姜萍事件中的正面叙事，批评主要集中在事件本身。",
    },
    {
        "range": [-0.35, -0.10],
        "label": "轻度怀疑",
        "description": "轻度怀疑，更倾向于等待更充分证据后再接受支持性说法。",
    },
    {
        "range": [-0.10, 0.10],
        "label": "中立或未知",
        "description": "不关心、没听说过，或暂时不愿表态。",
    },
    {
        "range": [0.10, 0.35],
        "label": "轻度同情或谨慎支持",
        "description": "轻度同情或谨慎支持，同时承认事件仍存在不确定性。",
    },
    {
        "range": [0.35, 0.75],
        "label": "明显支持",
        "description": "明显支持姜萍，或支持围绕她的正面励志叙事。",
    },
    {
        "range": [0.75, 1.0],
        "label": "完全赞成",
        "description": "完全赞成或高度支持姜萍，并强烈接受该事件的励志解释。",
    },
]


def clamp_opinion(value: float) -> float:
    return round(max(OPINION_MIN, min(OPINION_MAX, float(value))), 4)
