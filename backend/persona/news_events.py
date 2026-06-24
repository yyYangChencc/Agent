from __future__ import annotations

from persona.opinion.scale import JIANG_PING_OPINION_SCALE, JIANG_PING_TOPIC


# Format: tick -> {"topic": str, "title": str, "content": str, "opinion_index": float}
# The schedule is a simulation-oriented reconstruction of the public Jiang Ping
# incident timeline, not a verbatim news archive.
OPINION_SCALE = JIANG_PING_OPINION_SCALE

NEWS_SCHEDULE: dict[int, dict] = {
    5: {
        "topic": JIANG_PING_TOPIC,
        "title": "阿里数学竞赛预赛结果引发关注",
        "content": (
            "公开报道显示，江苏涟水一名中专学生姜萍在2024阿里巴巴全球数学竞赛预赛中取得第12名。"
            "她的学校背景、年龄和排名形成强烈反差，社交平台开始大量转发励志叙事。"
        ),
        "opinion_index": 0.80,
    },
    12: {
        "topic": JIANG_PING_TOPIC,
        "title": "媒体集中报道姜萍的学习经历",
        "content": (
            "多家媒体继续报道姜萍自学高等数学、参加竞赛的经历。"
            "支持者认为这说明普通教育路径之外也可能出现突出个体，相关讨论以鼓励和赞赏为主。"
        ),
        "opinion_index": 0.65,
    },
    20: {
        "topic": JIANG_PING_TOPIC,
        "title": "网络出现对成绩真实性的讨论",
        "content": (
            "随着姜萍事件关注度上升，部分网友开始围绕竞赛难度、解题过程、公开材料和师生关系提出疑问。"
            "讨论从单纯赞赏转向支持与怀疑并存。"
        ),
        "opinion_index": -0.20,
    },
    30: {
        "topic": JIANG_PING_TOPIC,
        "title": "争议围绕辅导、证明材料和舆论放大继续发酵",
        "content": (
            "围绕姜萍是否独立完成预赛、指导教师是否提供过度帮助、媒体是否过早塑造典型的讨论持续升温。"
            "反对者要求公开更多证据，支持者则认为质疑不应变成人身攻击。"
        ),
        "opinion_index": -0.45,
    },
    40: {
        "topic": JIANG_PING_TOPIC,
        "title": "决赛阶段结束，公众等待进一步结果",
        "content": (
            "姜萍参加的竞赛进入决赛和结果等待阶段。由于缺少新的可核验证据，公共讨论短暂转为观望，"
            "争议焦点集中在最终成绩能否回应外界疑问。"
        ),
        "opinion_index": 0.00,
    },
    55: {
        "topic": JIANG_PING_TOPIC,
        "title": "长时间缺少结果加剧不确定感",
        "content": (
            "在姜萍决赛结果和说明尚未充分公开的阶段，部分讨论转向对赛事流程、媒体报道节奏和公众情绪反转的质疑。"
            "群体态度进一步分化。"
        ),
        "opinion_index": -0.30,
    },
    70: {
        "topic": JIANG_PING_TOPIC,
        "title": "组委会公布竞赛有关情况说明",
        "content": (
            "据公开的组委会说明，姜萍在预选赛中存在违规情形，相关教师也被点名。"
            "这一说明使舆论明显转向质疑与反对，讨论焦点从个人故事扩展到竞赛公信力。"
        ),
        "opinion_index": -0.85,
    },
    85: {
        "topic": JIANG_PING_TOPIC,
        "title": "学校和相关方处理结果引发后续讨论",
        "content": (
            "姜萍事件后续报道继续讨论学校管理、教师责任、媒体报道边界和公众追捧机制。"
            "反对意见更多指向包装典型、流量叙事和制度性把关。"
        ),
        "opinion_index": -0.70,
    },
    95: {
        "topic": JIANG_PING_TOPIC,
        "title": "舆论开始反思事件中的媒体放大与教育焦虑",
        "content": (
            "事件后期，讨论逐渐从支持或反对姜萍本人转向反思媒体造神、教育公平想象、竞赛透明度和公众情绪反转。"
            "此时立场强度回落，但负面评价仍占重要位置。"
        ),
        "opinion_index": -0.20,
    },
}
