from __future__ import annotations

from typing import Any

from persona.opinion.scale import JIANG_PING_TOPIC
from .builder import build_runtime_from_spec
from .map_designs import default_town_map_design


AGENT_IDS = ["agent_1", "agent_2", "agent_3", "agent_4", "agent_5"]

# 默认小镇使用的系统新闻排期。极化实验使用独立场景，不读取这里的真实结果线。
NEWS_SCHEDULE: dict[int, dict] = {
    1: {
        "topic": JIANG_PING_TOPIC,
        "title": "阿里数学竞赛预赛结果引发关注",
        "content": (
            "公开报道显示，江苏涟水一名中专学生姜萍在2024阿里巴巴全球数学竞赛预赛中取得第12名。"
            "她的学校背景、年龄和排名形成强烈反差，社交平台开始大量转发励志叙事。"
        ),
        "opinion_index": 0.80,
    },
    8: {
        "topic": JIANG_PING_TOPIC,
        "title": "媒体集中报道姜萍的学习经历",
        "content": (
            "多家媒体继续报道姜萍自学高等数学、参加竞赛的经历。"
            "支持者认为这说明普通教育路径之外也可能出现突出个体，相关讨论以鼓励和赞赏为主。"
        ),
        "opinion_index": 0.65,
    },
    16: {
        "topic": JIANG_PING_TOPIC,
        "title": "网络出现对成绩真实性的讨论",
        "content": (
            "随着姜萍事件关注度上升，部分网友开始围绕竞赛难度、解题过程、公开材料和师生关系提出疑问。"
            "讨论从单纯赞赏转向支持与怀疑并存。"
        ),
        "opinion_index": -0.20,
    },
    26: {
        "topic": JIANG_PING_TOPIC,
        "title": "争议围绕辅导、证明材料和舆论放大继续发酵",
        "content": (
            "围绕姜萍是否独立完成预赛、指导教师是否提供过度帮助、媒体是否过早塑造典型的讨论持续升温。"
            "反对者要求公开更多证据，支持者则认为质疑不应变成人身攻击。"
        ),
        "opinion_index": -0.45,
    },
    36: {
        "topic": JIANG_PING_TOPIC,
        "title": "决赛阶段结束，公众等待进一步结果",
        "content": (
            "姜萍参加的竞赛进入决赛和结果等待阶段。由于缺少新的可核验证据，公共讨论短暂转为观望，"
            "争议焦点集中在最终成绩能否回应外界疑问。"
        ),
        "opinion_index": 0.00,
    },
    51: {
        "topic": JIANG_PING_TOPIC,
        "title": "长时间缺少结果加剧不确定感",
        "content": (
            "在姜萍决赛结果和说明尚未充分公开的阶段，部分讨论转向对赛事流程、媒体报道节奏和公众情绪反转的质疑。"
            "群体态度进一步分化。"
        ),
        "opinion_index": -0.30,
    },
    66: {
        "topic": JIANG_PING_TOPIC,
        "title": "组委会公布竞赛有关情况说明",
        "content": (
            "据公开的组委会说明，姜萍在预选赛中存在违规情形，相关教师也被点名。"
            "这一说明使舆论明显转向质疑与反对，讨论焦点从个人故事扩展到竞赛公信力。"
        ),
        "opinion_index": -0.85,
    },
    81: {
        "topic": JIANG_PING_TOPIC,
        "title": "学校和相关方处理结果引发后续讨论",
        "content": (
            "姜萍事件后续报道继续讨论学校管理、教师责任、媒体报道边界和公众追捧机制。"
            "反对意见更多指向包装典型、流量叙事和制度性把关。"
        ),
        "opinion_index": -0.70,
    },
    91: {
        "topic": JIANG_PING_TOPIC,
        "title": "舆论开始反思事件中的媒体放大与教育焦虑",
        "content": (
            "事件后期，讨论逐渐从支持或反对姜萍本人转向反思媒体造神、教育公平想象、竞赛透明度和公众情绪反转。"
            "此时立场强度回落，但负面评价仍占重要位置。"
        ),
        "opinion_index": -0.20,
    },
}


SPEC: dict[str, Any] = {
    "name": "default_town",
    "agent_ids": AGENT_IDS,
    "map_design": default_town_map_design(),
    "default_initial_satisfaction": {"satiety": 55.0, "relax": 55.0},
    "agents": [
        {"id": "agent_1", "position": [3, 3], "speaking_style": "沉稳、措辞谨慎", "initial_money": 8.0},
        {"id": "agent_2", "position": [4, 2], "speaking_style": "热情、喜欢分享", "initial_money": 14.0},
        {"id": "agent_3", "position": [6, 6], "speaking_style": "理性、措辞中立", "initial_money": 18.0},
        {"id": "agent_4", "position": [2, 9], "speaking_style": "直接、充满激情", "initial_money": 24.0},
        {"id": "agent_5", "position": [8, 4], "speaking_style": "温和、善于调解", "initial_money": 12.0},
    ],
    "offline_trust": [
        {"source": "agent_1", "target": "agent_2", "value": 0.75},
        {"source": "agent_1", "target": "agent_3", "value": 0.65},
        {"source": "agent_2", "target": "agent_1", "value": 0.75},
        {"source": "agent_2", "target": "agent_3", "value": 0.70},
        {"source": "agent_3", "target": "agent_1", "value": 0.65},
        {"source": "agent_3", "target": "agent_2", "value": 0.70},
        {"source": "agent_4", "target": "agent_5", "value": 0.80},
        {"source": "agent_5", "target": "agent_4", "value": 0.80},
        {"source": "agent_3", "target": "agent_5", "value": 0.62},
        {"source": "agent_5", "target": "agent_3", "value": 0.62},
    ],
    "online_trust": [
        {"source": "agent_1", "target": "agent_2", "value": 0.55},
        {"source": "agent_1", "target": "agent_3", "value": 0.60},
        {"source": "agent_2", "target": "agent_1", "value": 0.50},
        {"source": "agent_2", "target": "agent_4", "value": 0.35},
        {"source": "agent_3", "target": "agent_1", "value": 0.55},
        {"source": "agent_3", "target": "agent_4", "value": 0.55},
        {"source": "agent_3", "target": "agent_5", "value": 0.60},
        {"source": "agent_4", "target": "agent_5", "value": 0.65},
        {"source": "agent_4", "target": "agent_1", "value": 0.30},
        {"source": "agent_5", "target": "agent_4", "value": 0.60},
        {"source": "agent_5", "target": "agent_3", "value": 0.65},
    ],
    "follow_edges": [
        {"follower": "agent_1", "author": "agent_2"},
        {"follower": "agent_1", "author": "agent_3"},
        {"follower": "agent_2", "author": "agent_1"},
        {"follower": "agent_2", "author": "agent_3"},
        {"follower": "agent_2", "author": "agent_5"},
        {"follower": "agent_3", "author": "agent_1"},
        {"follower": "agent_3", "author": "agent_4"},
        {"follower": "agent_3", "author": "agent_5"},
        {"follower": "agent_4", "author": "agent_5"},
        {"follower": "agent_4", "author": "agent_3"},
        {"follower": "agent_5", "author": "agent_4"},
        {"follower": "agent_5", "author": "agent_3"},
        {"follower": "agent_5", "author": "agent_2"},
    ],
    "objects": [
        {"kind": "bed", "id": "bed_1", "position": [2, 2]},
        {"kind": "bed", "id": "bed_2", "position": [2, 4]},
        {"kind": "bed", "id": "bed_3", "position": [2, 6]},
        {"kind": "bed", "id": "bed_4", "position": [2, 8]},
        {"kind": "bed", "id": "bed_5", "position": [2, 10]},
        {"kind": "company", "id": "company_1", "position": [20, 3], "params": {"salary": 8, "relax_cost": 8}},
        {"kind": "company", "id": "company_2", "position": [22, 6], "params": {"salary": 12, "relax_cost": 12}},
        {"kind": "food_shop", "id": "shop_1", "position": [2, 18], "params": {"food_num": 20, "provide": 30, "price": 6}},
        {"kind": "food_shop", "id": "shop_2", "position": [4, 21], "params": {"food_num": 15, "provide": 20, "price": 4}},
        {"kind": "playground", "id": "playground_1", "position": [20, 20], "params": {"provide": 18, "price": 5}},
        {"kind": "food", "id": "food_1", "position": [10, 8], "params": {"num": 3, "provide": 20}},
        {"kind": "food", "id": "food_2", "position": [12, 14], "params": {"num": 3, "provide": 20}},
        {"kind": "food", "id": "food_3", "position": [8, 18], "params": {"num": 3, "provide": 20}},
        {"kind": "food", "id": "food_4", "position": [15, 5], "params": {"num": 3, "provide": 20}},
        {"kind": "food", "id": "food_5", "position": [16, 20], "params": {"num": 3, "provide": 20}},
    ],
    "map_memory": "地图包含住宅区、商业区、中心食物区、工作区和娱乐区；坐标格式为[row, col]，可根据观察和记忆导航。",
    "official_news_schedule": NEWS_SCHEDULE,
    "influencers": [],
    "influencer_schedule": {},
}


def build_runtime(**kwargs):
    """构建默认小镇场景。"""

    return build_runtime_from_spec(SPEC, **kwargs)
