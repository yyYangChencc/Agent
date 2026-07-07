from __future__ import annotations

import json
from typing import TYPE_CHECKING

from persona.logger import get_logger
from persona.llm.interface import JSON_OBJECT_RESPONSE_FORMAT
from persona.llm.json_utils import parse_json_object
from persona.opinion.scale import OPINION_NEUTRAL, clamp_opinion, get_opinion_topic_definition

if TYPE_CHECKING:
    from persona.config import AgentConfig
    from persona.llm.interface import LLMClient

logger = get_logger(__name__)


def evaluate_opinion(
    content: str,
    *,
    topic: str = "",
    llm: "LLMClient | None" = None,
    config: "AgentConfig | None" = None,
) -> float:
    """评估自然语言内容表达的观念分数。

    参数：
        content: 自然语言文本，例如社交帖子或对话内容。
        topic: 系统新闻主题；为空时使用通用正负向语义。
        llm: 可选 LLM 客户端，用于优先进行主题立场评估。
        config: 可选配置，控制 LLM/规则模式和回退行为。

    返回：
        [-1.0, 1.0] 范围内的观念分数；-1 表示强烈反对/负向，
        0 表示中立，1 表示强烈支持/正向。
    """
    text = str(content or "").strip()
    if not text:
        return OPINION_NEUTRAL

    mode = getattr(config, "post_opinion_scoring_mode", "rule")
    fallback_to_rule = getattr(config, "post_opinion_llm_fallback_to_rule", True)
    if mode == "llm" and llm is not None:
        try:
            return _score_with_llm(text, topic=topic, llm=llm)
        except Exception as exc:
            logger.warning("[OpinionScorer] LLM 发帖立场评分失败，回退规则: %s", exc)
            if not fallback_to_rule:
                return OPINION_NEUTRAL
    return _score_with_rules(text, topic=topic)


def _score_with_llm(content: str, *, topic: str, llm: "LLMClient") -> float:
    """使用项目统一 LLM 风格评估发帖文本立场。"""

    topic_definition = get_opinion_topic_definition(topic)
    system = (
        "你是社交平台文本立场评分器。请判断 user JSON 中 content 对 topic 所示系统新闻主题正面叙事的立场。"
        "分数必须在 -1 到 1 之间。"
        f"{topic_definition.direction_prompt}"
        "只输出 JSON，不要输出额外文字。"
    )
    user = json.dumps(
        {
            "topic": topic,
            "topic_narrative": topic_definition.narrative,
            "content": content,
            "output_schema": {
                "score": "float in [-1,1]",
                "reason": "short Chinese explanation",
            },
        },
        ensure_ascii=False,
    )
    raw = llm.generate(system, user, response_format=JSON_OBJECT_RESPONSE_FORMAT)
    payload = _parse_json_object(raw)
    return clamp_opinion(payload.get("score", OPINION_NEUTRAL))


def _score_with_rules(content: str, *, topic: str) -> float:
    """关键词规则兜底，保证无 LLM 时发帖立场不再全部为 0。"""

    definition = get_opinion_topic_definition(topic)
    positive_hits = [keyword for keyword in definition.positive_keywords if keyword in content]
    negative_hits = [keyword for keyword in definition.negative_keywords if keyword in content]
    score = 0.12 * len(positive_hits) - 0.12 * len(negative_hits)
    if not positive_hits and not negative_hits:
        return OPINION_NEUTRAL
    return clamp_opinion(max(-0.85, min(0.85, score)))


def _parse_json_object(raw: str) -> dict:
    """兼容纯 JSON 和 fenced JSON block。"""

    return parse_json_object(raw, context="post opinion scorer output")
