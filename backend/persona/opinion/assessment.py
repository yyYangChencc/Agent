from __future__ import annotations

import asyncio
import json
import re
from typing import TYPE_CHECKING

from persona.logger import get_logger
from persona.opinion.scale import JIANG_PING_TOPIC, clamp_opinion

if TYPE_CHECKING:
    from persona.agents.agent import Agent
    from persona.config import AgentConfig
    from persona.llm.interface import LLMClient

logger = get_logger(__name__)


class OpinionAssessmentCoordinator:
    """每个 tick 末评测智能体对系统新闻主题的观念。

    `agent.opinion` 是智能体对 `AgentConfig.default_opinion_topic` 的当前立场。
    本模块只评测这个系统新闻主题，不把 current_focus、task 或其他临时上下文当作评测主题。
    评测得到的 score 会写回 `agent.opinion`，不执行线上/线下加权平均式的公式化观念更新。
    """

    def __init__(self, config: "AgentConfig", llm: "LLMClient | None" = None):
        self.config = config
        self.llm = llm

    def assess_agent(self, agent: "Agent", tick: int) -> dict:
        topic = self._current_topic(agent)
        context = self._build_context(agent, topic)
        assessment_payload = self._assess_with_configured_method(agent, topic, context)
        return self._store_assessment(agent, tick, topic, context, assessment_payload)

    async def aassess_agent(self, agent: "Agent", tick: int) -> dict:
        topic = self._current_topic(agent)
        context = self._build_context(agent, topic)
        assessment_payload = await self._aassess_with_configured_method(agent, topic, context)
        return self._store_assessment(agent, tick, topic, context, assessment_payload)

    def _store_assessment(
        self,
        agent: "Agent",
        tick: int,
        topic: str,
        context: dict,
        assessment_payload: dict,
    ) -> dict:
        score = clamp_opinion(assessment_payload["score"])
        assessment = {
            "tick": tick,
            "topic": topic,
            "score": score,
            "source": assessment_payload["source"],
            "confidence": assessment_payload["confidence"],
            "reason": assessment_payload["reason"],
            "evidence": assessment_payload["evidence"],
            "current_focus": agent.current_focus,
            "task": agent.task,
            "context": context,
        }
        # 观念评测结果就是智能体当前持有的系统新闻主题 opinion。
        agent.opinion = score
        agent.last_opinion_assessment = assessment
        # opinion_scores 保留为前端展示和历史分析用的镜像；真实状态以 agent.opinion 为准。
        agent.opinion_scores[topic] = score
        agent.opinion_assessment_history.append(assessment)
        max_history = max(1, self.config.opinion_assessment_history_limit)
        if len(agent.opinion_assessment_history) > max_history:
            agent.opinion_assessment_history = agent.opinion_assessment_history[-max_history:]
        logger.debug(
            "[OpinionAssessment] tick=%d agent=%s topic=%s score=%.3f",
            tick,
            agent.id,
            topic,
            score,
        )
        return assessment

    def assess_all(self, agents: list["Agent"], tick: int) -> None:
        for agent in agents:
            self.assess_agent(agent, tick)

    async def aassess_all(self, agents: list["Agent"], tick: int) -> None:
        # 观念评测可能调用 LLM，因此在世界主循环中并发执行；单个智能体失败不影响其他智能体。
        results = await asyncio.gather(
            *(self.aassess_agent(agent, tick) for agent in agents),
            return_exceptions=True,
        )
        for agent, result in zip(agents, results):
            if isinstance(result, Exception):
                logger.warning(
                    "[OpinionAssessment] assessment failed for %s at tick=%d: %s",
                    agent.id,
                    tick,
                    result,
                )

    def _current_topic(self, agent: "Agent") -> str:
        # 观念评测只面向系统投放新闻配置的主题；current_focus 仅是 micro-reflect 的策略焦点。
        return self.config.default_opinion_topic

    def _assess_with_configured_method(self, agent: "Agent", topic: str, context: dict) -> dict:
        # 默认优先走 LLM；任何调用或解析异常都会回退到规则评测，避免中断 tick。
        if self.config.opinion_assessment_mode == "llm" and self.llm is not None:
            try:
                return self._assess_with_llm(agent, topic, context)
            except Exception as exc:
                logger.warning(
                    "[OpinionAssessment] LLM assessment failed for %s topic=%s: %s; fallback to rule",
                    agent.id,
                    topic,
                    exc,
                )
        return self._assess_with_rules(agent, topic, context)

    async def _aassess_with_configured_method(self, agent: "Agent", topic: str, context: dict) -> dict:
        if self.config.opinion_assessment_mode == "llm" and self.llm is not None:
            try:
                return await self._aassess_with_llm(agent, topic, context)
            except Exception as exc:
                logger.warning(
                    "[OpinionAssessment] async LLM assessment failed for %s topic=%s: %s; fallback to rule",
                    agent.id,
                    topic,
                    exc,
                )
        return self._assess_with_rules(agent, topic, context)

    def _assess_with_rules(self, agent: "Agent", topic: str, context: dict) -> dict:
        # 规则评测是 LLM 的兜底路径：以当前观念为锚点，只做小幅上下文修正。
        evidence_text = self._joined_context_text(context)
        keyword_adjustment, keyword_evidence = self._keyword_adjustment(topic, evidence_text)
        psychological_adjustment = self._psychological_adjustment(context.get("dynamic_role_card", {}))
        base_score = self._base_topic_score(agent, topic)
        evidence = []
        if keyword_evidence:
            evidence.extend(keyword_evidence)
        if psychological_adjustment:
            evidence.append(f"动态角色卡调整 {psychological_adjustment:+.3f}")
        if not evidence:
            evidence.append("未发现明确新证据，沿用当前观念状态")
        repeated_rule_evidence = self._is_repeated_rule_evidence(agent, topic, evidence)
        if repeated_rule_evidence:
            score = base_score
        else:
            score = clamp_opinion(base_score + keyword_adjustment + psychological_adjustment)
        reason = (
            f"以当前 opinion {base_score:.3f} 为锚点，结合系统新闻主题相关的近期经历、"
            "社交文本、记忆和动态角色卡进行规则评测。"
        )
        if repeated_rule_evidence:
            reason = "本轮规则评测证据与上一轮相同，保持当前系统新闻主题 opinion，避免重复累加。"
        return {
            "score": score,
            "source": "rule_context_assessment",
            "confidence": self._confidence(agent, context, evidence),
            "reason": reason,
            "evidence": evidence[:6],
        }

    def _assess_with_llm(self, agent: "Agent", topic: str, context: dict) -> dict:
        # prompt 保持项目统一风格：system 说明角色，user 传 JSON 上下文和输出 schema。
        system = (
            "你是生活在沙盒世界中的智能体，正在接受新闻观念调研，请结合你的近期经历、近期发帖/评论、相关记忆，给出 user JSON 中 topic 字段所示系统新闻主题的观念分数。"
            "你的观念表现受到动态角色卡约束，请给出符合动态角色卡的观念分数。"
            "previous_topic_score 只能作为上一轮记录参考，不能把同一证据重复累加。"
            "分数必须在 -1 到 1 之间，-1 表示强烈反对/负向，0 表示中立/不关心/未知，1 表示强烈支持/正向。"
            "只输出 JSON，不要输出额外文字。"
        )
        user = json.dumps(
            {
                "agent_id": agent.id,
                "topic": topic,
                "current_opinion": agent.opinion,
                "previous_topic_score": getattr(agent, "opinion_scores", {}).get(topic),
                "context": context,
                "output_schema": {
                    "score": "float in [-1,1]",
                    "confidence": "float in [0,1]",
                    "reason": "short Chinese explanation",
                    "evidence": ["short evidence strings"],
                },
            },
            ensure_ascii=False,
        )
        raw = self.llm.generate(system, user)
        payload = self._parse_llm_json(raw)
        score = clamp_opinion(payload.get("score", agent.opinion))
        confidence = self._clamp01(payload.get("confidence", 0.5))
        evidence = payload.get("evidence", [])
        if not isinstance(evidence, list):
            evidence = [str(evidence)]
        reason = payload.get("reason", "")
        if not isinstance(reason, str) or not reason:
            reason = "LLM 根据上下文完成观念评测。"
        return {
            "score": score,
            "source": "llm_context_assessment",
            "confidence": confidence,
            "reason": reason,
            "evidence": [str(item) for item in evidence[:6]],
        }

    async def _aassess_with_llm(self, agent: "Agent", topic: str, context: dict) -> dict:
        system = (
            "你是生活在沙盒世界中的智能体，正在接受新闻观念调研，请结合你的近期经历、近期发帖/评论、相关记忆，给出 user JSON 中 topic 字段所示系统新闻主题的观念分数。"
            "你的观念表现受到动态角色卡约束，请给出符合动态角色卡的观念分数。"
            "previous_topic_score 只能作为上一轮记录参考，不能把同一证据重复累加。"
            "分数必须在 -1 到 1 之间，-1 表示强烈反对/负向，0 表示中立/不关心/未知，1 表示强烈支持/正向。"
            "只输出 JSON，不要输出额外文字。"
        )
        user = json.dumps(
            {
                "agent_id": agent.id,
                "topic": topic,
                "current_opinion": agent.opinion,
                "previous_topic_score": getattr(agent, "opinion_scores", {}).get(topic),
                "context": context,
                "output_schema": {
                    "score": "float in [-1,1]",
                    "confidence": "float in [0,1]",
                    "reason": "short Chinese explanation",
                    "evidence": ["short evidence strings"],
                },
            },
            ensure_ascii=False,
        )
        raw = await self.llm.agenerate(system, user)
        payload = self._parse_llm_json(raw)
        score = clamp_opinion(payload.get("score", agent.opinion))
        confidence = self._clamp01(payload.get("confidence", 0.5))
        evidence = payload.get("evidence", [])
        if not isinstance(evidence, list):
            evidence = [str(evidence)]
        reason = payload.get("reason", "")
        if not isinstance(reason, str) or not reason:
            reason = "LLM 根据上下文完成观念评测。"
        return {
            "score": score,
            "source": "llm_context_assessment",
            "confidence": confidence,
            "reason": reason,
            "evidence": [str(item) for item in evidence[:6]],
        }

    def _parse_llm_json(self, raw: str) -> dict:
        # 兼容模型返回纯 JSON 或 ```json fenced block；非 JSON 对象会触发回退。
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text).strip()
            text = re.sub(r"```$", "", text).strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("LLM opinion assessment output is not a JSON object")
        return data

    def _base_topic_score(self, agent: "Agent", topic: str) -> float:
        # `agent.opinion` 是系统新闻主题立场的唯一权威状态，opinion_scores 只是历史/展示镜像。
        return clamp_opinion(agent.opinion)

    def _build_context(self, agent: "Agent", topic: str) -> dict:
        # 观念评测的上下文同时包含线下经历、线上表达、长期记忆和完整动态角色卡。
        recent_history = list(getattr(agent, "history", []) or [])[
            -self.config.opinion_assessment_recent_history:
        ]
        recent_social = self._recent_social_texts(agent)
        memory_query = self._memory_query(topic, recent_history, recent_social)
        memories = []
        try:
            memories = agent.recall(memory_query, context="social")
        except Exception as exc:
            logger.debug(
                "[OpinionAssessment] memory recall failed for %s topic=%s: %s",
                agent.id,
                topic,
                exc,
            )
        return {
            "topic": topic,
            "current_opinion": clamp_opinion(agent.opinion),
            "previous_topic_score": getattr(agent, "opinion_scores", {}).get(topic),
            "recent_experiences": recent_history,
            "recent_social": recent_social,
            "memories": memories,
            "dynamic_role_card": self._dynamic_role_card(agent),
            "needs": {
                "satisfaction": dict(agent.satisfaction),
                "effective_pressure": dict(agent.effective_pressure),
            },
        }

    def _memory_query(self, topic: str, recent_history: list[str], recent_social: list[str]) -> str:
        parts = [f"观念评测主题：{topic}"]
        parts.extend(recent_history[-3:])
        parts.extend(recent_social[-3:])
        return "\n".join(parts)

    def _recent_social_texts(self, agent: "Agent") -> list[str]:
        texts = []
        for post in getattr(agent, "post_history", [])[-self.config.opinion_assessment_recent_social:]:
            if hasattr(post, "show"):
                texts.append(post.show())
            else:
                texts.append(str(post))
        platform = getattr(agent, "platform", None)
        if platform is not None:
            for post in getattr(platform, "posts", [])[-20:]:
                comments = getattr(post, "comments_list", []) or []
                authored_comments = [
                    comment for comment in comments
                    if getattr(comment, "author_id", None) == agent.id
                ]
                for comment in authored_comments[-self.config.opinion_assessment_recent_social:]:
                    texts.append(f"评论帖子 {getattr(post, 'id', '')}: {getattr(comment, 'content', '')}")
        return texts[-self.config.opinion_assessment_recent_social:]

    def _joined_context_text(self, context: dict) -> str:
        chunks = []
        for field in ["recent_experiences", "recent_social", "memories"]:
            values = context.get(field, [])
            if isinstance(values, list):
                chunks.extend(str(item) for item in values)
            elif values:
                chunks.append(str(values))
        dynamic_role_card = context.get("dynamic_role_card", {})
        if dynamic_role_card:
            chunks.append(json.dumps(dynamic_role_card, ensure_ascii=False))
        return "\n".join(chunks)

    def _keyword_adjustment(self, topic: str, text: str) -> tuple[float, list[str]]:
        # 这是轻量启发式，不替代正式立场识别；主要用于 LLM 不可用时保持可运行。
        if not self._is_topic_relevant_text(topic, text):
            return 0.0, []
        if topic == JIANG_PING_TOPIC:
            positive_keywords = [
                "支持", "赞同", "同情", "鼓励", "励志", "突出个体", "正面叙事", "谨慎支持", "姜萍",
            ]
            negative_keywords = [
                "反对", "质疑", "怀疑", "违规", "造神", "公信力", "不透明", "包装典型", "流量叙事", "过度帮助",
            ]
        else:
            positive_keywords = ["支持", "赞同", "认可", "正面", "谨慎支持"]
            negative_keywords = ["反对", "质疑", "怀疑", "负面", "不透明"]
        positive_hits = [kw for kw in positive_keywords if kw in text]
        negative_hits = [kw for kw in negative_keywords if kw in text]
        adjustment = 0.02 * len(positive_hits) - 0.02 * len(negative_hits)
        adjustment = max(-0.12, min(0.12, adjustment))
        evidence = []
        if positive_hits:
            evidence.append(f"正向线索：{', '.join(positive_hits[:5])}")
        if negative_hits:
            evidence.append(f"负向线索：{', '.join(negative_hits[:5])}")
        return adjustment, evidence

    def _is_topic_relevant_text(self, topic: str, text: str) -> bool:
        if not topic:
            return False
        if topic in text:
            return True
        if topic == JIANG_PING_TOPIC:
            return any(alias in text for alias in ["姜萍", "阿里数学", "数学竞赛", "竞赛", "中专"])
        return False

    def _is_repeated_rule_evidence(self, agent: "Agent", topic: str, evidence: list[str]) -> bool:
        last_assessment = getattr(agent, "last_opinion_assessment", None)
        if not isinstance(last_assessment, dict):
            return False
        return (
            last_assessment.get("topic") == topic
            and last_assessment.get("source") == "rule_context_assessment"
            and last_assessment.get("evidence") == evidence[:6]
        )

    def _psychological_adjustment(self, dynamic_role_card: dict) -> float:
        if not isinstance(dynamic_role_card, dict):
            return 0.0
        role_card = dynamic_role_card.get("role_card_delta")
        if not isinstance(role_card, dict):
            role_card = dynamic_role_card
        summary = str(role_card.get("summary", ""))
        adjustment = 0.0
        if any(word in summary for word in ["经济", "安全", "资源", "不公平", "保障"]):
            adjustment += 0.03
        if any(word in summary for word in ["焦虑", "防御", "威胁", "疲惫"]):
            adjustment -= 0.02
        return max(-0.05, min(0.05, adjustment))

    def _confidence(self, agent: "Agent", context: dict, evidence: list[str]) -> float:
        history_size = len(getattr(agent, "history", []) or [])
        recent_activity = min(1.0, history_size / 10.0)
        evidence_bonus = min(0.2, 0.04 * len(evidence))
        memory_bonus = 0.1 if context.get("memories") else 0.0
        return self._clamp01(0.45 + 0.25 * recent_activity + evidence_bonus + memory_bonus)

    def _dynamic_role_card(self, agent: "Agent") -> dict:
        assessment = getattr(agent, "last_psychological_assessment", None)
        if not isinstance(assessment, dict):
            return {}
        role_card = assessment.get("role_card_delta")
        if not isinstance(role_card, dict):
            return {}
        return {
            "status": assessment.get("status"),
            "activated_needs": assessment.get("activated_needs", []),
            "role_card_delta": dict(role_card),
        }

    def _clamp01(self, value: float) -> float:
        return round(max(0.0, min(1.0, float(value))), 4)
