from __future__ import annotations

import asyncio
import hashlib
import json
from typing import TYPE_CHECKING

from persona.logger import get_logger
from persona.llm.interface import JSON_OBJECT_RESPONSE_FORMAT
from persona.llm.json_utils import parse_json_object
from persona.opinion.scale import clamp_opinion, get_opinion_topic_definition

if TYPE_CHECKING:
    from persona.agents.agent import Agent
    from persona.config import AgentConfig
    from persona.llm.interface import LLMClient

logger = get_logger(__name__)


class OpinionAssessmentCoordinator:
    """每个 tick 末评测智能体对系统新闻主题的观念。

    `agent.opinion` 是智能体对 `AgentConfig.default_opinion_topic` 的当前立场。
    分数表示对该主题正面叙事的态度，不是对 system 账号或某一条新闻帖子的相信度。
    本模块只评测这个系统新闻主题，不把 current_focus、task 或其他临时上下文当作评测主题。
    评测得到的 score 会写回 `agent.opinion`，不执行线上/线下加权平均式的公式化观念更新。
    """

    def __init__(self, config: "AgentConfig", llm: "LLMClient | None" = None):
        self.config = config
        self.llm = llm

    def assess_agent(self, agent: "Agent", tick: int) -> dict:
        topic = self._current_topic(agent)
        if not self.should_assess_agent(agent, tick):
            return self._unchanged_assessment(agent, tick, topic, reason="no_new_evidence")
        context = self._build_context(agent, topic)
        if not self._context_has_assessment_evidence(context):
            return self._unchanged_assessment(agent, tick, topic, context=context, reason="no_cycle_evidence")
        assessment_payload = self._assess_with_configured_method(agent, topic, context)
        return self._store_assessment(agent, tick, topic, context, assessment_payload)

    async def aassess_agent(self, agent: "Agent", tick: int) -> dict:
        topic = self._current_topic(agent)
        if not self.should_assess_agent(agent, tick):
            return self._unchanged_assessment(agent, tick, topic, reason="no_new_evidence")
        context = await self._abuild_context(agent, topic)
        if not self._context_has_assessment_evidence(context):
            return self._unchanged_assessment(agent, tick, topic, context=context, reason="no_cycle_evidence")
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
        before_score = clamp_opinion(agent.opinion)
        score = clamp_opinion(assessment_payload["score"])
        assessment = {
            "tick": tick,
            "topic": topic,
            "before_score": before_score,
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
        agent.last_opinion_before_assessment = before_score
        agent.opinion = score
        agent.last_opinion_assessment = assessment
        # opinion_scores 保留为前端展示和历史分析用的镜像；真实状态以 agent.opinion 为准。
        agent.opinion_scores[topic] = score
        agent.opinion_assessment_history.append(assessment)
        agent._last_opinion_assessment_tick = tick
        agent._last_opinion_evidence_signature = self._evidence_signature(context)
        # 本周期实际看过的当前主题帖子已经参与评测，评测完成后清空，避免重复影响后续分数。
        agent.opinion_seen_posts_buffer = []
        mem = getattr(agent, "mem", None)
        if mem is not None and hasattr(mem, "store_opinion_assessment"):
            # 观念评测的 reason/evidence 是后续 opinion 查询的重要证据，写入 reflective 记忆。
            mem.store_opinion_assessment(agent.id, assessment)
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

    def should_assess_agent(self, agent: "Agent", tick: int) -> bool:
        """事件触发加间隔兜底；没有新证据时不进入 LLM 评测。"""

        if not getattr(self.config, "opinion_assessment_triggered_only", True):
            return True
        topic = self._current_topic(agent)
        if self._seen_posts_since_last_assessment(agent):
            context = self._light_context_for_signature(agent, topic)
            return self._evidence_signature(context) != getattr(agent, "_last_opinion_evidence_signature", "")
        interval = max(1, int(getattr(self.config, "opinion_assessment_interval", 5)))
        last_tick = int(getattr(agent, "_last_opinion_assessment_tick", 0) or 0)
        return tick - last_tick >= interval and self._has_cycle_evidence(agent, topic)

    def _unchanged_assessment(
        self,
        agent: "Agent",
        tick: int,
        topic: str,
        *,
        context: dict | None = None,
        reason: str,
    ) -> dict:
        """无新证据时沿用当前 opinion，不调用 LLM，也不写入观念评测记忆。"""

        score = clamp_opinion(agent.opinion)
        return {
            "tick": tick,
            "topic": topic,
            "score": score,
            "source": "unchanged_no_new_evidence",
            "confidence": 1.0,
            "reason": reason,
            "evidence": [],
            "current_focus": agent.current_focus,
            "task": agent.task,
            "context": context or {},
        }

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
        topic_definition = get_opinion_topic_definition(topic)
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
            raw_score = clamp_opinion(base_score + keyword_adjustment + psychological_adjustment)
            score = self._limited_score(base_score, raw_score)
        reason = (
            f"以当前 opinion {base_score:.3f} 为锚点，评测对象是{topic_definition.narrative}。结合本周期实际看过的当前主题帖子、"
            "补充记忆和动态角色卡进行规则评测。"
        )
        if repeated_rule_evidence:
            reason = f"本轮规则评测证据与上一轮相同，保持当前对{topic_definition.narrative}的 opinion，避免重复累加。"
        return {
            "score": score,
            "source": "rule_context_assessment",
            "confidence": self._confidence(agent, context, evidence),
            "reason": reason,
            "evidence": evidence[:6],
        }

    def _assess_with_llm(self, agent: "Agent", topic: str, context: dict) -> dict:
        # prompt 保持项目统一风格：system 说明角色，user 传 JSON 上下文和输出 schema。
        topic_definition = get_opinion_topic_definition(topic)
        system = (
            "你是生活在沙盒世界中的智能体，正在接受新闻观念调研，请结合你的近期经历、近期发帖/评论、相关记忆，给出 user JSON 中 topic 字段所示系统新闻主题的观念分数。"
            f"本次分数评测对象是“{topic_definition.narrative}”，不是对 system 账号或某一条新闻帖子的相信度。"
            "你的观念表现受到动态角色卡约束，请给出符合动态角色卡的观念分数。"
            "previous_topic_score 只能作为上一轮记录参考，不能把同一证据重复累加。"
            f"分数必须在 -1 到 1 之间，{topic_definition.direction_prompt}"
            "只输出 JSON，不要输出额外文字。"
        )
        user = json.dumps(
            {
                "agent_id": agent.id,
                "topic": topic,
                "current_opinion": agent.opinion,
                "previous_topic_score": getattr(agent, "opinion_scores", {}).get(topic),
                "topic_narrative": topic_definition.narrative,
                "score_direction": topic_definition.direction_prompt,
                "opinion_scale": topic_definition.scale,
                "context": context,
                "output_schema": {
                    "delta": "float in [-1,1], optional change from current_opinion",
                    "score": "float in [-1,1], optional absolute score after this assessment",
                    "confidence": "float in [0,1]",
                    "reason": "short Chinese explanation",
                    "evidence": ["short evidence strings"],
                },
            },
            ensure_ascii=False,
        )
        raw = self.llm.generate(system, user, response_format=JSON_OBJECT_RESPONSE_FORMAT)
        payload = self._parse_llm_json(raw)
        score = self._score_from_llm_payload(payload, agent.opinion)
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
        topic_definition = get_opinion_topic_definition(topic)
        system = (
            "你是生活在沙盒世界中的智能体，正在接受新闻观念调研，请结合你的近期经历、近期发帖/评论、相关记忆，给出 user JSON 中 topic 字段所示系统新闻主题的观念分数。"
            f"本次分数评测对象是“{topic_definition.narrative}”，不是对 system 账号或某一条新闻帖子的相信度。"
            "你的观念表现受到动态角色卡约束，请给出符合动态角色卡的观念分数。"
            "previous_topic_score 只能作为上一轮记录参考，不能把同一证据重复累加。"
            f"分数必须在 -1 到 1 之间，{topic_definition.direction_prompt}"
            "只输出 JSON，不要输出额外文字。"
        )
        user = json.dumps(
            {
                "agent_id": agent.id,
                "topic": topic,
                "current_opinion": agent.opinion,
                "previous_topic_score": getattr(agent, "opinion_scores", {}).get(topic),
                "topic_narrative": topic_definition.narrative,
                "score_direction": topic_definition.direction_prompt,
                "opinion_scale": topic_definition.scale,
                "context": context,
                "output_schema": {
                    "delta": "float in [-1,1], optional change from current_opinion",
                    "score": "float in [-1,1], optional absolute score after this assessment",
                    "confidence": "float in [0,1]",
                    "reason": "short Chinese explanation",
                    "evidence": ["short evidence strings"],
                },
            },
            ensure_ascii=False,
        )
        raw = await self.llm.agenerate(system, user, response_format=JSON_OBJECT_RESPONSE_FORMAT)
        payload = self._parse_llm_json(raw)
        score = self._score_from_llm_payload(payload, agent.opinion)
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
        return parse_json_object(raw, context="LLM opinion assessment output")

    def _base_topic_score(self, agent: "Agent", topic: str) -> float:
        # `agent.opinion` 是系统新闻主题立场的唯一权威状态，opinion_scores 只是历史/展示镜像。
        return clamp_opinion(agent.opinion)

    def _build_context(self, agent: "Agent", topic: str) -> dict:
        # 观念评测的周期证据只来自本周期实际看过的当前主题帖子。
        topic_definition = get_opinion_topic_definition(topic)
        recent_social = self._recent_social_texts(agent, topic)
        memory_query = self._memory_query(topic, recent_social)
        memories = []
        try:
            # 使用 opinion_assessment 场景召回，优先取社交、对话和历史观念证据。
            memories = agent.recall(memory_query, context="opinion_assessment")
        except Exception as exc:
            logger.debug(
                "[OpinionAssessment] memory recall failed for %s topic=%s: %s",
                agent.id,
                topic,
                exc,
            )
        return {
            "topic": topic,
            "topic_narrative": topic_definition.narrative,
            "score_direction": topic_definition.direction_prompt,
            "current_opinion": clamp_opinion(agent.opinion),
            "previous_topic_score": getattr(agent, "opinion_scores", {}).get(topic),
            "recent_social": recent_social,
            "seen_posts": recent_social,
            "memories": memories,
            "dynamic_role_card": self._dynamic_role_card(agent),
            "needs": {
                "satisfaction": dict(agent.satisfaction),
                "effective_pressure": dict(agent.effective_pressure),
            },
        }

    def _memory_query(self, topic: str, recent_social: list[dict]) -> str:
        topic_definition = get_opinion_topic_definition(topic)
        parts = [f"观念评测主题：{topic}", f"评测对象：{topic_definition.narrative}"]
        parts.extend(json.dumps(item, ensure_ascii=False) for item in recent_social[-3:])
        return "\n".join(parts)

    def _legacy_recent_social_texts(self, agent: "Agent") -> list[str]:
        texts = []
        # 旧路径保留为空实现，实际使用后面的同名新版函数；禁止扫描历史帖或平台全量帖子。
        for post in []:
            if hasattr(post, "show"):
                texts.append(post.show())
            else:
                texts.append(str(post))
        platform = None
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

    async def _abuild_context(self, agent: "Agent", topic: str) -> dict:
        """异步构造观念评测上下文，避免评测阶段同步记忆召回阻塞事件循环。"""

        topic_definition = get_opinion_topic_definition(topic)
        recent_social = self._recent_social_texts(agent, topic)
        memory_query = self._memory_query(topic, recent_social)
        memories = []
        try:
            # 观念评测召回只补充社交/对话事实，旧 opinion_assessment 不再作为证据。
            memories = await agent.arecall(memory_query, context="opinion_assessment")
        except Exception as exc:
            logger.debug(
                "[OpinionAssessment] async memory recall failed for %s topic=%s: %s",
                agent.id,
                topic,
                exc,
            )
        return {
            "topic": topic,
            "topic_narrative": topic_definition.narrative,
            "score_direction": topic_definition.direction_prompt,
            "current_opinion": clamp_opinion(agent.opinion),
            "previous_topic_score": getattr(agent, "opinion_scores", {}).get(topic),
            "recent_social": recent_social,
            "seen_posts": recent_social,
            "memories": memories,
            "dynamic_role_card": self._dynamic_role_card(agent),
            "needs": {
                "satisfaction": dict(agent.satisfaction),
                "effective_pressure": dict(agent.effective_pressure),
            },
        }

    def _recent_social_texts(self, agent: "Agent", topic: str) -> list[dict]:
        """观念评测只使用本评测周期内实际看过且 topic 等于当前主题的帖子。"""

        posts = getattr(agent, "opinion_seen_posts_buffer", []) or []
        out = []
        for post in posts:
            # 只按帖子 topic 判断是否属于当前观念主题。
            if isinstance(post, dict) and self._post_matches_topic(post, topic):
                out.append(self._post_evidence_snapshot(post))
        return out[-self.config.opinion_assessment_recent_social:]

    def _post_matches_topic(self, post: dict, topic: str) -> bool:
        """按帖子 topic 判断是否属于当前观念主题。"""

        return bool(topic) and str(post.get("topic") or "") == topic

    def _post_evidence_snapshot(self, post: dict) -> dict:
        """保留帖子原始 JSON 的关键字段，供 LLM 直接评估证据。"""

        return {
            "id": post.get("id"),
            "author_id": post.get("author_id"),
            "topic": post.get("topic"),
            "content": post.get("content"),
            "time": post.get("time"),
            "likes": post.get("likes"),
            "dislikes": post.get("dislikes"),
            "comments_count": post.get("comments_count"),
            "opinion_index": post.get("opinion_index"),
            "is_news": post.get("is_news"),
            "is_rumor": post.get("is_rumor"),
            "source_type": post.get("source_type"),
            "comments": post.get("comments") if isinstance(post.get("comments"), list) else [],
        }

    def _joined_context_text(self, context: dict) -> str:
        chunks = []
        for field in ["recent_social", "memories"]:
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
        topic_definition = get_opinion_topic_definition(topic)
        positive_keywords = topic_definition.positive_keywords
        negative_keywords = topic_definition.negative_keywords
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

    def _seen_posts_since_last_assessment(self, agent: "Agent") -> bool:
        """帖子缓冲非空表示智能体本周期实际看过当前主题帖子。"""

        topic = self._current_topic(agent)
        return bool(self._recent_social_texts(agent, topic))

    def _has_cycle_evidence(self, agent: "Agent", topic: str) -> bool:
        """间隔兜底只在实际看过当前主题新帖子时触发，不因时间到就重跑 LLM。"""

        return self._seen_posts_since_last_assessment(agent)

    def _context_has_assessment_evidence(self, context: dict) -> bool:
        """没有本周期实际看过的当前主题帖子时沿用当前 opinion，不写新评测记忆。"""

        return bool(context.get("seen_posts"))

    def _light_context_for_signature(self, agent: "Agent", topic: str) -> dict:
        """只用周期证据生成签名，不额外触发记忆检索。"""

        return {
            "seen_posts": self._recent_social_texts(agent, topic),
        }

    def _evidence_signature(self, context: dict) -> str:
        """给本轮观念证据生成稳定签名，避免同一证据重复评测。"""

        evidence = {
            "seen_posts": context.get("seen_posts") or context.get("recent_social") or [],
        }
        text = json.dumps(evidence, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _score_from_llm_payload(self, payload: dict, current_score: float) -> float:
        """LLM 可输出 delta 或 score，但单次变化必须受配置上限约束。"""

        if "delta" in payload:
            try:
                raw_score = clamp_opinion(current_score + float(payload.get("delta")))
            except (TypeError, ValueError):
                raw_score = current_score
        else:
            try:
                raw_score = clamp_opinion(float(payload.get("score", current_score)))
            except (TypeError, ValueError):
                raw_score = current_score
        return self._limited_score(current_score, raw_score)

    def _limited_score(self, current_score: float, raw_score: float) -> float:
        """限制单次观念变化幅度，避免一条证据造成过度跳变。"""

        max_delta = max(0.0, float(getattr(self.config, "opinion_max_delta_per_assessment", 0.25)))
        current = clamp_opinion(current_score)
        target = clamp_opinion(raw_score)
        delta = max(-max_delta, min(max_delta, target - current))
        return clamp_opinion(current + delta)

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
        if not getattr(self.config, "dynamic_role_card_enabled", True):
            return {}
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
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = 0.5
        return round(max(0.0, min(1.0, number)), 4)
