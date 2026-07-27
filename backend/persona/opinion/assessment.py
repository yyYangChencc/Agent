from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import TYPE_CHECKING

from persona.logger import get_logger
from persona.opinion.scale import classify_voting_stance
from persona.llm.interface import JSON_OBJECT_RESPONSE_FORMAT
from persona.llm.debug_trace import annotate_current_llm_trace, trace_llm_call
from persona.llm.json_utils import parse_json_object
from persona.opinion.scale import (
    VOTING_POLARIZATION_ROLES,
    VOTING_ROLE_OPPOSE,
    VOTING_ROLE_SUPPORT,
    VOTING_ROLE_UNKNOWN,
    clamp_opinion,
    get_opinion_topic_definition,
)
from persona.opinion.flan_scorer import FlanT5OpinionScorer

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

    def __init__(
        self,
        config: "AgentConfig",
        llm: "LLMClient | None" = None,
        flan_scorer: FlanT5OpinionScorer | None = None,
    ):
        self.config = config
        self.llm = llm
        self.flan_scorer = flan_scorer or FlanT5OpinionScorer(
            str(getattr(config, "opinion_flan_model_name", "google/flan-t5-large"))
        )
        # 投票使用独立信号量和启动间隔，不改变其他 LLM 请求的全局上限。
        self._voting_concurrency = max(1, int(getattr(config, "opinion_voting_max_concurrent_requests", 3)))
        self._voting_request_interval = max(
            0.0,
            float(getattr(config, "opinion_voting_request_interval_seconds", 1.0)),
        )
        self._voting_semaphore = None
        self._voting_semaphore_loop = None
        self._voting_start_lock = None
        self._voting_start_lock_loop = None
        self._voting_next_start_at = 0.0

    def assess_agent(self, agent: "Agent", tick: int) -> dict:
        topic = self._current_topic(agent)
        if not self.should_assess_agent(agent, tick):
            return self._unchanged_assessment(agent, tick, topic, reason="no_new_evidence")
        if self._assessment_mode() == "llm_voting":
            return self._unchanged_assessment(agent, tick, topic, reason="voting_runs_after_simulation")
        context = self._build_context(agent, topic, tick)
        if not self._context_has_assessment_evidence(context):
            return self._unchanged_assessment(agent, tick, topic, context=context, reason="no_window_evidence")
        try:
            assessment_payload = self._assess_with_configured_method(agent, topic, context)
        except Exception as exc:
            # 上游 LLM 或 FLAN 失败时跳过本轮，禁止用另一套量表污染 opinion。
            logger.warning(
                "[OpinionAssessment] assessment failed for %s topic=%s: %s; skip this round",
                agent.id,
                topic,
                exc,
            )
            return self._unchanged_assessment(
                agent,
                tick,
                topic,
                context=context,
                reason="llm_assessment_failed",
                source="skipped_llm_failure",
            )
        return self._store_assessment(agent, tick, topic, context, assessment_payload)

    async def aassess_agent(self, agent: "Agent", tick: int) -> dict:
        topic = self._current_topic(agent)
        if not self.should_assess_agent(agent, tick):
            return self._unchanged_assessment(agent, tick, topic, reason="no_new_evidence")
        if self._assessment_mode() == "llm_voting":
            return self._unchanged_assessment(agent, tick, topic, reason="voting_runs_after_simulation")
        context = await self._abuild_context(agent, topic, tick)
        if not self._context_has_assessment_evidence(context):
            return self._unchanged_assessment(agent, tick, topic, context=context, reason="no_window_evidence")
        try:
            assessment_payload = await self._aassess_with_configured_method(agent, topic, context)
        except Exception as exc:
            # 异步路径与同步路径保持相同的跳过语义。
            logger.warning(
                "[OpinionAssessment] async assessment failed for %s topic=%s: %s; skip this round",
                agent.id,
                topic,
                exc,
            )
            return self._unchanged_assessment(
                agent,
                tick,
                topic,
                context=context,
                reason="llm_assessment_failed",
                source="skipped_llm_failure",
            )
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
            "current_honest_belief": assessment_payload.get("current_honest_belief", ""),
            "flan_rating": assessment_payload.get("flan_rating"),
            "flan_model": assessment_payload.get("flan_model", ""),
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

    def _store_voting(
        self,
        agent: "Agent",
        tick: int,
        topic: str,
        speech_history: list[dict],
        voting_payload: dict,
        *,
        window_start_tick: int | None = None,
        window_end_tick: int | None = None,
    ) -> dict:
        """独立保存投票结果，不把投票合成为 agent.opinion。"""

        option_roles = self._voting_option_roles(topic, voting_payload["options"])
        vote_metrics = self._voting_metrics(voting_payload, option_roles)
        stance_metrics = classify_voting_stance(
            requested_voters=voting_payload["requested_voters"],
            successful_votes=voting_payload["successful_votes"],
            choice_counts=voting_payload["choice_counts"],
            option_roles=option_roles,
        )
        voting = {
            "agent_id": agent.id,
            "tick": tick,
            "topic": topic,
            "method": "llm_voting",
            "window_start_tick": (
                int(window_start_tick)
                if window_start_tick is not None
                else max(1, tick - self._assessment_interval() + 1)
            ),
            "window_end_tick": int(window_end_tick) if window_end_tick is not None else tick,
            "options": voting_payload["options"],
            "requested_voters": voting_payload["requested_voters"],
            "successful_votes": voting_payload["successful_votes"],
            "failed_votes": voting_payload["failed_votes"],
            "choice_counts": voting_payload["choice_counts"],
            "option_roles": option_roles,
            **vote_metrics,
            **stance_metrics,
            "votes": voting_payload["votes"],
            "speech_history": speech_history,
            "skipped_reason": voting_payload.get("skipped_reason", ""),
        }
        agent.last_opinion_voting = voting
        agent.opinion_voting_history.append(voting)
        agent._last_opinion_assessment_tick = tick
        agent.opinion_seen_posts_buffer = []
        max_history = max(1, self.config.opinion_assessment_history_limit)
        if len(agent.opinion_voting_history) > max_history:
            agent.opinion_voting_history = agent.opinion_voting_history[-max_history:]
        logger.debug(
            "[OpinionVoting] tick=%d agent=%s votes=%d failed=%d",
            tick,
            agent.id,
            voting["successful_votes"],
            voting["failed_votes"],
        )
        return voting

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

    async def afinalize_voting(self, agents: list["Agent"], final_tick: int) -> list[dict]:
        """模拟结束后按固定十步窗口评测全部发帖和评论。"""

        window_size = max(1, int(getattr(self.config, "opinion_voting_window_size", 10)))
        all_results = []
        for window_start in range(1, max(0, int(final_tick)) + 1, window_size):
            window_end = min(window_start + window_size - 1, int(final_tick))
            results = await asyncio.gather(
                *(
                    self._aposthoc_vote_agent(agent, window_start, window_end)
                    for agent in agents
                ),
                return_exceptions=True,
            )
            for agent, result in zip(agents, results):
                if isinstance(result, Exception):
                    logger.warning(
                        "[OpinionVoting] posthoc vote failed agent=%s window=%d-%d: %s",
                        agent.id,
                        window_start,
                        window_end,
                        result,
                    )
                    continue
                all_results.append(result)
            completed = [result for result in results if isinstance(result, dict)]
            logger.info(
                "[OpinionVoting] posthoc window=%d-%d agents=%d requested=%d successful=%d skipped=%d",
                window_start,
                window_end,
                len(agents),
                sum(int(item.get("requested_voters", 0) or 0) for item in completed),
                sum(int(item.get("successful_votes", 0) or 0) for item in completed),
                sum(bool(item.get("skipped_reason")) for item in completed),
            )
        return all_results

    def finalize_voting(self, agents: list["Agent"], final_tick: int) -> list[dict]:
        """同步归档入口；按十步窗口依次完成结束后投票。"""

        window_size = max(1, int(getattr(self.config, "opinion_voting_window_size", 10)))
        all_results = []
        for window_start in range(1, max(0, int(final_tick)) + 1, window_size):
            window_end = min(window_start + window_size - 1, int(final_tick))
            for agent in agents:
                topic = self._current_topic(agent)
                speech_history = self._online_speech_window(agent, window_start, window_end)
                if self._skip_empty_voting_window(speech_history):
                    voting_payload = self._empty_voting_payload(topic)
                else:
                    voting_payload = self._assess_with_llm_voting(
                        agent,
                        topic,
                        speech_history,
                        trace_tick=window_end,
                    )
                all_results.append(
                    self._store_voting(
                        agent,
                        window_end,
                        topic,
                        speech_history,
                        voting_payload,
                        window_start_tick=window_start,
                        window_end_tick=window_end,
                    )
                )
        return all_results

    async def _aposthoc_vote_agent(self, agent: "Agent", window_start: int, window_end: int) -> dict:
        """执行单个智能体单个结束后窗口的全部投票。"""

        topic = self._current_topic(agent)
        speech_history = self._online_speech_window(agent, window_start, window_end)
        if self._skip_empty_voting_window(speech_history):
            voting_payload = self._empty_voting_payload(topic)
        else:
            voting_payload = await self._aassess_with_llm_voting(
                agent,
                topic,
                speech_history,
                trace_tick=window_end,
            )
        return self._store_voting(
            agent,
            window_end,
            topic,
            speech_history,
            voting_payload,
            window_start_tick=window_start,
            window_end_tick=window_end,
        )

    def _current_topic(self, agent: "Agent") -> str:
        # 观念评测只面向系统投放新闻配置的主题；current_focus 仅是 micro-reflect 的策略焦点。
        return self.config.default_opinion_topic

    def should_assess_agent(self, agent: "Agent", tick: int) -> bool:
        """只在固定窗口末端执行观念评测；投票在模拟结束后执行。"""

        if self._assessment_mode() == "llm_voting":
            return False
        interval = self._assessment_interval()
        return int(tick) > 0 and int(tick) % interval == 0

    def _unchanged_assessment(
        self,
        agent: "Agent",
        tick: int,
        topic: str,
        *,
        context: dict | None = None,
        reason: str,
        source: str = "unchanged_no_new_evidence",
    ) -> dict:
        """无新证据时沿用当前 opinion，不调用 LLM，也不写入观念评测记忆。"""

        score = clamp_opinion(agent.opinion)
        return {
            "tick": tick,
            "topic": topic,
            "score": score,
            "source": source,
            "confidence": 1.0,
            "reason": reason,
            "evidence": [],
            "current_focus": agent.current_focus,
            "task": agent.task,
            "context": context or {},
        }

    def _assess_with_configured_method(self, agent: "Agent", topic: str, context: dict) -> dict:
        # llm_as_judge 使用单一量表；失败由调用方记录并跳过本轮。
        if self._assessment_mode() == "llm_as_judge":
            if self.llm is None:
                raise RuntimeError("llm_as_judge requires an LLM client")
            return self._assess_with_llm(agent, topic, context)
        return self._assess_with_rules(agent, topic, context)

    async def _aassess_with_configured_method(self, agent: "Agent", topic: str, context: dict) -> dict:
        if self._assessment_mode() == "llm_as_judge":
            if self.llm is None:
                raise RuntimeError("llm_as_judge requires an LLM client")
            return await self._aassess_with_llm(agent, topic, context)
        return self._assess_with_rules(agent, topic, context)

    def _assessment_mode(self) -> str:
        """返回配置中的精确观念评测方式。"""

        return str(getattr(self.config, "opinion_assessment_mode", "llm_as_judge"))

    def _assessment_interval(self) -> int:
        """返回投票周期及发言窗口共同使用的时间步数。"""

        return max(1, int(getattr(self.config, "opinion_assessment_interval", 5)))

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
        # 通用 LLM 只生成自然语言观念，评分完全交给本地 FLAN 分类器。
        topic_definition = get_opinion_topic_definition(topic)
        system, user = self._honest_belief_prompt(agent, topic, context)
        trace_tick = int(context.get("window_end_tick") or agent.world.time)
        with trace_llm_call(
            self.llm,
            agent_id=agent.id,
            tick=trace_tick,
            stage="opinion_honest_belief",
            metadata={"topic": topic},
        ):
            raw = self.llm.generate(system, user, response_format=JSON_OBJECT_RESPONSE_FORMAT)
            payload = self._parse_llm_json(raw)
            self._validate_honest_belief_payload(payload)
        return self._score_honest_belief(agent, topic_definition, payload, trace_tick)

    async def _aassess_with_llm(self, agent: "Agent", topic: str, context: dict) -> dict:
        topic_definition = get_opinion_topic_definition(topic)
        system, user = self._honest_belief_prompt(agent, topic, context)
        trace_tick = int(context.get("window_end_tick") or agent.world.time)
        with trace_llm_call(
            self.llm,
            agent_id=agent.id,
            tick=trace_tick,
            stage="opinion_honest_belief",
            metadata={"topic": topic},
        ):
            raw = await self.llm.agenerate(system, user, response_format=JSON_OBJECT_RESPONSE_FORMAT)
            payload = self._parse_llm_json(raw)
            self._validate_honest_belief_payload(payload)
        return await asyncio.to_thread(
            self._score_honest_belief,
            agent,
            topic_definition,
            payload,
            trace_tick,
        )

    def _honest_belief_prompt(self, agent: "Agent", topic: str, context: dict) -> tuple[str, str]:
        """要求智能体依据上一轮状态和当前窗口发言生成简短立场反应。"""

        topic_definition = get_opinion_topic_definition(topic)
        role_card_instruction = (
            "Use the provided dynamic_role_card only as a bounded psychological template; it cannot replace or invent evidence."
            if getattr(self.config, "dynamic_role_card_enabled", True)
            and getattr(self.config, "dynamic_role_card_opinion_enabled", True)
            else "The current experiment disables the opinion role card. Do not infer, invent, or cite any role-card influence."
        )
        system = (
            f"{role_card_instruction}\\n"
            "你是 user JSON 中 agent_id 对应的智能体。"
            "请参考上一次观念与上一次自然语言反应，并结合本窗口内自己的全部发帖、评论以及实际看到的其他人发言，"
            "生成一段简短的第一人称自然语言反应，清楚阐述自己对指定议题的当前立场。"
            "必须区分自己的表达与他人的表达；他人的发言只是你看到的信息，不能写成自己的经历。"
            "historical_memory 是带来源标签的历史背景，不是本窗口新证据；不得把它改写成本窗口的新发帖、评论、观察或亲身经历，"
            "也不得把其中的数据库引用写入本窗口 evidence_ids。"
            "若新证据不足以改变立场，应保持与上一次反应一致。不得输出任何分数，不得补充 JSON 中没有的经历。只输出 JSON。"
        )
        user = json.dumps(
            {
                "agent_id": agent.id,
                "topic": topic,
                "topic_statement": topic_definition.narrative,
                "window_start_tick": context.get("window_start_tick"),
                "window_end_tick": context.get("window_end_tick"),
                "previous_honest_belief": context.get("previous_honest_belief", ""),
                "dynamic_role_card": context.get("dynamic_role_card", {}),
                "evidence_counts": context.get("evidence_counts", {}),
                "evidence": {
                    "self_authored_posts": context.get("self_authored_posts", []),
                    "self_authored_comments": context.get("self_authored_comments", []),
                    "observed_other_speech": context.get("observed_other_speech", []),
                    "likes_and_dislikes": context.get("likes_and_dislikes", []),
                },
                "historical_memory": context.get("recalled_memories", []),
                "output_schema": {
                    "current_honest_belief": "brief natural-language first-person stance",
                    "evidence_ids": ["exact post or comment identifiers used in this window"],
                },
            },
            ensure_ascii=False,
        )
        return system, user

    def _score_honest_belief(
        self,
        agent: "Agent",
        topic_definition,
        payload: dict,
        trace_tick: int,
    ) -> dict:
        """使用 FLAN 五级评分并映射到项目的连续观念区间。"""

        honest_belief, evidence_ids = self._validate_honest_belief_payload(payload)
        with trace_llm_call(
            self.llm,
            agent_id=agent.id,
            tick=trace_tick,
            stage="opinion_flan_scoring",
            metadata={"model": self.flan_scorer.model_name},
        ):
            rating = self.flan_scorer.score(
                topic_statement=topic_definition.narrative,
                honest_belief=honest_belief,
            )
        return {
            "score": rating / 2.0,
            "source": "llm_as_judge",
            "confidence": 0.0,
            "reason": honest_belief,
            "evidence": [str(item) for item in evidence_ids[:12]],
            "current_honest_belief": honest_belief,
            "flan_rating": rating,
            "flan_model": self.flan_scorer.model_name,
        }

    @staticmethod
    def _validate_honest_belief_payload(payload: dict) -> tuple[str, list]:
        """校验自然语言观念响应，并返回标准化字段。"""

        honest_belief = payload.get("current_honest_belief")
        if not isinstance(honest_belief, str) or not honest_belief.strip():
            raise ValueError("current_honest_belief must be a non-empty string")
        evidence_ids = payload.get("evidence_ids", [])
        if not isinstance(evidence_ids, list):
            raise ValueError("evidence_ids must be a list")
        return honest_belief.strip(), evidence_ids

    def _assess_with_llm_voting(
        self,
        agent: "Agent",
        topic: str,
        speech_history: list[dict],
        *,
        trace_tick: int | None = None,
    ) -> dict:
        """同步兼容入口；正式异步运行使用全并行投票路径。"""

        options = self._voting_options(topic)
        voter_count = self._voter_count()
        votes = []
        for voter_index in range(voter_count):
            try:
                votes.append(self._one_vote(
                    agent,
                    topic,
                    speech_history,
                    options,
                    voter_index,
                    trace_tick=trace_tick if trace_tick is not None else agent.world.time,
                ))
            except Exception as exc:
                votes.append({"voter_index": voter_index, "status": "failed", "error": str(exc)})
        return self._summarize_votes(options, voter_count, votes)

    async def _aassess_with_llm_voting(
        self,
        agent: "Agent",
        topic: str,
        speech_history: list[dict],
        *,
        trace_tick: int | None = None,
    ) -> dict:
        """创建该智能体的投票任务，实际启动受独立并发和间隔限制。"""

        options = self._voting_options(topic)
        voter_count = self._voter_count()
        votes = await asyncio.gather(
            *(
                self._aone_vote(
                    agent,
                    topic,
                    speech_history,
                    options,
                    voter_index,
                    trace_tick=trace_tick if trace_tick is not None else agent.world.time,
                )
                for voter_index in range(voter_count)
            )
        )
        return self._summarize_votes(options, voter_count, votes)

    def _one_vote(
        self,
        agent: "Agent",
        topic: str,
        speech_history: list[dict],
        options: list[str],
        voter_index: int,
        *,
        trace_tick: int,
    ) -> dict:
        if self.llm is None:
            return {"voter_index": voter_index, "status": "failed", "error": "LLM client is not configured"}
        system, user = self._voting_prompt(agent, topic, speech_history, options)
        with trace_llm_call(
            self.llm,
            agent_id=agent.id,
            tick=trace_tick,
            stage="opinion_voting",
            metadata={"topic": topic, "voter_index": voter_index},
        ):
            self._wait_for_sync_voting_start_slot()
            raw = self.llm.generate(system, user, response_format=JSON_OBJECT_RESPONSE_FORMAT)
            return self._parse_vote(raw, options, voter_index)

    async def _aone_vote(
        self,
        agent: "Agent",
        topic: str,
        speech_history: list[dict],
        options: list[str],
        voter_index: int,
        *,
        trace_tick: int,
    ) -> dict:
        if self.llm is None:
            return {"voter_index": voter_index, "status": "failed", "error": "LLM client is not configured"}
        system, user = self._voting_prompt(agent, topic, speech_history, options)
        with trace_llm_call(
            self.llm,
            agent_id=agent.id,
            tick=trace_tick,
            stage="opinion_voting",
            metadata={"topic": topic, "voter_index": voter_index},
        ):
            try:
                async with self._current_voting_semaphore():
                    await self._wait_for_voting_start_slot()
                    raw = await self.llm.agenerate(system, user, response_format=JSON_OBJECT_RESPONSE_FORMAT)
                return self._parse_vote(raw, options, voter_index)
            except Exception as exc:
                annotate_current_llm_trace(f"{type(exc).__name__}: {exc}")
                return {"voter_index": voter_index, "status": "failed", "error": str(exc)}

    def _current_voting_semaphore(self) -> asyncio.Semaphore:
        """为当前事件循环返回投票信号量，兼容重复调用同步 step。"""

        loop = asyncio.get_running_loop()
        if self._voting_semaphore is None or self._voting_semaphore_loop is not loop:
            self._voting_semaphore = asyncio.Semaphore(self._voting_concurrency)
            self._voting_semaphore_loop = loop
        return self._voting_semaphore

    def _current_voting_start_lock(self) -> asyncio.Lock:
        """为当前事件循环返回投票启动节流锁。"""

        loop = asyncio.get_running_loop()
        if self._voting_start_lock is None or self._voting_start_lock_loop is not loop:
            self._voting_start_lock = asyncio.Lock()
            self._voting_start_lock_loop = loop
            self._voting_next_start_at = 0.0
        return self._voting_start_lock

    async def _wait_for_voting_start_slot(self) -> None:
        """按统一间隔启动投票请求，避免同一时刻形成请求尖峰。"""

        if self._voting_request_interval <= 0:
            return
        loop = asyncio.get_running_loop()
        async with self._current_voting_start_lock():
            while True:
                delay = self._voting_next_start_at - loop.time()
                if delay <= 0:
                    break
                # Windows 计时器可能提前唤醒，必须按单调时钟复核截止时间。
                await asyncio.sleep(delay)
            self._voting_next_start_at = loop.time() + self._voting_request_interval

    def _wait_for_sync_voting_start_slot(self) -> None:
        """同步路径按相同间隔启动投票请求，避免归档时形成请求尖峰。"""

        if self._voting_request_interval <= 0:
            return
        while True:
            delay = self._voting_next_start_at - time.monotonic()
            if delay <= 0:
                break
            # Windows 计时器可能提前唤醒，必须按单调时钟复核截止时间。
            time.sleep(delay)
        self._voting_next_start_at = time.monotonic() + self._voting_request_interval

    def _skip_empty_voting_window(self, speech_history: list[dict]) -> bool:
        """仅在配置允许且窗口没有线上表达时跳过 LLM 投票。"""

        return bool(getattr(self.config, "opinion_voting_skip_empty_windows", True)) and not speech_history

    def _empty_voting_payload(self, topic: str) -> dict:
        """为空发言窗口生成结构完整、零请求的投票结果。"""

        options = self._voting_options(topic)
        payload = self._summarize_votes(options, 0, [])
        payload["skipped_reason"] = "no_online_speech_in_window"
        return payload

    def _voting_prompt(
        self,
        agent: "Agent",
        topic: str,
        speech_history: list[dict],
        options: list[str],
    ) -> tuple[str, str]:
        """构造只依据线上发帖和评论的单选投票提示词。"""

        option_roles = self._voting_option_roles(topic, options)
        system = (
            "你是观念评测投票者。你只能根据 user JSON 中该用户最近时间窗内的全部线上帖子和评论，"
            "判断该用户对指定议题的立场。不得使用线下发言，不得补充未提供的信息。"
            "你必须且只能从 options 中原样选择一项，并只输出 JSON。"
        )
        user = json.dumps(
            {
                "agent_id": agent.id,
                "topic": topic,
                "options": options,
                "option_roles": option_roles,
                "online_speech_history": speech_history,
                "output_schema": {"choice": "options 中一个完全相同的字符串"},
            },
            ensure_ascii=False,
        )
        return system, user

    def _parse_vote(self, raw: str, options: list[str], voter_index: int) -> dict:
        payload = parse_json_object(raw, context="LLM opinion voting output")
        choice = payload.get("choice")
        if not isinstance(choice, str) or choice not in options:
            raise ValueError(f"vote choice must exactly match one configured option: {choice!r}")
        return {"voter_index": voter_index, "status": "success", "choice": choice}

    def _summarize_votes(self, options: list[str], voter_count: int, votes: list[dict]) -> dict:
        choice_counts = {option: 0 for option in options}
        for vote in votes:
            if vote.get("status") == "success":
                choice_counts[vote["choice"]] += 1
        successful_votes = sum(choice_counts.values())
        return {
            "options": options,
            "requested_voters": voter_count,
            "successful_votes": successful_votes,
            "failed_votes": voter_count - successful_votes,
            "choice_counts": choice_counts,
            "votes": votes,
        }

    def _voting_options(self, topic: str) -> list[str]:
        options = list(get_opinion_topic_definition(topic).voting_options)
        if not options or any(not isinstance(option, str) or not option for option in options):
            raise ValueError(f"topic has no valid voting options: {topic!r}")
        if len(set(options)) != len(options):
            raise ValueError(f"topic voting options must be unique: {topic!r}")
        self._voting_option_roles(topic, options)
        return options

    def _voting_option_roles(self, topic: str, options: list[str]) -> dict[str, str]:
        """校验每个选项都有明确且唯一的极化统计角色。"""

        roles = dict(get_opinion_topic_definition(topic).voting_option_roles)
        if set(roles) != set(options):
            raise ValueError(f"topic voting option roles must exactly match options: {topic!r}")
        invalid_roles = sorted(set(roles.values()) - VOTING_POLARIZATION_ROLES)
        if invalid_roles:
            raise ValueError(f"topic voting option roles are invalid: {invalid_roles!r}")
        if not any(role == VOTING_ROLE_SUPPORT for role in roles.values()):
            raise ValueError(f"topic voting options have no support role: {topic!r}")
        if not any(role == VOTING_ROLE_OPPOSE for role in roles.values()):
            raise ValueError(f"topic voting options have no oppose role: {topic!r}")
        return roles

    def _voting_metrics(self, voting_payload: dict, option_roles: dict[str, str]) -> dict:
        """计算单个智能体的投票份额，不生成连续 opinion。"""

        successful_votes = int(voting_payload["successful_votes"] or 0)
        counts = voting_payload["choice_counts"]
        shares = {
            option: (int(counts.get(option, 0) or 0) / successful_votes if successful_votes else 0.0)
            for option in voting_payload["options"]
        }
        role_shares = {
            role: sum(shares[option] for option, option_role in option_roles.items() if option_role == role)
            for role in VOTING_POLARIZATION_ROLES
        }
        support_share = role_shares[VOTING_ROLE_SUPPORT]
        oppose_share = role_shares[VOTING_ROLE_OPPOSE]
        unknown_share = role_shares[VOTING_ROLE_UNKNOWN]
        known_share = support_share + oppose_share
        decisiveness = abs(support_share - oppose_share) / known_share if known_share else 0.0
        return {
            "choice_shares": shares,
            "support_share": support_share,
            "oppose_share": oppose_share,
            "unknown_share": unknown_share,
            "known_share": known_share,
            "decisiveness": decisiveness,
        }

    def _voter_count(self) -> int:
        return max(1, int(getattr(self.config, "opinion_voter_count", 10)))

    def _recent_online_speech(self, agent: "Agent", tick: int) -> list[dict]:
        """读取最近 n 个时间步内本人发布的全部帖子和评论。"""

        return self._online_speech_window(agent, tick - self._assessment_interval() + 1, tick)

    def _online_speech_window(self, agent: "Agent", window_start: int, window_end: int) -> list[dict]:
        """读取闭区间窗口内本人发布的全部帖子和评论。"""

        platform = getattr(agent, "platform", None)
        if platform is None:
            return []
        speech = []
        for post in getattr(platform, "posts", []) or []:
            post_time = getattr(post, "time", None)
            if getattr(post, "author_id", None) == agent.id and self._time_in_closed_window(
                post_time, window_start, window_end
            ):
                speech.append({
                    "type": "post",
                    "id": getattr(post, "id", None),
                    "time": post_time,
                    "topic": getattr(post, "topic", ""),
                    "content": getattr(post, "content", ""),
                })
            for comment in getattr(post, "comments_list", []) or []:
                comment_time = getattr(comment, "time", None)
                if getattr(comment, "author_id", None) != agent.id:
                    continue
                if not self._time_in_closed_window(comment_time, window_start, window_end):
                    continue
                speech.append({
                    "type": "comment",
                    "id": getattr(comment, "id", None),
                    "time": comment_time,
                    "post_id": getattr(post, "id", None),
                    "post_topic": getattr(post, "topic", ""),
                    "content": getattr(comment, "content", ""),
                })
        return sorted(speech, key=lambda item: (int(item["time"]), str(item["id"])))

    def _time_in_closed_window(self, value, start_tick: int, end_tick: int) -> bool:
        """判断整数时间是否落入指定闭区间。"""

        if isinstance(value, bool) or not isinstance(value, int):
            return False
        return int(start_tick) <= value <= int(end_tick)

    def _time_in_window(self, value, start_tick: int, end_tick: int) -> bool:
        if isinstance(value, bool) or not isinstance(value, int):
            return False
        return start_tick < value <= end_tick

    def _parse_llm_json(self, raw: str) -> dict:
        return parse_json_object(raw, context="LLM opinion assessment output")

    def _llm_system_prompt(self, topic_definition) -> str:
        """统一同步/异步观念评测提示词，强调证据来源不可混用。"""

        role_card_instruction = (
            "当前实验版本启用动态角色卡；它只能约束信息解释、表达风格和合法动作偏好，不能改写证据。"
            if getattr(self.config, "dynamic_role_card_enabled", True)
            and getattr(self.config, "dynamic_role_card_opinion_enabled", True)
            else "当前实验版本关闭观念角色卡；不得假设、补造或引用任何角色卡影响。"
        )
        return (
            "你是生活在沙盒世界中的智能体，正在接受新闻观念调研。"
            "你只能评估 user JSON 中 topic 字段所示系统新闻主题的观念分数。"
            f"本次分数评测对象是“{topic_definition.narrative}”，不是对 system 账号或某一条新闻帖子的相信度。"
            "必须严格区分证据来源："
            "context.self_authored_posts 是你自己发布的帖子，是你的直接表达；"
            "context.self_authored_comments 是你自己写过的评论，是你的直接表达；"
            "context.observed_posts 是你看到的他人、官方新闻或投放者帖子，只能表示你接触到的信息环境，不能写成你的发言、你的立场或你的亲身经历。"
            "若依据 observed_posts 调整分数，reason 和 evidence 必须明确写成“看到/接触到他人帖子/新闻/投放内容”，不能写成“我发布”“我认为过”“我的帖子显示”。"
            f"{role_card_instruction}"
            "previous_topic_score 只能作为上一轮记录参考，不能把同一证据重复累加。"
            f"分数必须在 -1 到 1 之间，{topic_definition.direction_prompt}"
            "只输出 JSON，不要输出额外文字。"
        )

    def _base_topic_score(self, agent: "Agent", topic: str) -> float:
        # `agent.opinion` 是系统新闻主题立场的唯一权威状态，opinion_scores 只是历史/展示镜像。
        return clamp_opinion(agent.opinion)

    def _build_context(self, agent: "Agent", topic: str, tick: int) -> dict:
        """构造固定窗口证据，并附加独立标注的历史记忆背景。"""

        context = self._build_window_context(agent, topic, tick)
        recalled = self._recall_opinion_memories(agent, topic, tick, context)
        context["recalled_memories"] = recalled
        context["evidence_counts"]["recalled_memories"] = len(recalled)
        return context

    def _build_window_context(self, agent: "Agent", topic: str, tick: int) -> dict:
        """只构造本次观念评测固定时间窗口内的新证据。"""

        topic_definition = get_opinion_topic_definition(topic)
        window_end = int(tick)
        window_start = max(1, window_end - self._assessment_interval() + 1)
        personal_evidence = self._window_personal_evidence(agent, window_start, window_end)
        observed_other_speech = self._window_observed_other_speech(agent, window_start, window_end)
        reactions = self._window_reactions(agent, window_start, window_end)
        last_assessment = getattr(agent, "last_opinion_assessment", None)
        previous_belief = ""
        if isinstance(last_assessment, dict):
            previous_belief = str(last_assessment.get("current_honest_belief") or "")
        return {
            "topic": topic,
            "topic_narrative": topic_definition.narrative,
            "score_direction": topic_definition.direction_prompt,
            "window_start_tick": window_start,
            "window_end_tick": window_end,
            "previous_honest_belief": previous_belief,
            "dynamic_role_card": self._dynamic_role_card(agent),
            "self_authored_posts": personal_evidence["self_authored_posts"],
            "self_authored_comments": personal_evidence["self_authored_comments"],
            "observed_other_speech": observed_other_speech,
            "likes_and_dislikes": reactions,
            "evidence_counts": {
                "self_authored_posts": len(personal_evidence["self_authored_posts"]),
                "self_authored_comments": len(personal_evidence["self_authored_comments"]),
                "observed_other_speech": len(observed_other_speech),
                "likes_and_dislikes": len(reactions),
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

    async def _abuild_context(self, agent: "Agent", topic: str, tick: int) -> dict:
        """异步构造固定窗口，并使用异步记忆召回。"""

        context = self._build_window_context(agent, topic, tick)
        recalled = await self._arecall_opinion_memories(agent, topic, tick, context)
        context["recalled_memories"] = recalled
        context["evidence_counts"]["recalled_memories"] = len(recalled)
        return context

    def _recall_opinion_memories(
        self,
        agent: "Agent",
        topic: str,
        tick: int,
        context: dict,
    ) -> list[str]:
        """同步召回观念历史背景，不把它计为当前窗口新证据。"""

        if not getattr(self.config, "memory_forced_recall_enabled", True):
            return []
        recall = getattr(agent, "recall", None)
        if not callable(recall):
            return []
        try:
            return list(recall(self._opinion_memory_observation(topic, tick, context), context="opinion_assessment"))
        except Exception as exc:
            logger.warning("[OpinionAssessment] memory recall failed for %s: %s", agent.id, exc)
            return []

    async def _arecall_opinion_memories(
        self,
        agent: "Agent",
        topic: str,
        tick: int,
        context: dict,
    ) -> list[str]:
        """异步召回观念历史背景，失败时保留当前窗口证据。"""

        if not getattr(self.config, "memory_forced_recall_enabled", True):
            return []
        recall = getattr(agent, "arecall", None)
        if not callable(recall):
            return []
        try:
            return list(
                await recall(
                    self._opinion_memory_observation(topic, tick, context),
                    context="opinion_assessment",
                )
            )
        except Exception as exc:
            logger.warning("[OpinionAssessment] async memory recall failed for %s: %s", agent.id, exc)
            return []

    def _opinion_memory_observation(self, topic: str, tick: int, context: dict) -> dict:
        """生成只含主题、时间和窗口证据标识的结构化召回输入。"""

        return {
            "schema_version": 1,
            "type": "opinion_assessment",
            "time": int(tick),
            "topic": topic,
            "window_start_tick": context.get("window_start_tick"),
            "window_end_tick": context.get("window_end_tick"),
            "self_authored_post_ids": [
                item.get("id")
                for item in context.get("self_authored_posts", [])
                if isinstance(item, dict) and item.get("id") is not None
            ],
            "self_authored_comment_ids": [
                item.get("comment_id")
                for item in context.get("self_authored_comments", [])
                if isinstance(item, dict) and item.get("comment_id") is not None
            ],
            "observed_speech_ids": [
                item.get("id")
                for item in context.get("observed_other_speech", [])
                if isinstance(item, dict) and item.get("id") is not None
            ],
        }

    def _window_personal_evidence(self, agent: "Agent", window_start: int, window_end: int) -> dict:
        """读取窗口内本人发布的全部帖子和评论，不按主题删除。"""

        platform = getattr(agent, "platform", None)
        posts = []
        comments = []
        if platform is not None:
            for post in getattr(platform, "posts", []) or []:
                snapshot = post.to_dict() if hasattr(post, "to_dict") else {}
                if (
                    str(getattr(post, "author_id", "") or "") == agent.id
                    and self._time_in_closed_window(getattr(post, "time", None), window_start, window_end)
                ):
                    posts.append(self._post_evidence_snapshot(snapshot))
                for comment in self._self_comments_from_post(agent, snapshot):
                    if self._time_in_closed_window(comment.get("comment_time"), window_start, window_end):
                        comments.append(comment)
        return {
            "self_authored_posts": sorted(posts, key=lambda item: (int(item.get("time") or 0), str(item.get("id")))),
            "self_authored_comments": sorted(
                comments,
                key=lambda item: (int(item.get("comment_time") or 0), str(item.get("comment_id"))),
            ),
        }

    def _window_observed_other_speech(self, agent: "Agent", window_start: int, window_end: int) -> list[dict]:
        """展开窗口内实际看到的其他人帖子和评论。"""

        speech = []
        for post in getattr(agent, "opinion_seen_posts_buffer", []) or []:
            if not isinstance(post, dict):
                continue
            if (
                str(post.get("author_id") or "") != agent.id
                and self._time_in_closed_window(post.get("time"), window_start, window_end)
            ):
                speech.append({
                    "type": "post",
                    "id": post.get("id"),
                    "author_id": post.get("author_id"),
                    "time": post.get("time"),
                    "topic": post.get("topic"),
                    "content": post.get("content"),
                })
            for comment in post.get("comments") if isinstance(post.get("comments"), list) else []:
                if not isinstance(comment, dict) or str(comment.get("author_id") or "") == agent.id:
                    continue
                if not self._time_in_closed_window(comment.get("time"), window_start, window_end):
                    continue
                speech.append({
                    "type": "comment",
                    "id": comment.get("id"),
                    "author_id": comment.get("author_id"),
                    "time": comment.get("time"),
                    "post_id": post.get("id"),
                    "post_topic": post.get("topic"),
                    "content": comment.get("content"),
                })
        return sorted(speech, key=lambda item: (int(item.get("time") or 0), str(item.get("id"))))

    def _window_reactions(self, agent: "Agent", window_start: int, window_end: int) -> list[dict]:
        """读取窗口内点赞和点踩，且不向评测器暴露目标帖预设分数。"""

        reactions = []
        for item in getattr(agent, "social_reaction_history", []) or []:
            if not isinstance(item, dict) or not self._time_in_closed_window(
                item.get("tick"), window_start, window_end
            ):
                continue
            reactions.append({
                "action": item.get("action"),
                "tick": item.get("tick"),
                "post_id": item.get("post_id"),
                "post_author_id": item.get("post_author_id"),
                "post_topic": item.get("post_topic"),
                "post_content": item.get("post_content"),
            })
        return sorted(reactions, key=lambda item: (int(item.get("tick") or 0), str(item.get("post_id"))))

    def _recent_social_texts(self, agent: "Agent", topic: str) -> list[dict]:
        """观念评测只使用本评测周期内实际看过且 topic 等于当前主题的帖子。"""

        posts = getattr(agent, "opinion_seen_posts_buffer", []) or []
        out = []
        for post in posts:
            # 只按帖子 topic 判断是否属于当前观念主题。
            if isinstance(post, dict) and self._post_matches_topic(post, topic):
                out.append(self._post_evidence_snapshot(post))
        return out[-self.config.opinion_assessment_recent_social:]

    def _structured_social_evidence(self, agent: "Agent", topic: str) -> dict:
        """把本轮社交证据拆成自发表达和他人暴露，避免 LLM 混淆身份。"""

        seen_posts = self._recent_social_texts(agent, topic)
        self_authored_posts = []
        observed_posts = []
        self_authored_comments = []
        for post in seen_posts:
            if str(post.get("author_id") or "") == agent.id:
                self_authored_posts.append(post)
            else:
                observed_posts.append(post)
            self_authored_comments.extend(self._self_comments_from_post(agent, post))
        return {
            "seen_posts": seen_posts,
            "self_authored_posts": self_authored_posts,
            "self_authored_comments": self_authored_comments[-self.config.opinion_assessment_recent_social:],
            "observed_posts": observed_posts,
        }

    def _complete_personal_evidence(self, agent: "Agent", topic: str) -> dict:
        """读取当前议题下本人截至当前时间的全部发帖、评论和反应。"""

        platform = getattr(agent, "platform", None)
        posts = []
        comments = []
        if platform is not None:
            for post in getattr(platform, "posts", []) or []:
                if str(getattr(post, "topic", "") or "") != topic:
                    continue
                snapshot = post.to_dict() if hasattr(post, "to_dict") else {}
                if str(getattr(post, "author_id", "") or "") == agent.id:
                    posts.append(self._post_evidence_snapshot(snapshot))
                comments.extend(self._self_comments_from_post(agent, snapshot))
        reactions = [
            dict(item)
            for item in (getattr(agent, "social_reaction_history", []) or [])
            if isinstance(item, dict) and str(item.get("post_topic") or "") == topic
        ]
        return {
            "self_authored_posts": sorted(
                posts,
                key=lambda item: (int(item.get("time") or 0), str(item.get("id"))),
            ),
            "self_authored_comments": sorted(
                comments,
                key=lambda item: (int(item.get("comment_time") or 0), str(item.get("comment_id"))),
            ),
            "likes_and_dislikes": sorted(
                reactions,
                key=lambda item: (int(item.get("tick") or 0), str(item.get("post_id"))),
            ),
        }

    def _self_comments_from_post(self, agent: "Agent", post: dict) -> list[dict]:
        """抽取智能体自己写过的评论，并保留被评论帖子的来源信息。"""

        comments = post.get("comments") if isinstance(post.get("comments"), list) else []
        out = []
        for comment in comments:
            if not isinstance(comment, dict) or str(comment.get("author_id") or "") != agent.id:
                continue
            out.append(
                {
                    "post_id": post.get("id"),
                    "post_author_id": post.get("author_id"),
                    "post_topic": post.get("topic"),
                    "post_content": post.get("content"),
                    "comment_id": comment.get("id"),
                    "comment_content": comment.get("content"),
                    "comment_time": comment.get("time"),
                    "agreement_to_post": comment.get("agreement_to_post"),
                }
            )
        return out

    def _evidence_contract(self) -> list[str]:
        """写给 LLM 的结构化证据使用规则。"""

        return [
            "self_authored_posts：智能体自己发布的帖子，可作为自身立场直接证据。",
            "self_authored_comments：智能体自己写的评论，可作为自身立场直接证据。",
            "observed_posts：他人、官方新闻或投放者的帖子，只能作为信息暴露和环境输入。",
            "reason/evidence 必须标明证据来源，不能把 observed_posts 写成智能体自己的发言或经历。",
        ]

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
            "is_news": post.get("is_news"),
            "is_rumor": post.get("is_rumor"),
            "source_type": post.get("source_type"),
            "comments": post.get("comments") if isinstance(post.get("comments"), list) else [],
        }

    def _joined_context_text(self, context: dict) -> str:
        chunks = []
        for field in ["self_authored_posts", "self_authored_comments", "observed_other_speech"]:
            values = context.get(field, [])
            if isinstance(values, list):
                chunks.extend(f"{field}: {item}" for item in values)
            elif values:
                chunks.append(f"{field}: {values}")
        dynamic_role_card = context.get("dynamic_role_card", {})
        if dynamic_role_card:
            chunks.append(json.dumps(dynamic_role_card, ensure_ascii=False))
        recalled_memories = context.get("recalled_memories", [])
        if isinstance(recalled_memories, list):
            chunks.extend(f"historical_memory: {item}" for item in recalled_memories)
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
        """窗口没有任何新表达、信息暴露或反应时跳过模型评测。"""

        return any(
            bool(context.get(field))
            for field in [
                "self_authored_posts",
                "self_authored_comments",
                "observed_other_speech",
                "likes_and_dislikes",
            ]
        )

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
        return self._clamp01(0.45 + 0.25 * recent_activity + evidence_bonus)

    def _dynamic_role_card(self, agent: "Agent") -> dict:
        if not getattr(self.config, "dynamic_role_card_enabled", True):
            return {}
        if not getattr(self.config, "dynamic_role_card_opinion_enabled", True):
            return {}
        assessment = getattr(agent, "last_psychological_assessment", None)
        if not isinstance(assessment, dict):
            return {}
        role_card = assessment.get("role_card_delta")
        if not isinstance(role_card, dict):
            return {}
        # 原文溯源只用于记录审计，不重复发送给观念评测 LLM。
        prompt_role_card = {
            key: value
            for key, value in role_card.items()
            if key != "behavior_derivations"
        }
        return {
            "status": assessment.get("status"),
            "activated_needs": assessment.get("activated_needs", []),
            "role_card_delta": prompt_role_card,
        }

    def _clamp01(self, value: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = 0.5
        return round(max(0.0, min(1.0, number)), 4)
