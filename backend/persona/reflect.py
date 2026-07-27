from __future__ import annotations
import re
import json
from typing import TYPE_CHECKING
from persona.llm.interface import JSON_OBJECT_RESPONSE_FORMAT
from persona.llm.debug_trace import annotate_current_llm_trace, trace_llm_call
from persona.llm.json_utils import parse_json_object
from persona.logger import get_logger

if TYPE_CHECKING:
    from persona.agents.agent import Agent
    from persona.llm.interface import LLMClient
    from persona.agents.prompt import ReflectPromptBuilder
    from persona.config import AgentConfig

logger = get_logger(__name__)

class Reflect:
    def __init__(self, llm_client: "LLMClient", reflect_prompt: "ReflectPromptBuilder", config: "AgentConfig"):
        self.llm = llm_client
        self.prompt = reflect_prompt
        self.config = config

    def summarize_short_term_memory(
        self,
        entries: list[dict],
        *,
        agent_id: str,
        world_time: int,
        start_tick: int,
        end_tick: int,
        max_chars: int,
        chunk_max_chars: int,
    ) -> dict:
        """同步分块压缩短期记忆，并严格校验最终 JSON。"""

        chunks = self._short_term_chunks(entries, chunk_max_chars)
        partials = []
        for chunk in chunks:
            chunk_start, chunk_end = self._short_term_tick_range(chunk, start_tick, end_tick)
            partials.append(self._summarize_short_term_chunk(
                chunk,
                agent_id=agent_id,
                world_time=world_time,
                start_tick=chunk_start,
                end_tick=chunk_end,
                max_chars=max_chars,
            ))
        if len(partials) == 1:
            summary = partials[0]
        else:
            merge_entries = [
                {
                    "record_type": "partial_short_term_summary",
                    "world_time": item["end_tick"],
                    "content": item,
                }
                for item in partials
            ]
            summary = self._summarize_short_term_chunk(
                merge_entries,
                agent_id=agent_id,
                world_time=world_time,
                start_tick=start_tick,
                end_tick=end_tick,
                max_chars=max_chars,
            )
        self._validate_short_term_summary(summary, entries, start_tick, end_tick, max_chars)
        return summary

    async def asummarize_short_term_memory(
        self,
        entries: list[dict],
        *,
        agent_id: str,
        world_time: int,
        start_tick: int,
        end_tick: int,
        max_chars: int,
        chunk_max_chars: int,
    ) -> dict:
        """异步分块压缩短期记忆，并保持同步路径的校验规则。"""

        chunks = self._short_term_chunks(entries, chunk_max_chars)
        partials = []
        for chunk in chunks:
            chunk_start, chunk_end = self._short_term_tick_range(chunk, start_tick, end_tick)
            system, user = self.prompt.short_term_memory_summary(
                chunk,
                start_tick=chunk_start,
                end_tick=chunk_end,
                max_chars=max_chars,
            )
            with trace_llm_call(
                self.llm,
                agent_id=agent_id,
                tick=world_time,
                stage="short_term_memory_summary",
                metadata={"start_tick": chunk_start, "end_tick": chunk_end, "merge": False},
            ):
                raw = await self.llm.agenerate(system, user, response_format=JSON_OBJECT_RESPONSE_FORMAT)
                partial = parse_json_object(raw, context="short-term memory summary")
                self._validate_short_term_summary(partial, chunk, chunk_start, chunk_end, max_chars)
                partials.append(partial)
        if len(partials) == 1:
            summary = partials[0]
        else:
            merge_entries = [
                {
                    "record_type": "partial_short_term_summary",
                    "world_time": item.get("end_tick"),
                    "content": item,
                }
                for item in partials
            ]
            system, user = self.prompt.short_term_memory_summary(
                merge_entries,
                start_tick=start_tick,
                end_tick=end_tick,
                max_chars=max_chars,
            )
            with trace_llm_call(
                self.llm,
                agent_id=agent_id,
                tick=world_time,
                stage="short_term_memory_summary",
                metadata={"start_tick": start_tick, "end_tick": end_tick, "merge": True},
            ):
                raw = await self.llm.agenerate(system, user, response_format=JSON_OBJECT_RESPONSE_FORMAT)
                summary = parse_json_object(raw, context="merged short-term memory summary")
                self._validate_short_term_summary(summary, entries, start_tick, end_tick, max_chars)
        self._validate_short_term_summary(summary, entries, start_tick, end_tick, max_chars)
        return summary

    def _summarize_short_term_chunk(
        self,
        entries: list[dict],
        *,
        agent_id: str,
        world_time: int,
        start_tick: int,
        end_tick: int,
        max_chars: int,
    ) -> dict:
        system, user = self.prompt.short_term_memory_summary(
            entries,
            start_tick=start_tick,
            end_tick=end_tick,
            max_chars=max_chars,
        )
        with trace_llm_call(
            self.llm,
            agent_id=agent_id,
            tick=world_time,
            stage="short_term_memory_summary",
            metadata={"start_tick": start_tick, "end_tick": end_tick},
        ):
            raw = self.llm.generate(system, user, response_format=JSON_OBJECT_RESPONSE_FORMAT)
            summary = parse_json_object(raw, context="short-term memory summary")
            self._validate_short_term_summary(summary, entries, start_tick, end_tick, max_chars)
            return summary

    def _short_term_chunks(self, entries: list[dict], max_chars: int) -> list[list[dict]]:
        """只在完整 tick 边界分块，单个超长 tick 保持完整。"""

        groups: list[list[dict]] = []
        for entry in entries:
            tick = int(entry.get("world_time") or 0)
            if groups and int(groups[-1][-1].get("world_time") or 0) == tick:
                groups[-1].append(entry)
            else:
                groups.append([entry])
        chunks: list[list[dict]] = []
        current: list[dict] = []
        for group in groups:
            combined = current + group
            size = len(json.dumps(combined, ensure_ascii=False, default=str))
            if current and size > max(1, int(max_chars)):
                chunks.append(current)
                current = list(group)
            else:
                current = combined
        if current:
            chunks.append(current)
        return chunks

    def _short_term_tick_range(
        self,
        entries: list[dict],
        fallback_start: int,
        fallback_end: int,
    ) -> tuple[int, int]:
        ticks = [int(item.get("world_time") or 0) for item in entries]
        starts = [
            int((item.get("metadata") or {}).get("start_tick") or item.get("world_time") or 0)
            for item in entries
        ]
        return min(starts or [fallback_start]), max(ticks or [fallback_end])

    def _validate_short_term_summary(
        self,
        summary: dict,
        source_entries: list[dict],
        start_tick: int,
        end_tick: int,
        max_chars: int,
    ) -> None:
        """拒绝范围错误、字段错误和补造实体 ID 的总结。"""

        if int(summary.get("start_tick", -1)) != int(start_tick):
            raise ValueError("short-term summary start_tick mismatch")
        if int(summary.get("end_tick", -1)) != int(end_tick):
            raise ValueError("short-term summary end_tick mismatch")
        for key in ["chronology", "task_progress"]:
            if not isinstance(summary.get(key), str):
                raise ValueError(f"short-term summary {key} must be a string")
        for key in ["successful_actions", "failed_actions", "unresolved_goals", "referenced_entity_ids"]:
            value = summary.get(key)
            if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
                raise ValueError(f"short-term summary {key} must be a string list")
        source_text = json.dumps(source_entries, ensure_ascii=False, sort_keys=True, default=str)
        for entity_id in summary["referenced_entity_ids"]:
            if entity_id not in source_text:
                raise ValueError(f"short-term summary contains unknown entity id: {entity_id}")
        output_text = json.dumps(summary, ensure_ascii=False, sort_keys=True, default=str)
        if len(output_text) > max(1, int(max_chars)):
            raise ValueError("short-term summary exceeds configured character limit")

    def step(self, agent: "Agent") -> None:
        self._update_person_profiles(agent)
        if agent.task == "done":
            return
        if agent.task == "none":
            agent._prev_task_satisfaction = None
            agent.stuck_ticks = 0
            # 仅当有需求未满足时才发起新任务决策
            if self._has_unsatisfied_satisfaction(agent):
                self._decide_next_task(agent)
            return

        # Track progress: if satisfaction for current task is not increasing, increment stuck counter
        urgency_key = agent.task_urgency_key
        if urgency_key:
            current = agent.satisfaction.get(urgency_key, 0.0)
            if agent._prev_task_satisfaction is not None and current > agent._prev_task_satisfaction:
                agent.stuck_ticks = 0
            else:
                agent.stuck_ticks += 1
            agent._prev_task_satisfaction = current
            if agent.stuck_ticks >= self.config.micro_reflect_interval:
                self._micro_reflect(agent)
                agent.stuck_ticks = 0

        ans = self.task_reset(agent)
        if ans == 1:
            completed_task = agent.task
            completed_need_key = agent.task_urgency_key
            agent.set_task("none")
            agent.current_focus = ""
            agent._prev_task_urgency = None
            agent.flush_trajectory(completed_task, completed_need_key)

    def _has_unsatisfied_satisfaction(self, agent: "Agent") -> bool:
        """判断智能体是否有任何 satisfaction 未满足（低于对应阈值）。"""
        for key, satisfaction_val in agent.satisfaction.items():
            threshold = agent.satisfaction_threshold.get(key)
            if threshold is not None and satisfaction_val <= threshold:
                return True
        return False

    def _decide_next_task(self, agent: "Agent") -> None:
        system, user = self.prompt.task_decide_prompt(agent)
        with trace_llm_call(
            self.llm,
            agent_id=agent.id,
            tick=agent.world.time,
            stage="task_selection",
        ):
            raw = self.llm.generate(system, user)
            task_match = re.search(r"<Task>(.*?)</Task>", raw, re.DOTALL)
            key_match = re.search(r"<UrgencyKey>(.*?)</UrgencyKey>", raw, re.DOTALL)
            if not task_match or not key_match:
                annotate_current_llm_trace("缺少 Task 或 UrgencyKey 标签")
            elif key_match.group(1).strip() not in set(agent.satisfaction.keys()):
                annotate_current_llm_trace(f"无效 UrgencyKey: {key_match.group(1).strip()}")
        logger.debug("[%s] 任务决策原始输出: %s", agent.id, raw)
        if not task_match or not key_match:
            logger.warning("[%s] 无法解析新任务或需求键，LLM输出: %s", agent.id, raw)
            return
        new_task = task_match.group(1).strip()
        urgency_key = key_match.group(1).strip()
        valid_urgency_keys = set(agent.satisfaction.keys())
        if urgency_key not in valid_urgency_keys:
            logger.warning("[%s] LLM返回无效需求键: %s，合法值为 %s", agent.id, urgency_key, valid_urgency_keys)
            return
        agent.set_task(new_task, urgency_key)
        logger.info("[%s] LLM决定新任务：%s（需求键: %s）", agent.id, new_task, urgency_key)

    async def _adecide_next_task(self, agent: "Agent") -> None:
        system, user = self.prompt.task_decide_prompt(agent)
        with trace_llm_call(
            self.llm,
            agent_id=agent.id,
            tick=agent.world.time,
            stage="task_selection",
        ):
            raw = await self.llm.agenerate(system, user)
            task_match = re.search(r"<Task>(.*?)</Task>", raw, re.DOTALL)
            key_match = re.search(r"<UrgencyKey>(.*?)</UrgencyKey>", raw, re.DOTALL)
            if not task_match or not key_match:
                annotate_current_llm_trace("缺少 Task 或 UrgencyKey 标签")
            elif key_match.group(1).strip() not in set(agent.satisfaction.keys()):
                annotate_current_llm_trace(f"无效 UrgencyKey: {key_match.group(1).strip()}")
        logger.debug("[%s] 任务决策原始输出: %s", agent.id, raw)
        if not task_match or not key_match:
            logger.warning("[%s] 无法解析新任务或需求键，LLM输出: %s", agent.id, raw)
            return
        new_task = task_match.group(1).strip()
        urgency_key = key_match.group(1).strip()
        valid_urgency_keys = set(agent.satisfaction.keys())
        if urgency_key not in valid_urgency_keys:
            logger.warning("[%s] LLM返回无效需求键: %s，合法值为 %s", agent.id, urgency_key, valid_urgency_keys)
            return
        agent.set_task(new_task, urgency_key)
        logger.info("[%s] LLM决定新任务：%s（需求键: %s）", agent.id, new_task, urgency_key)

    def _micro_reflect(self, agent: "Agent") -> None:
        system, user = self.prompt.micro_reflect_prompt(agent)
        with trace_llm_call(
            self.llm,
            agent_id=agent.id,
            tick=agent.world.time,
            stage="micro_reflection",
        ):
            raw = self.llm.generate(system, user)
            insight_match = re.search(r"<Insight>(.*?)</Insight>", raw, re.DOTALL)
            focus_match = re.search(r"<Focus>(.*?)</Focus>", raw, re.DOTALL)
            if not insight_match or not focus_match:
                annotate_current_llm_trace("缺少 Insight 或 Focus 标签")
        logger.debug("[%s] 微反思原始输出: %s", agent.id, raw)

        if insight_match:
            insight = insight_match.group(1).strip()
            agent.remember(
                insight,
                memory_type="reflective",
                task=agent.task,
                need_key=agent.task_urgency_key,
                importance=0.65,
                confidence=0.6,
            )
            logger.info("[%s] 微反思洞察: %s", agent.id, insight)

        if focus_match:
            agent.current_focus = focus_match.group(1).strip()
            logger.info("[%s] 微反思更新焦点: %s", agent.id, agent.current_focus)
        else:
            logger.warning("[%s] 微反思未能解析 Focus，保持原焦点", agent.id)
        if hasattr(agent, "store_trajectory_checkpoint"):
            # 卡住时保留未完成经历，避免只有成功任务才能进入长期记忆。
            agent.store_trajectory_checkpoint(outcome="stuck")

    async def _amicro_reflect(self, agent: "Agent") -> None:
        system, user = self.prompt.micro_reflect_prompt(agent)
        with trace_llm_call(
            self.llm,
            agent_id=agent.id,
            tick=agent.world.time,
            stage="micro_reflection",
        ):
            raw = await self.llm.agenerate(system, user)
            insight_match = re.search(r"<Insight>(.*?)</Insight>", raw, re.DOTALL)
            focus_match = re.search(r"<Focus>(.*?)</Focus>", raw, re.DOTALL)
            if not insight_match or not focus_match:
                annotate_current_llm_trace("缺少 Insight 或 Focus 标签")
        logger.debug("[%s] 微反思原始输出: %s", agent.id, raw)

        if insight_match:
            insight = insight_match.group(1).strip()
            await agent.aremember(
                insight,
                memory_type="reflective",
                task=agent.task,
                need_key=agent.task_urgency_key,
                importance=0.65,
                confidence=0.6,
            )
            logger.info("[%s] 微反思洞察: %s", agent.id, insight)

        if focus_match:
            agent.current_focus = focus_match.group(1).strip()
            logger.info("[%s] 微反思更新焦点: %s", agent.id, agent.current_focus)
        else:
            logger.warning("[%s] 微反思未能解析 Focus，保持原焦点", agent.id)
        if hasattr(agent, "store_trajectory_checkpoint"):
            agent.store_trajectory_checkpoint(outcome="stuck")

    async def astep(self, agent: "Agent") -> None:
        await self._aupdate_person_profiles(agent)
        if agent.task == "done":
            return
        if agent.task == "none":
            agent._prev_task_satisfaction = None
            agent.stuck_ticks = 0
            if self._has_unsatisfied_satisfaction(agent):
                await self._adecide_next_task(agent)
            return

        urgency_key = agent.task_urgency_key
        if urgency_key:
            current = agent.satisfaction.get(urgency_key, 0.0)
            if agent._prev_task_satisfaction is not None and current > agent._prev_task_satisfaction:
                agent.stuck_ticks = 0
            else:
                agent.stuck_ticks += 1
            agent._prev_task_satisfaction = current
            if agent.stuck_ticks >= self.config.micro_reflect_interval:
                await self._amicro_reflect(agent)
                agent.stuck_ticks = 0

        ans = self.task_reset(agent)
        if ans == 1:
            completed_task = agent.task
            completed_need_key = agent.task_urgency_key
            agent.set_task("none")
            agent.current_focus = ""
            agent._prev_task_urgency = None
            await agent.aflush_trajectory(completed_task, completed_need_key)

    def task_reset(self, agent: "Agent") -> int:
        urgency_key = agent.task_urgency_key
        if not urgency_key:
            logger.warning("[%s] 任务 '%s' 没有对应的需求键", agent.id, agent.task)
            return -1
        threshold = agent.satisfaction_threshold.get(urgency_key)
        if threshold is None:
            logger.warning("[%s] 需求 '%s' 没有对应的阈值配置", agent.id, urgency_key)
            return -1
        current_satisfaction = agent.satisfaction.get(urgency_key, 0.0)
        if current_satisfaction > threshold:
            logger.info("[%s] 任务完成：%s satisfaction=%.2f > 阈值 %.2f",
                        agent.id, urgency_key, current_satisfaction, threshold)
            return 1
        logger.debug("[%s] 任务未完成：%s satisfaction=%.2f <= 阈值 %.2f",
                     agent.id, urgency_key, current_satisfaction, threshold)
        return 0

    def _update_person_profiles(self, agent: "Agent") -> None:
        """在 reflect 阶段根据新增事实更新人物档案。"""

        mem = getattr(agent, "mem", None)
        world = getattr(agent, "world", None)
        if mem is None or world is None or not hasattr(mem, "update_person_profiles_from_reflection"):
            return
        try:
            # 同步入口也遵守人物档案的 LLM 开关。
            profile_llm = self.llm if self.config.memory_person_profile_llm_enabled else None
            mem.update_person_profiles_from_reflection(agent.id, current_time=world.time, llm=profile_llm)
        except Exception as exc:
            logger.debug("[%s] 更新人物档案失败: %s", agent.id, exc)

    async def _aupdate_person_profiles(self, agent: "Agent") -> None:
        """异步 reflect 阶段的人物档案更新。"""

        mem = getattr(agent, "mem", None)
        world = getattr(agent, "world", None)
        if mem is None or world is None:
            return
        try:
            # 默认使用规则摘要，只有显式开启时才调用 LLM。
            profile_llm = self.llm if self.config.memory_person_profile_llm_enabled else None
            if hasattr(mem, "aupdate_person_profiles_from_reflection"):
                await mem.aupdate_person_profiles_from_reflection(agent.id, current_time=world.time, llm=profile_llm)
            elif hasattr(mem, "update_person_profiles_from_reflection"):
                mem.update_person_profiles_from_reflection(agent.id, current_time=world.time, llm=profile_llm)
        except Exception as exc:
            logger.debug("[%s] 异步更新人物档案失败: %s", agent.id, exc)
