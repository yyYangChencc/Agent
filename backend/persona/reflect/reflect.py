from __future__ import annotations
import re
from typing import TYPE_CHECKING
from persona.logger import get_logger

if TYPE_CHECKING:
    from persona.agents.agent import Agent
    from persona.llm.interface import LLMClient
    from persona.agents.prompt import ReflectPromptBuilder
    from persona.config import AgentConfig

logger = get_logger(__name__)

# Valid urgency keys that agents can target
VALID_URGENCY_KEYS = {"satiety", "relax", "money"}


class Reflect:
    def __init__(self, llm_client: "LLMClient", reflect_prompt: "ReflectPromptBuilder", config: "AgentConfig"):
        self.llm = llm_client
        self.prompt = reflect_prompt
        self.config = config

    def step(self, agent: "Agent") -> None:
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
            agent.set_task("none")
            agent.current_focus = ""
            agent._prev_task_urgency = None
            agent.flush_trajectory(completed_task)

    def _has_unsatisfied_satisfaction(self, agent: "Agent") -> bool:
        """判断智能体是否有任何 satisfaction 未满足（低于对应阈值）。"""
        for key, satisfaction_val in agent.satisfaction.items():
            threshold = agent.satisfaction_threshold.get(key)
            if threshold is not None and satisfaction_val <= threshold:
                return True
        return False

    def _decide_next_task(self, agent: "Agent") -> None:
        system, user = self.prompt.task_decide_prompt(agent)
        raw = self.llm.generate(system, user)
        logger.debug("[%s] 任务决策原始输出: %s", agent.id, raw)
        task_match = re.search(r"<Task>(.*?)</Task>", raw, re.DOTALL)
        key_match = re.search(r"<UrgencyKey>(.*?)</UrgencyKey>", raw, re.DOTALL)
        if not task_match or not key_match:
            logger.warning("[%s] 无法解析新任务或需求键，LLM输出: %s", agent.id, raw)
            return
        new_task = task_match.group(1).strip()
        urgency_key = key_match.group(1).strip()
        if urgency_key not in VALID_URGENCY_KEYS:
            logger.warning("[%s] LLM返回无效需求键: %s，合法值为 %s", agent.id, urgency_key, VALID_URGENCY_KEYS)
            return
        agent.set_task(new_task, urgency_key)
        logger.info("[%s] LLM决定新任务：%s（需求键: %s）", agent.id, new_task, urgency_key)

    async def _adecide_next_task(self, agent: "Agent") -> None:
        system, user = self.prompt.task_decide_prompt(agent)
        raw = await self.llm.agenerate(system, user)
        logger.debug("[%s] 任务决策原始输出: %s", agent.id, raw)
        task_match = re.search(r"<Task>(.*?)</Task>", raw, re.DOTALL)
        key_match = re.search(r"<UrgencyKey>(.*?)</UrgencyKey>", raw, re.DOTALL)
        if not task_match or not key_match:
            logger.warning("[%s] 无法解析新任务或需求键，LLM输出: %s", agent.id, raw)
            return
        new_task = task_match.group(1).strip()
        urgency_key = key_match.group(1).strip()
        if urgency_key not in VALID_URGENCY_KEYS:
            logger.warning("[%s] LLM返回无效需求键: %s，合法值为 %s", agent.id, urgency_key, VALID_URGENCY_KEYS)
            return
        agent.set_task(new_task, urgency_key)
        logger.info("[%s] LLM决定新任务：%s（需求键: %s）", agent.id, new_task, urgency_key)

    def _micro_reflect(self, agent: "Agent") -> None:
        system, user = self.prompt.micro_reflect_prompt(agent)
        raw = self.llm.generate(system, user)
        logger.debug("[%s] 微反思原始输出: %s", agent.id, raw)

        insight_match = re.search(r"<Insight>(.*?)</Insight>", raw, re.DOTALL)
        focus_match = re.search(r"<Focus>(.*?)</Focus>", raw, re.DOTALL)

        if insight_match:
            insight = insight_match.group(1).strip()
            agent.remember(insight, type="micro_reflection")
            logger.info("[%s] 微反思洞察: %s", agent.id, insight)

        if focus_match:
            agent.current_focus = focus_match.group(1).strip()
            logger.info("[%s] 微反思更新焦点: %s", agent.id, agent.current_focus)
        else:
            logger.warning("[%s] 微反思未能解析 Focus，保持原焦点", agent.id)

    async def _amicro_reflect(self, agent: "Agent") -> None:
        system, user = self.prompt.micro_reflect_prompt(agent)
        raw = await self.llm.agenerate(system, user)
        logger.debug("[%s] 微反思原始输出: %s", agent.id, raw)

        insight_match = re.search(r"<Insight>(.*?)</Insight>", raw, re.DOTALL)
        focus_match = re.search(r"<Focus>(.*?)</Focus>", raw, re.DOTALL)

        if insight_match:
            insight = insight_match.group(1).strip()
            await agent.aremember(insight, type="micro_reflection")
            logger.info("[%s] 微反思洞察: %s", agent.id, insight)

        if focus_match:
            agent.current_focus = focus_match.group(1).strip()
            logger.info("[%s] 微反思更新焦点: %s", agent.id, agent.current_focus)
        else:
            logger.warning("[%s] 微反思未能解析 Focus，保持原焦点", agent.id)

    async def astep(self, agent: "Agent") -> None:
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
            agent.set_task("none")
            agent.current_focus = ""
            agent._prev_task_urgency = None
            await agent.aflush_trajectory(completed_task)

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
