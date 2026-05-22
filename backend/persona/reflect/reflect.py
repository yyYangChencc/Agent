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

# Valid demand keys that agents can target
VALID_DEMAND_KEYS = {"satiety", "relax", "money"}


class Reflect:
    def __init__(self, llm_client: "LLMClient", reflect_prompt: "ReflectPromptBuilder", config: "AgentConfig"):
        self.llm = llm_client
        self.prompt = reflect_prompt
        self.config = config

    def step(self, agent: "Agent") -> None:
        if agent.task == "done":
            return
        if agent.task == "none":
            agent._prev_task_need = None
            agent.stuck_ticks = 0
            # 仅当有需求未满足时才发起新任务决策
            if self._has_unsatisfied_need(agent):
                self._decide_next_task(agent)
            return

        # Track progress: if need for current task is not increasing, increment stuck counter
        demand_key = agent.task_demand_key
        if demand_key:
            current = agent.need.get(demand_key, 0.0)
            if agent._prev_task_need is not None and current > agent._prev_task_need:
                agent.stuck_ticks = 0
            else:
                agent.stuck_ticks += 1
            agent._prev_task_need = current
            if agent.stuck_ticks >= self.config.micro_reflect_interval:
                self._micro_reflect(agent)
                agent.stuck_ticks = 0

        ans = self.task_reset(agent)
        if ans == 1:
            completed_task = agent.task
            agent.set_task("none")
            agent.current_focus = ""
            agent._prev_task_demand = None
            agent.flush_trajectory(completed_task)

    def _has_unsatisfied_need(self, agent: "Agent") -> bool:
        """判断智能体是否有任何 need 未满足（低于对应阈值）。"""
        for key, need_val in agent.need.items():
            threshold = agent.demand_threshold.get(key)
            if threshold is not None and need_val <= threshold:
                return True
        return False

    def _decide_next_task(self, agent: "Agent") -> None:
        system, user = self.prompt.task_decide_prompt(agent)
        raw = self.llm.generate(system, user)
        logger.debug("[%s] 任务决策原始输出: %s", agent.id, raw)
        task_match = re.search(r"<Task>(.*?)</Task>", raw, re.DOTALL)
        key_match = re.search(r"<DemandKey>(.*?)</DemandKey>", raw, re.DOTALL)
        if not task_match or not key_match:
            logger.warning("[%s] 无法解析新任务或需求键，LLM输出: %s", agent.id, raw)
            return
        new_task = task_match.group(1).strip()
        demand_key = key_match.group(1).strip()
        if demand_key not in VALID_DEMAND_KEYS:
            logger.warning("[%s] LLM返回无效需求键: %s，合法值为 %s", agent.id, demand_key, VALID_DEMAND_KEYS)
            return
        agent.set_task(new_task, demand_key)
        logger.info("[%s] LLM决定新任务：%s（需求键: %s）", agent.id, new_task, demand_key)

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

    def task_reset(self, agent: "Agent") -> int:
        demand_key = agent.task_demand_key
        if not demand_key:
            logger.warning("[%s] 任务 '%s' 没有对应的需求键", agent.id, agent.task)
            return -1
        threshold = agent.demand_threshold.get(demand_key)
        if threshold is None:
            logger.warning("[%s] 需求 '%s' 没有对应的阈值配置", agent.id, demand_key)
            return -1
        current_need = agent.need.get(demand_key, 0.0)
        if current_need > threshold:
            logger.info("[%s] 任务完成：%s need=%.2f > 阈值 %.2f",
                        agent.id, demand_key, current_need, threshold)
            return 1
        logger.debug("[%s] 任务未完成：%s need=%.2f <= 阈值 %.2f",
                     agent.id, demand_key, current_need, threshold)
        return 0
