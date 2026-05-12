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

# Maps each task name to the demand key it is meant to satisfy
_TASK_DEMAND_MAP: dict[str, str] = {
    "eat something": "satiety",
    "none": "none",
}

# Initial focus string injected into agent when a task is assigned
_TASK_INITIAL_FOCUS: dict[str, str] = {
    "eat something": "搜寻并吃到食物；若视野内与记忆中均无食物相关信息，可与附近智能体沟通询问，或用社交平台发帖求助",
    "none": "",
}

# Extra notes for tasks with effects beyond simple demand reduction
# Add entries here when introducing special task types
_TASK_EXTRA_DESC: dict[str, str] = {
    "none": "任务为 none 时系统会自动缓慢降低 relax，也可主动选择放松、社交或浏览帖子来加速满足",
}


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
            self._decide_next_task(agent)
            return

        # Track progress: if demand for current task is not decreasing, increment stuck counter
        demand_key = _TASK_DEMAND_MAP.get(agent.task)
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

    def _decide_next_task(self, agent: "Agent") -> None:
        task_info: dict[str, str] = {}
        for task, demand_key in _TASK_DEMAND_MAP.items():
            need_val = agent.need.get(demand_key, 0.0) if demand_key and demand_key != "none" else 0.0
            dmnd_val = agent.demand.get(demand_key, 0.0) if demand_key and demand_key != "none" else 0.0
            desc = f"降低 {demand_key} 需求"
            if demand_key and demand_key != "none":
                desc += f"（客观缺乏度 need={need_val:.2f}，主观急迫度 demand={dmnd_val:.2f}）"
            if task in _TASK_EXTRA_DESC:
                desc += f"；{_TASK_EXTRA_DESC[task]}"
            task_info[task] = desc
        system, user = self.prompt.task_decide_prompt(agent, task_info)
        raw = self.llm.generate(system, user)
        logger.debug("[%s] 任务决策原始输出: %s", agent.id, raw)
        match = re.search(r"<Task>(.*?)</Task>", raw, re.DOTALL)
        if not match:
            logger.warning("[%s] 无法解析新任务，LLM输出: %s", agent.id, raw)
            return
        new_task = match.group(1).strip()
        agent.set_task(new_task)
        logger.info("[%s] LLM决定新任务：%s", agent.id, new_task)

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
        demand_key = _TASK_DEMAND_MAP.get(agent.task)
        if demand_key is None:
            logger.warning("[%s] 任务 '%s' 没有对应的需求映射", agent.id, agent.task)
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
