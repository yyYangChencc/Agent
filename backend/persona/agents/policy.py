from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from persona.logger import get_logger

if TYPE_CHECKING:
    from persona.agents.agent import Agent
    from persona.llm.interface import LLMClient

logger = get_logger(__name__)

MAX_RETRIES = 2
NO_ACTION_DECISION = '{"think": "本轮没有得到可执行动作，选择不行动", "action": {}}'
NO_MEMORY_PLAN = '{"think": "本轮没有得到有效记忆查询计划，沿用保守召回", "context": "world", "queries": []}'


class Policy(ABC):
    @abstractmethod
    def decide(self, agent: "Agent", observation: str, mem_info) -> str:
        raise NotImplementedError


class LLMPolicy(Policy):
    """普通行动决策 policy，要求 LLM 输出含 think/action 的 JSON 字符串。"""

    def __init__(self, llm_client: "LLMClient", prompt_builder, parser):
        self.llm = llm_client
        self.prompt_builder = prompt_builder
        self.parser = parser

    def decide(self, agent: "Agent", observation: str, mem_info) -> str:
        system, user = self.prompt_builder.build(agent, observation, mem_info)
        raw = self.llm.generate(system, user)
        if not raw.strip():
            logger.warning("[%s] LLM returned an empty response; skip action this tick", agent.id)
            return NO_ACTION_DECISION
        logger.debug("[%s] LLM 输出: %s", agent.id, raw)
        action, error = self.parser.parse_action_with_error(raw)

        for attempt in range(MAX_RETRIES):
            if not error:
                break
            logger.warning("[%s] 解析失败（第%d次），错误：%s，尝试重试", agent.id, attempt + 1, error)
            retry_user = self._retry_prompt(user, raw, error, target="action")
            raw = self.llm.generate(system, retry_user)
            logger.debug("[%s] 重试%d LLM 输出: %s", agent.id, attempt + 1, raw)
            action, error = self.parser.parse_action_with_error(raw)

        if error:
            logger.warning("[%s] 重试%d次后仍解析失败，本轮跳过行动", agent.id, MAX_RETRIES)
            return NO_ACTION_DECISION
        return action

    async def adecide(self, agent: "Agent", observation: str, mem_info) -> str:
        system, user = self.prompt_builder.build(agent, observation, mem_info)
        raw = await self.llm.agenerate(system, user)
        if not raw.strip():
            logger.warning("[%s] LLM returned an empty response; skip action this tick", agent.id)
            return NO_ACTION_DECISION
        logger.debug("[%s] LLM 输出: %s", agent.id, raw)
        action, error = self.parser.parse_action_with_error(raw)

        for attempt in range(MAX_RETRIES):
            if not error:
                break
            logger.warning("[%s] 解析失败（第%d次），错误：%s，尝试重试", agent.id, attempt + 1, error)
            retry_user = self._retry_prompt(user, raw, error, target="action")
            raw = await self.llm.agenerate(system, retry_user)
            logger.debug("[%s] 重试%d LLM 输出: %s", agent.id, attempt + 1, raw)
            action, error = self.parser.parse_action_with_error(raw)

        if error:
            logger.warning("[%s] 重试%d次后仍解析失败，本轮跳过行动", agent.id, MAX_RETRIES)
            return NO_ACTION_DECISION
        return action

    def _retry_prompt(self, user: str, raw: str, error: str, *, target: str) -> str:
        if target == "memory_plan":
            instruction = (
                "请重新输出合法 JSON 对象字符串，顶层必须包含 think、context、queries；"
                "queries 只能使用允许的 type，不能输出 SQL 或行动工具。"
            )
        else:
            instruction = (
                "请检查上述错误，重新输出符合格式要求的内容。"
                "必须只输出一个合法 JSON 对象字符串，顶层包含 think 和 action 字段；"
                "think 必须说明本次选择动作的思考过程和理由；"
                "不要输出 Markdown、额外解释、<Think> 或 <Action> 标签。"
            )
        return (
            f"{user}\n\n"
            f"[上一次输出]\n{raw}\n\n"
            f"[错误信息]\n{error}\n\n"
            f"{instruction}"
        )


class LLMMemoryPlannerPolicy:
    """行动前的 LLM 记忆查询规划器，输出受控 JSON 查询计划。"""

    def __init__(self, llm_client: "LLMClient", prompt_builder, parser):
        self.llm = llm_client
        self.prompt_builder = prompt_builder
        self.parser = parser

    def plan(self, agent: "Agent", observation: str, context: str = "world") -> str:
        system, user = self.prompt_builder.build(agent, observation, context)
        raw = self.llm.generate(system, user)
        if not raw.strip():
            logger.warning("[%s] memory planner returned an empty response", agent.id)
            return self._fallback_plan(context)
        logger.debug("[%s] memory planner 输出: %s", agent.id, raw)
        plan, error = self.parser.parse_plan_with_error(raw)

        for attempt in range(MAX_RETRIES):
            if not error:
                break
            logger.warning("[%s] memory planner 解析失败（第%d次）：%s", agent.id, attempt + 1, error)
            retry_user = self._retry_prompt(user, raw, error)
            raw = self.llm.generate(system, retry_user)
            logger.debug("[%s] memory planner 重试%d 输出: %s", agent.id, attempt + 1, raw)
            plan, error = self.parser.parse_plan_with_error(raw)

        if error:
            logger.warning("[%s] memory planner 重试%d次后仍解析失败，使用保守计划", agent.id, MAX_RETRIES)
            return self._fallback_plan(context)
        return plan

    async def aplan(self, agent: "Agent", observation: str, context: str = "world") -> str:
        system, user = self.prompt_builder.build(agent, observation, context)
        raw = await self.llm.agenerate(system, user)
        if not raw.strip():
            logger.warning("[%s] memory planner returned an empty response", agent.id)
            return self._fallback_plan(context)
        logger.debug("[%s] memory planner 输出: %s", agent.id, raw)
        plan, error = self.parser.parse_plan_with_error(raw)

        for attempt in range(MAX_RETRIES):
            if not error:
                break
            logger.warning("[%s] memory planner 解析失败（第%d次）：%s", agent.id, attempt + 1, error)
            retry_user = self._retry_prompt(user, raw, error)
            raw = await self.llm.agenerate(system, retry_user)
            logger.debug("[%s] memory planner 重试%d 输出: %s", agent.id, attempt + 1, raw)
            plan, error = self.parser.parse_plan_with_error(raw)

        if error:
            logger.warning("[%s] memory planner 重试%d次后仍解析失败，使用保守计划", agent.id, MAX_RETRIES)
            return self._fallback_plan(context)
        return plan

    def _retry_prompt(self, user: str, raw: str, error: str) -> str:
        return (
            f"{user}\n\n"
            f"[上一次输出]\n{raw}\n\n"
            f"[错误信息]\n{error}\n\n"
            "请重新输出合法 JSON 对象字符串，顶层必须包含 think、context、queries；"
            "queries 只能使用允许的 type，不能输出 SQL 或行动工具。"
        )

    def _fallback_plan(self, context: str) -> str:
        return NO_MEMORY_PLAN.replace('"context": "world"', f'"context": "{context}"')
