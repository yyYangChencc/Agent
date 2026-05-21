from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from persona.logger import get_logger

if TYPE_CHECKING:
    from persona.agents.agent import Agent
    from persona.llm.interface import LLMClient

logger = get_logger(__name__)

MAX_RETRIES = 2


class Policy(ABC):
    @abstractmethod
    def decide(self, agent: "Agent", observation: str, mem_info) -> str:
        raise NotImplementedError


class LLMPolicy(Policy):
    def __init__(self, llm_client: "LLMClient", prompt_builder, parser):
        self.llm = llm_client
        self.prompt_builder = prompt_builder
        self.parser = parser

    def decide(self, agent: "Agent", observation: str, mem_info) -> str:
        system, user = self.prompt_builder.build(agent, observation, mem_info)
        raw = self.llm.generate(system, user)
        logger.debug("[%s] LLM 输出: %s", agent.id, raw)
        action, error = self.parser.parse_action_with_error(raw)

        for attempt in range(MAX_RETRIES):
            if not error:
                break
            logger.warning("[%s] 解析失败（第%d次），错误：%s，尝试重试", agent.id, attempt + 1, error)
            retry_user = (
                f"{user}\n\n"
                f"[上一次输出]\n{raw}\n\n"
                f"[错误信息]\n{error}\n\n"
                "请检查上述错误，重新输出符合格式要求的内容。"
                "必须包含 <Action>{...}</Action> 标签，内容为合法 JSON。"
            )
            raw = self.llm.generate(system, retry_user)
            logger.debug("[%s] 重试%d LLM 输出: %s", agent.id, attempt + 1, raw)
            action, error = self.parser.parse_action_with_error(raw)

        if error:
            logger.warning("[%s] 重试%d次后仍解析失败，本轮跳过行动", agent.id, MAX_RETRIES)
        return action
