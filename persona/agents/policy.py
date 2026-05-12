from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from persona.logger import get_logger

if TYPE_CHECKING:
    from persona.agents.agent import Agent
    from persona.llm.interface import LLMClient

logger = get_logger(__name__)


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
        return self.parser.parse_action(raw)
