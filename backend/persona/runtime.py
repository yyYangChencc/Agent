from __future__ import annotations
import os
from dataclasses import dataclass, field

from persona.config import AgentConfig
from persona.llm.interface import LLMClient
from persona.agent_memory.mem import MultiAgentMemoryManager
from world.world import World
from persona.agents.agent import Agent
from persona.agents.policy import LLMPolicy, LLMMemoryPlannerPolicy
from persona.agents.prompt import (
    WorldPromptBuilder,
    SocialPromptBuilder,
    ConversationPromptBuilder,
    ReflectPromptBuilder,
    MemoryPlannerPromptBuilder,
)
from persona.agents.parser import ActionParser, MemoryQueryPlanParser
from persona.reflect.reflect import Reflect
from social_sys.platform.platform import SocialPlatform
from persona.opinion import OpinionAssessmentCoordinator
from persona.psychology import PsychologicalAssessmentCoordinator


@dataclass
class SimulationRuntime:
    """仿真的组合根。

    所有共享服务都在这里创建并注入：LLM、记忆、世界、社交平台、各类 policy、
    心理评测器和观念评测器。这样 reset 或创建 agent 时不会散落重复初始化逻辑。
    """
    config: AgentConfig
    llm: LLMClient
    mem: MultiAgentMemoryManager
    world: World
    platform: SocialPlatform
    policy: LLMPolicy
    social_policy: LLMPolicy
    conv_policy: LLMPolicy
    memory_planner: LLMMemoryPlannerPolicy
    reflect: Reflect
    conversation_max_rounds: int = field(default=2)

    @classmethod
    def build(
        cls,
        config: AgentConfig | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        conversation_max_rounds: int = 2,
    ) -> SimulationRuntime:
        """从环境变量和配置构建一套完整运行时。"""

        if config is None:
            config = AgentConfig()
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
        if base_url is None:
            base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        embedding_key = os.environ.get("EMBEDDING_KEY")
        embedding_base_url = os.environ.get("EMBEDDING_BASE_URL", "https://api.openai.com/v1")

        # LLM 客户端同时承担文本生成和 embedding；记忆、policy、评测器共享同一客户端。
        from persona.llm.openai_client import AsyncOpenAIClient
        llm = AsyncOpenAIClient(api_key=api_key, base_url=base_url, embedding_key=embedding_key, embedding_base_url=embedding_base_url, config=config)
        mem = MultiAgentMemoryManager(llm)
        opinion_assessor = OpinionAssessmentCoordinator(config, llm)
        psychological_assessor = PsychologicalAssessmentCoordinator(config, llm)
        platform = SocialPlatform()
        world = World(
            platform=platform,
            psychological_assessor=psychological_assessor,
            opinion_assessor=opinion_assessor,
        )
        policy = LLMPolicy(llm, WorldPromptBuilder(), ActionParser())
        social_policy = LLMPolicy(llm, SocialPromptBuilder(), ActionParser())
        conv_policy = LLMPolicy(llm, ConversationPromptBuilder(), ActionParser())
        memory_planner = LLMMemoryPlannerPolicy(llm, MemoryPlannerPromptBuilder(), MemoryQueryPlanParser())
        reflect = Reflect(llm, ReflectPromptBuilder(), config)

        world.conversation_policy = conv_policy
        world.conversation_max_rounds = conversation_max_rounds

        return cls(
            config=config,
            llm=llm,
            mem=mem,
            world=world,
            platform=platform,
            policy=policy,
            social_policy=social_policy,
            conv_policy=conv_policy,
            memory_planner=memory_planner,
            reflect=reflect,
            conversation_max_rounds=conversation_max_rounds,
        )

    def create_agent(self, agent_id: str, position: list[int],
                     speaking_style: str = "",
                     salary: float = 0.0) -> Agent:
        """创建已接入全部服务的智能体，并注册到社交平台。"""
        agent = Agent(
            agent_id=agent_id,
            position=position,
            world=self.world,
            policy=self.policy,
            mem=self.mem,
            reflect=self.reflect,
            platform=self.platform,
            social_policy=self.social_policy,
            memory_planner=self.memory_planner,
            config=self.config,
            speaking_style=speaking_style,
            salary=salary,
        )
        self.platform.add_agent(agent)
        return agent

    def reset(self) -> None:
        """清空记忆并重建 world/platform，用于从干净状态重新开始实验。"""

        self.mem.reset_all()
        self.platform = SocialPlatform()
        self.world = World(
            platform=self.platform,
            psychological_assessor=PsychologicalAssessmentCoordinator(self.config, self.llm),
            opinion_assessor=OpinionAssessmentCoordinator(self.config, self.llm),
        )
        self.world.conversation_policy = self.conv_policy
        self.world.conversation_max_rounds = self.conversation_max_rounds
