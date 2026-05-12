from __future__ import annotations
import os
from dataclasses import dataclass, field

from persona.config import AgentConfig
from persona.llm.openai_client import OpenAIClient
from persona.agent_memory.mem import MultiAgentMemoryManager
from world.world import World
from persona.agents.agent import Agent
from persona.agents.policy import LLMPolicy
from persona.agents.prompt import (
    WorldPromptBuilder,
    SocialPromptBuilder,
    ConversationPromptBuilder,
    ReflectPromptBuilder,
)
from persona.agents.parser import ActionParser
from persona.reflect.reflect import Reflect
from social_sys.platform.platform import SocialPlatform
from persona.opinion.updater import OpinionUpdater


@dataclass
class SimulationRuntime:
    """Composition root: holds every shared service; create once per simulation."""
    config: AgentConfig
    llm: OpenAIClient
    mem: MultiAgentMemoryManager
    world: World
    platform: SocialPlatform
    policy: LLMPolicy
    social_policy: LLMPolicy
    conv_policy: LLMPolicy
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
        if config is None:
            config = AgentConfig()
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
        if base_url is None:
            base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

        llm = OpenAIClient(api_key=api_key, base_url=base_url, config=config)
        mem = MultiAgentMemoryManager(llm)
        opinion_updater = OpinionUpdater(config)
        world = World(opinion_updater=opinion_updater)
        platform = SocialPlatform()
        policy = LLMPolicy(llm, WorldPromptBuilder(), ActionParser())
        social_policy = LLMPolicy(llm, SocialPromptBuilder(), ActionParser())
        conv_policy = LLMPolicy(llm, ConversationPromptBuilder(), ActionParser())
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
            reflect=reflect,
            conversation_max_rounds=conversation_max_rounds,
        )

    def create_agent(self, agent_id: str, position: list[int],
                     role: str = "", speaking_style: str = "") -> Agent:
        """Create an agent wired to all runtime services and register it on the platform."""
        agent = Agent(
            agent_id=agent_id,
            position=position,
            world=self.world,
            policy=self.policy,
            mem=self.mem,
            reflect=self.reflect,
            platform=self.platform,
            social_policy=self.social_policy,
            config=self.config,
            role=role,
            speaking_style=speaking_style,
        )
        self.platform.add_agent(agent)
        return agent

    def reset(self) -> None:
        """Wipe memory and rebuild world and platform for a fresh run."""
        self.mem.reset_all()
        self.world = World(opinion_updater=OpinionUpdater(self.config))
        self.world.conversation_policy = self.conv_policy
        self.world.conversation_max_rounds = self.conversation_max_rounds
        self.platform = SocialPlatform()
