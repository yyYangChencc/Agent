from __future__ import annotations

import os
import sys
import types
import unittest


BACKEND_DIR = os.path.dirname(__file__)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


if "chromadb" not in sys.modules:
    chromadb_stub = types.ModuleType("chromadb")

    class _PersistentClient:
        def __init__(self, *args, **kwargs):
            pass

    chromadb_stub.PersistentClient = _PersistentClient
    sys.modules["chromadb"] = chromadb_stub

if "chromadb.config" not in sys.modules:
    chromadb_config_stub = types.ModuleType("chromadb.config")

    class Settings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    chromadb_config_stub.Settings = Settings
    sys.modules["chromadb.config"] = chromadb_config_stub


from persona.agents.agent import Agent
from persona.agents.prompt import ConversationPromptBuilder, SocialPromptBuilder, WorldPromptBuilder
from persona.config import AgentConfig


class _World:
    time = 1
    tools_prompt = "move/eat/sleep/social_step"

    def __init__(self):
        self.agents = {}

    def add_agent(self, agent):
        self.agents[agent.id] = agent


class _Platform:
    tools_prompt = "send_post/comment_post/like_post/dislike_post"


class _Stub:
    def smart_retrieve(self, *args, **kwargs):
        return []


class DynamicRoleCardPromptTest(unittest.TestCase):
    def _agent(self):
        agent = Agent(
            agent_id="agent_1",
            position=[0, 0],
            world=_World(),
            policy=_Stub(),
            mem=_Stub(),
            reflect=_Stub(),
            platform=_Platform(),
            social_policy=_Stub(),
            config=AgentConfig(opinion_assessment_mode="rule"),
            speaking_style="简洁",
        )
        self.assertFalse(hasattr(agent, "role"))
        agent.last_psychological_assessment = {
            "tick": 10,
            "status": "llm_single_evaluator",
            "activated_needs": ["satiety"],
            "role_card_delta": {
                "source_need": "satiety",
                "summary": "饥饿压力上升",
                "emotion_tone": "急迫",
                "cognition": ["优先关注食物"],
                "behavior": ["寻找补给"],
                "social_expression": ["表达更直接"],
                "online_behavior": ["询问资源"],
                "decision_bias": ["提高补给优先级"],
                "constraints": ["不得编造地点"],
                "mediator_focus": [{"key": "resource_seeking", "value": 0.8}],
            },
        }
        return agent

    def _assert_uses_dynamic_role_card(self, system: str, user: str):
        combined = system + "\n" + user
        self.assertIn("必须按照“动态心理角色卡”", system)
        self.assertIn("## 动态心理角色卡", user)
        self.assertIn("饥饿压力上升", user)
        self.assertIn("优先关注食物", user)
        self.assertIn("表达更直接", user)
        self.assertNotIn("role:", combined)
        self.assertNotIn("角色：", combined)

    def test_world_prompt_uses_dynamic_role_card_without_static_role(self):
        agent = self._agent()
        system, user = WorldPromptBuilder().build(agent, "观察", [])
        self._assert_uses_dynamic_role_card(system, user)

    def test_social_prompt_uses_dynamic_role_card_without_static_role(self):
        agent = self._agent()
        system, user = SocialPromptBuilder().build(agent, "当前可互动帖子ID列表：1", [])
        self._assert_uses_dynamic_role_card(system, user)

    def test_conversation_prompt_uses_dynamic_role_card_without_static_role(self):
        agent = self._agent()
        system, user = ConversationPromptBuilder().build(agent, "agent_2 对你说: 你好", [])
        self._assert_uses_dynamic_role_card(system, user)


if __name__ == "__main__":
    unittest.main()
