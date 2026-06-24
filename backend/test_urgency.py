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
from persona.config import AgentConfig


class _World:
    time = 0

    def __init__(self):
        self.agents = {}

    def add_agent(self, agent):
        self.agents[agent.id] = agent


class _Stub:
    pass


class UrgencyTest(unittest.TestCase):
    def _agent(self, agent_id: str = "agent_1", config: AgentConfig | None = None) -> Agent:
        return Agent(
            agent_id=agent_id,
            position=[0, 0],
            world=_World(),
            policy=_Stub(),
            mem=_Stub(),
            reflect=_Stub(),
            platform=None,
            social_policy=_Stub(),
            config=config or AgentConfig(money_threshold=1.0),
        )

    def test_satisfied_need_uses_nonzero_floor(self):
        agent = self._agent()
        agent.satisfaction.update({
            "satiety": 40.0,
            "relax": 40.0,
            "money": 2.0,
        })
        agent.update_urgency_from_satisfaction()

        self.assertAlmostEqual(agent.urgency["satiety"], 0.08)
        self.assertAlmostEqual(agent.urgency["relax"], 0.08)
        self.assertAlmostEqual(agent.urgency["money"], 0.05)

    def test_lower_satisfaction_produces_higher_urgency(self):
        agent = self._agent()
        agent.satisfaction["satiety"] = 25.0
        agent.update_urgency_from_satisfaction()
        urgency_at_25 = agent.urgency["satiety"]

        agent.satisfaction["satiety"] = 10.0
        agent.update_urgency_from_satisfaction()
        urgency_at_10 = agent.urgency["satiety"]

        self.assertGreater(urgency_at_10, urgency_at_25)

    def test_urgency_rises_when_satisfaction_drops_below_threshold(self):
        agent = self._agent()
        agent.satisfaction["satiety"] = 31.0
        agent.update_urgency_from_satisfaction()
        urgency_above_threshold = agent.urgency["satiety"]

        agent.satisfaction["satiety"] = 29.0
        agent.update_urgency_from_satisfaction()
        urgency_below_threshold = agent.urgency["satiety"]

        self.assertAlmostEqual(urgency_above_threshold, 0.08)
        self.assertGreater(urgency_below_threshold, urgency_above_threshold)

    def test_money_is_capped_when_physiological_layer_is_unmet(self):
        agent = self._agent()
        agent.satisfaction.update({
            "satiety": 29.0,
            "relax": 40.0,
            "money": 0.0,
        })
        agent.update_urgency_from_satisfaction()
        physiological_urgency = max(agent.urgency["satiety"], agent.urgency["relax"])

        self.assertAlmostEqual(physiological_urgency, 0.12)
        self.assertLessEqual(agent.urgency["money"], physiological_urgency)
        self.assertAlmostEqual(agent.urgency["money"], physiological_urgency)

    def test_money_uses_own_gap_when_physiological_layer_is_satisfied(self):
        agent = self._agent()
        agent.satisfaction.update({
            "satiety": 40.0,
            "relax": 40.0,
            "money": 0.0,
        })
        agent.update_urgency_from_satisfaction()

        self.assertAlmostEqual(agent.urgency["money"], 1.0)


if __name__ == "__main__":
    unittest.main()
