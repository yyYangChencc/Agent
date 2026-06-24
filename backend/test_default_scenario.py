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


from default_scenario import DEFAULT_INITIAL_MONEY, _configure_opinions_and_money
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


class DefaultScenarioTest(unittest.TestCase):
    def _agent(self, agent_id: str) -> Agent:
        return Agent(
            agent_id=agent_id,
            position=[0, 0],
            world=_World(),
            policy=_Stub(),
            mem=_Stub(),
            reflect=_Stub(),
            platform=None,
            social_policy=_Stub(),
            config=AgentConfig(money_threshold=1.0),
        )

    def test_default_initial_money_distribution_updates_pressure_state(self):
        agents = {
            f"agent_{idx}": self._agent(f"agent_{idx}")
            for idx in range(1, 6)
        }

        _configure_opinions_and_money(agents)

        for agent_id, agent in agents.items():
            self.assertEqual(agent.satisfaction["money"], DEFAULT_INITIAL_MONEY[agent_id])
            self.assertEqual(agent.need_gap["money"], 0.0)
            self.assertEqual(agent.effective_pressure["money"], 0.0)

    def test_initial_money_none_keeps_zero_money(self):
        agents = {
            f"agent_{idx}": self._agent(f"agent_{idx}")
            for idx in range(1, 6)
        }

        _configure_opinions_and_money(agents, initial_money=None)

        for agent in agents.values():
            self.assertEqual(agent.satisfaction["money"], 0.0)
            self.assertEqual(agent.need_gap["money"], 1.0)
            self.assertGreater(agent.effective_pressure["money"], 0.0)


if __name__ == "__main__":
    unittest.main()
