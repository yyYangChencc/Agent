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


class NeedPressureTest(unittest.TestCase):
    def _agent(self) -> Agent:
        config = AgentConfig(
            satiety_threshold=30.0,
            relax_threshold=30.0,
            money_threshold=1.0,
            satiety_decay_rate=0.0,
            relax_decay_rate=0.0,
            relax_increase_rate=0.0,
            need_pressure_dt=1.0,
            pressure_recovery_rates={"satiety": 0.25, "relax": 0.25, "money": 0.25},
            pressure_load_kappas={"satiety": 2.0, "relax": 2.0, "money": 2.0},
            pressure_current_weights={"satiety": 0.6, "relax": 0.6, "money": 0.6},
            pressure_residual_weights={"satiety": 0.2, "relax": 0.2, "money": 0.2},
            pressure_amplification_weights={"satiety": 0.2, "relax": 0.2, "money": 0.2},
        )
        return Agent(
            agent_id="agent_1",
            position=[0, 0],
            world=_World(),
            policy=_Stub(),
            mem=_Stub(),
            reflect=_Stub(),
            platform=None,
            social_policy=_Stub(),
            config=config,
        )

    def test_gap_is_normalized_by_threshold(self):
        agent = self._agent()
        agent.satisfaction["satiety"] = 15.0
        agent.update_need_pressure(accumulate=False)

        self.assertAlmostEqual(agent.need_gap["satiety"], 0.5)
        self.assertAlmostEqual(agent.pressure_memory["satiety"], 0.0)
        self.assertAlmostEqual(agent.effective_pressure["satiety"], 0.3)

    def test_pressure_memory_accumulates_then_decays_after_recovery(self):
        agent = self._agent()
        agent.satisfaction["satiety"] = 0.0

        agent.update_need_pressure()
        agent.update_need_pressure()

        self.assertAlmostEqual(agent.need_gap["satiety"], 1.0)
        self.assertAlmostEqual(agent.pressure_memory["satiety"], 2.0)
        self.assertGreater(agent.load_saturation["satiety"], 0.0)
        pressure_under_gap = agent.effective_pressure["satiety"]

        agent.satisfaction["satiety"] = 40.0
        agent.update_need_pressure()

        self.assertAlmostEqual(agent.need_gap["satiety"], 0.0)
        self.assertAlmostEqual(agent.pressure_memory["satiety"], 1.5)
        self.assertGreater(agent.effective_pressure["satiety"], 0.0)
        self.assertLess(agent.effective_pressure["satiety"], pressure_under_gap)

    def test_pressure_does_not_change_during_sleep(self):
        agent = self._agent()
        agent.satisfaction["satiety"] = 0.0
        agent.update_need_pressure()
        memory_before = dict(agent.pressure_memory)
        pressure_before = dict(agent.effective_pressure)

        agent.sleeping = True
        agent.tick_satisfaction()

        self.assertEqual(agent.pressure_memory, memory_before)
        self.assertEqual(agent.effective_pressure, pressure_before)

    def test_sleep_recovery_restores_relax_per_tick_without_accumulating_pressure(self):
        agent = self._agent()
        agent.config.sleep_time = 4
        agent.config.sleep_relax_recover = 40.0
        agent.satisfaction["relax"] = 0.0
        agent.update_need_pressure()
        memory_before = agent.pressure_memory["relax"]

        agent.sleeping = True
        agent.tick_sleep_recovery()

        self.assertAlmostEqual(agent.satisfaction["relax"], 10.0)
        self.assertAlmostEqual(agent.pressure_memory["relax"], memory_before)
        self.assertAlmostEqual(agent.need_gap["relax"], (30.0 - 10.0) / 30.0)

    def test_sleep_status_uses_at_least_one_tick(self):
        agent = self._agent()
        agent.config.sleep_time = 0

        agent.sleep_status("bed_1")

        self.assertTrue(agent.sleeping)
        self.assertEqual(agent.sleep_ticks_remaining, 1)


if __name__ == "__main__":
    unittest.main()
