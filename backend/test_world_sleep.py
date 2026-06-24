from __future__ import annotations

import os
import sys
import types
import unittest
from unittest.mock import patch


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
from world.objects import bed
from world.world import World


class _MemoryStub:
    def smart_retrieve(self, *args, **kwargs):
        return []


class _PolicyStub:
    async def adecide(self, *args, **kwargs):
        return ""


class _ReflectStub:
    def __init__(self):
        self.calls = 0

    async def astep(self, agent):
        self.calls += 1


class WorldSleepTest(unittest.TestCase):
    def test_waking_agent_skips_normal_reflect_and_decay_in_same_tick(self):
        config = AgentConfig(
            sleep_time=1,
            sleep_relax_recover=10.0,
            satiety_decay_rate=0.0,
            relax_decay_rate=2.0,
            relax_increase_rate=0.0,
        )
        world = World()
        reflect = _ReflectStub()
        agent = Agent(
            agent_id="agent_1",
            position=[2, 2],
            world=world,
            policy=_PolicyStub(),
            mem=_MemoryStub(),
            reflect=reflect,
            platform=None,
            social_policy=_PolicyStub(),
            config=config,
        )
        bed("bed_1", [2, 2], world)
        agent.sleep_status("bed_1")

        world.step()

        self.assertFalse(agent.sleeping)
        self.assertEqual(agent.sleep_ticks_remaining, 0)
        self.assertEqual(reflect.calls, 0)
        self.assertAlmostEqual(agent.satisfaction["relax"], 10.0)

    def test_wakeup_does_not_overwrite_bed_when_no_adjacent_cell_is_empty(self):
        config = AgentConfig()
        world = World()
        agent = Agent(
            agent_id="agent_1",
            position=[2, 2],
            world=world,
            policy=_PolicyStub(),
            mem=_MemoryStub(),
            reflect=_ReflectStub(),
            platform=None,
            social_policy=_PolicyStub(),
            config=config,
        )
        bed_obj = bed("bed_1", [2, 2], world)
        bed_obj.occupant_id = agent.id
        bed_obj.free_num = 0
        agent.sleeping = True
        agent.sleeping_on_bed_id = bed_obj.id
        agent.position = [2, 2]
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            world.map.place(2 + dx, 2 + dy, f"block_{dx}_{dy}")

        with patch("persona.agents.agent.logger.warning"):
            agent.wakeup(bed_obj)

        self.assertFalse(agent.sleeping)
        self.assertEqual(world.map.get_e(2, 2), "bed_1")


if __name__ == "__main__":
    unittest.main()
