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
from tools.operator_tools import Operator
from world.objects import building
from world.world import World


class _MemoryStub:
    def smart_retrieve(self, *args, **kwargs):
        return []


class _PolicyStub:
    pass


class _ReflectStub:
    pass


class BuildingOccupancyTest(unittest.TestCase):
    def _world_agent_operator(self):
        world = World()
        agent = Agent(
            agent_id="agent_1",
            position=[2, 3],
            world=world,
            policy=_PolicyStub(),
            mem=_MemoryStub(),
            reflect=_ReflectStub(),
            platform=None,
            social_policy=_PolicyStub(),
            config=AgentConfig(),
        )
        target = building("building_1", [2, 2], world)
        return world, agent, target, Operator(world)

    def test_enter_building_releases_agent_map_cell_and_keeps_building_cell(self):
        world, agent, target, operator = self._world_agent_operator()

        result = operator.enter_building("agent_1", "building_1")

        self.assertIn("进入了 building_1", result)
        self.assertEqual(agent.inside_building_id, "building_1")
        self.assertEqual(agent.position, target.position)
        self.assertEqual(world.map.get_e(2, 3), "0")
        self.assertEqual(world.map.get_e(2, 2), "building_1")
        self.assertEqual(target.occupants, ["agent_1"])

    def test_exit_building_places_agent_next_to_building(self):
        world, agent, target, operator = self._world_agent_operator()
        operator.enter_building("agent_1", "building_1")

        result = operator.exit_building("agent_1")

        self.assertIn("离开了 building_1", result)
        self.assertIsNone(agent.inside_building_id)
        self.assertNotEqual(agent.position, target.position)
        self.assertEqual(world.map.get_e(2, 2), "building_1")
        self.assertEqual(world.map.get_e(agent.position[0], agent.position[1]), "agent_1")
        self.assertEqual(target.occupants, [])

    def test_enter_building_is_rejected_while_agent_is_sleeping(self):
        _world, agent, _target, operator = self._world_agent_operator()
        agent.sleeping = True

        result = operator.enter_building("agent_1", "building_1")

        self.assertEqual(result, "睡眠中无法进入建筑")

    def test_sleep_is_rejected_while_agent_is_inside_building(self):
        _world, agent, _target, operator = self._world_agent_operator()
        operator.enter_building("agent_1", "building_1")

        result = operator.sleep("agent_1", "building_1")

        self.assertEqual(result, "当前在建筑内，需先离开建筑再睡觉")
        self.assertEqual(agent.inside_building_id, "building_1")


if __name__ == "__main__":
    unittest.main()
