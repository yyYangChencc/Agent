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
from world.objects import company, food_shop
from world.world import World


class _MemoryStub:
    def smart_retrieve(self, *args, **kwargs):
        return []


class _Stub:
    pass


class EconomyBalanceTest(unittest.TestCase):
    def _agent(self) -> tuple[World, Agent]:
        world = World()
        agent = Agent(
            agent_id="agent_1",
            position=[0, 0],
            world=world,
            policy=_Stub(),
            mem=_MemoryStub(),
            reflect=_Stub(),
            platform=None,
            social_policy=_Stub(),
            config=AgentConfig(),
        )
        return world, agent

    def test_company_salary_has_configurable_relax_cost(self):
        world, agent = self._agent()
        workplace = company("company_1", [1, 1], world, salary=8, relax_cost=6)
        agent.satisfaction["relax"] = 20.0

        workplace.interact(agent)

        self.assertEqual(agent.satisfaction["money"], 8.0)
        self.assertEqual(agent.satisfaction["relax"], 14.0)

    def test_food_shop_rejects_purchase_when_money_is_insufficient(self):
        world, agent = self._agent()
        shop = food_shop("shop_1", [1, 1], world, food_num=3, provide=30, price=6)
        agent.satisfaction["money"] = 4.0
        agent.satisfaction["satiety"] = 0.0

        result = shop.interact(agent)

        self.assertIn("余额不足", result)
        self.assertEqual(shop.food_num, 3)
        self.assertEqual(agent.satisfaction["money"], 4.0)
        self.assertEqual(agent.satisfaction["satiety"], 0.0)

    def test_food_shop_purchase_spends_money_and_restores_satiety(self):
        world, agent = self._agent()
        shop = food_shop("shop_1", [1, 1], world, food_num=3, provide=30, price=6)
        agent.satisfaction["money"] = 8.0
        agent.satisfaction["satiety"] = 0.0

        shop.interact(agent)

        self.assertEqual(shop.food_num, 2)
        self.assertEqual(agent.satisfaction["money"], 2.0)
        self.assertEqual(agent.satisfaction["satiety"], 30.0)


if __name__ == "__main__":
    unittest.main()
