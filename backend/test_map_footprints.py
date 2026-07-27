from __future__ import annotations

import unittest
from types import SimpleNamespace

from persona.llm.mock_client import MockLLMClient
from scenarios.iac_gay_marriage_10 import build_runtime
from tools.operator_tools import Operator
from world.objects import company
from world.serializer import snapshot
from world.world import World


class MapFootprintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # 使用确定性客户端构建真实场景，避免测试访问外部模型服务。
        cls.runtime = build_runtime(reset_memory=False, llm_client=MockLLMClient())

    def test_compact_scene_registers_every_footprint_cell(self) -> None:
        world = self.runtime.world
        self.assertEqual([world.map.width, world.map.height], [25, 25])
        self.assertEqual(len(world.objects), 18)
        self.assertEqual(sum(len(obj.footprint) for obj in world.objects.values()), 52)
        for obj in world.objects.values():
            for row, col in obj.footprint:
                self.assertEqual(world.map.get_e(row, col), obj.id)

    def test_snapshot_exposes_layout_and_sprite_protocol(self) -> None:
        state = snapshot(self.runtime.world)
        by_id = {item["id"]: item for item in state["objects"]}
        self.assertEqual(by_id["bed_1"]["footprint"], [[1, 1], [1, 2]])
        self.assertEqual(by_id["bed_1"]["sprite_key"], "home_0")
        self.assertEqual(by_id["company_1"]["entrance"], [18, 2])
        self.assertEqual(len(state["map_design"]["decorations"]), 24)
        self.assertEqual(
            state["map_design"]["tile_sprites"],
            {"grass": "grass", "road": "stone_road"},
        )

    def test_path_never_crosses_building_footprints(self) -> None:
        world = self.runtime.world
        operator = Operator(world)
        path = operator._find_path((0, 0), (24, 24))
        blocked = {
            tuple(cell)
            for obj in world.objects.values()
            for cell in obj.footprint
        }
        self.assertTrue(path)
        self.assertTrue(all(tuple(cell) not in blocked for cell in path))

    def test_interaction_distance_uses_entire_footprint(self) -> None:
        world = World()
        target = company(
            "company_test",
            [2, 2],
            world,
            footprint=[[2, 2], [2, 3], [3, 2], [3, 3]],
            entrance=[3, 2],
            sprite_key="company_0",
        )
        agent = SimpleNamespace(
            get_position=lambda: [4, 4],
            config=SimpleNamespace(eat_distance_sq=2),
        )
        self.assertTrue(Operator(world)._within_interact_distance(agent, target))

    def test_exit_uses_empty_cell_around_entire_footprint(self) -> None:
        world = World()
        target = company(
            "company_test",
            [2, 2],
            world,
            footprint=[[2, 2], [2, 3], [3, 2], [3, 3]],
            entrance=[3, 2],
            sprite_key="company_0",
        )
        agent = SimpleNamespace(
            id="agent_test",
            position=[0, 0],
            inside_building_id="company_test",
        )
        world.add_agent(agent)
        world.map.remove(0, 0)
        agent.position = list(target.position)
        target.occupants.append(agent.id)

        result = Operator(world).exit_building(agent.id)

        self.assertIn("离开了", result)
        self.assertNotIn(agent.position, target.footprint)
        self.assertEqual(world.map.get_e(*agent.position), agent.id)

    def test_overlapping_footprints_are_rejected(self) -> None:
        world = World()
        company(
            "company_first",
            [2, 2],
            world,
            footprint=[[2, 2], [2, 3], [3, 2], [3, 3]],
        )
        with self.assertRaisesRegex(ValueError, "occupied"):
            company(
                "company_second",
                [3, 3],
                world,
                footprint=[[3, 3], [3, 4], [4, 3], [4, 4]],
            )

    def test_explicit_footprint_rejects_agent_cell(self) -> None:
        world = World()
        world.add_agent(SimpleNamespace(id="agent_test", position=[2, 2]))
        with self.assertRaisesRegex(ValueError, "occupied"):
            company(
                "company_test",
                [2, 2],
                world,
                footprint=[[2, 2], [2, 3], [3, 2], [3, 3]],
            )


if __name__ == "__main__":
    unittest.main()
