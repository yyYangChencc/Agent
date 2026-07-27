from __future__ import annotations

import unittest
from collections import Counter

from persona.llm.mock_client import MockLLMClient
from scenarios.iac_gay_marriage import COMMUNITY_ORIGINS
from scenarios.iac_gay_marriage_30_balanced import SPEC as BALANCED_SPEC, build_runtime
from scenarios.iac_gay_marriage_30_cross_side import SPEC as CROSS_SIDE_SPEC
from scenarios.iac_gay_marriage_30_same_side_isolated import SPEC as SAME_SIDE_SPEC
from tools.operator_tools import Operator
from world.serializer import snapshot


class IAC30MapLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # 确定性客户端只用于构建真实运行时，不访问外部模型服务。
        cls.runtime = build_runtime(reset_memory=False, llm_client=MockLLMClient())

    def test_three_network_modes_share_one_map(self) -> None:
        self.assertEqual(SAME_SIDE_SPEC["map_design"], CROSS_SIDE_SPEC["map_design"])
        self.assertEqual(CROSS_SIDE_SPEC["map_design"], BALANCED_SPEC["map_design"])
        self.assertEqual(SAME_SIDE_SPEC["objects"], CROSS_SIDE_SPEC["objects"])
        self.assertEqual(CROSS_SIDE_SPEC["objects"], BALANCED_SPEC["objects"])
        for specs in (SAME_SIDE_SPEC, CROSS_SIDE_SPEC, BALANCED_SPEC):
            positions = {item["id"]: item["position"] for item in specs["agents"]}
            self.assertEqual(
                positions,
                {item["id"]: item["position"] for item in BALANCED_SPEC["agents"]},
            )

    def test_object_counts_and_footprints(self) -> None:
        objects = BALANCED_SPEC["objects"]
        self.assertEqual(
            Counter(item["kind"] for item in objects),
            {"bed": 30, "company": 8, "food_shop": 8, "playground": 4},
        )
        self.assertEqual(sum(len(item["footprint"]) for item in objects), 140)
        for item in objects:
            expected_size = 2 if item["kind"] == "bed" else 4
            self.assertEqual(len(item["footprint"]), expected_size)
            self.assertIn(item["position"], item["footprint"])
            self.assertTrue(item["sprite_key"])

    def test_objects_stay_inside_their_functional_districts(self) -> None:
        for item in BALANCED_SPEC["objects"]:
            row_offset, col_offset = COMMUNITY_ORIGINS[item["community_id"]]
            local_cells = [
                (row - row_offset, col - col_offset)
                for row, col in item["footprint"]
            ]
            if item["kind"] == "bed":
                bounds = (0, 0, 23, 23)
            elif item["kind"] == "food_shop":
                bounds = (0, 25, 23, 47)
            elif item["kind"] == "company":
                bounds = (25, 0, 47, 23)
            else:
                bounds = (25, 25, 47, 47)
            row_start, col_start, row_end, col_end = bounds
            self.assertTrue(
                all(
                    row_start <= row <= row_end and col_start <= col <= col_end
                    for row, col in local_cells
                )
            )

    def test_personal_bed_spawns_are_outside_buildings(self) -> None:
        objects = {item["id"]: item for item in BALANCED_SPEC["objects"]}
        object_regions = BALANCED_SPEC["map_design"]["object_regions"]
        occupied = {
            tuple(cell)
            for item in objects.values()
            for cell in item["footprint"]
        }
        spawn_positions = []
        for agent in BALANCED_SPEC["agents"]:
            bed_id = agent["personal_bed_id"]
            entrance = object_regions[bed_id]["entrance"]
            self.assertEqual(agent["position"], entrance)
            self.assertNotIn(tuple(entrance), occupied)
            self.assertEqual(objects[bed_id]["params"]["owner_agent_id"], agent["id"])
            spawn_positions.append(tuple(agent["position"]))
        self.assertEqual(len(spawn_positions), len(set(spawn_positions)))

    def test_decorations_are_fixed_and_clear_of_roads_and_objects(self) -> None:
        design = BALANCED_SPEC["map_design"]
        decorations = design["decorations"]
        self.assertEqual(Counter(item["kind"] for item in decorations), {"tree": 32, "shrub": 64})
        positions = [tuple(item["pos"]) for item in decorations]
        self.assertEqual(len(positions), len(set(positions)))
        blocked = {
            tuple(cell)
            for item in BALANCED_SPEC["objects"]
            for cell in item["footprint"]
        }
        roads = {tuple(cell) for road in design["roads"] for cell in road["cells"]}
        self.assertTrue(all(pos not in blocked and pos not in roads for pos in positions))

    def test_runtime_registers_all_cells_and_paths_avoid_them(self) -> None:
        state = snapshot(self.runtime.world)
        self.assertEqual(state["map_size"], [100, 100])
        self.assertEqual(len(state["agents"]), 30)
        self.assertEqual(len(state["objects"]), 50)
        blocked = set()
        for item in state["objects"]:
            for row, col in item["footprint"]:
                blocked.add((row, col))
                self.assertEqual(self.runtime.world.map.get_e(row, col), item["id"])
        start = tuple(state["agents"][0]["pos"])
        path = Operator(self.runtime.world)._find_path(start, (97, 97))
        self.assertTrue(path)
        self.assertTrue(all(tuple(cell) not in blocked for cell in path))


if __name__ == "__main__":
    unittest.main()
