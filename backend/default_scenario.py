from __future__ import annotations

from persona.runtime import SimulationRuntime
from world.objects import bed, company, food, food_shop, playground


DEFAULT_AGENT_IDS = ["agent_1", "agent_2", "agent_3", "agent_4", "agent_5"]


def build_default_runtime(
    *,
    conversation_max_rounds: int = 2,
    history_recorder=None,
    reset_memory: bool = True,
) -> SimulationRuntime:
    """Build the default five-agent town scenario used by web and CLI entrypoints."""
    runtime = SimulationRuntime.build(conversation_max_rounds=conversation_max_rounds)
    if reset_memory:
        runtime.mem.reset_all()

    agents = _create_default_agents(runtime)
    _configure_opinions_and_money(agents)
    _configure_trust(agents)
    _configure_followers(agents)
    _place_default_objects(runtime)
    _seed_default_memories(runtime)

    runtime.world.history_recorder = history_recorder
    return runtime


def build_cli_demo_runtime(
    *,
    conversation_max_rounds: int = 2,
    reset_memory: bool = True,
) -> SimulationRuntime:
    """Build the smaller CLI opinion demo previously defined in main.py."""
    runtime = SimulationRuntime.build(conversation_max_rounds=conversation_max_rounds)
    if reset_memory:
        runtime.mem.reset_all()

    agents = {
        "agent_1": runtime.create_agent(
            "agent_1",
            [3, 3],
            speaking_style="沉稳、措辞谨慎",
        ),
        "agent_2": runtime.create_agent(
            "agent_2",
            [4, 2],
            speaking_style="热情、喜欢分享",
        ),
        "agent_3": runtime.create_agent(
            "agent_3",
            [6, 6],
            speaking_style="理性、措辞中立",
        ),
        "agent_4": runtime.create_agent(
            "agent_4",
            [2, 8],
            speaking_style="直接、充满激情",
        ),
        "agent_5": runtime.create_agent(
            "agent_5",
            [8, 4],
            speaking_style="温和、善于调解",
        ),
    }

    _configure_opinions_and_money(agents, initial_money=None)
    _configure_trust(agents)
    _configure_followers(agents)

    food("food_1", 1, 2, [10, 10], runtime.world)
    food("food_2", 1, 2, [5, 15], runtime.world)
    food("food_3", 1, 2, [18, 5], runtime.world)

    runtime.mem.store_agent_memory(
        "agent_1",
        "在（10，10）附近可能存在食物",
        memory_type="system",
        importance=0.9,
    )
    runtime.mem.store_agent_memory(
        "agent_4",
        "在（5，15）附近可能存在食物",
        memory_type="system",
        importance=0.9,
    )
    runtime.mem.store_agent_memory(
        "agent_5",
        "在（18，5）附近可能存在食物",
        memory_type="system",
        importance=0.9,
    )

    return runtime


def _create_default_agents(runtime: SimulationRuntime) -> dict[str, object]:
    return {
        "agent_1": runtime.create_agent(
            "agent_1",
            [3, 3],
            speaking_style="沉稳、措辞谨慎",
        ),
        "agent_2": runtime.create_agent(
            "agent_2",
            [4, 2],
            speaking_style="热情、喜欢分享",
        ),
        "agent_3": runtime.create_agent(
            "agent_3",
            [6, 6],
            speaking_style="理性、措辞中立",
        ),
        "agent_4": runtime.create_agent(
            "agent_4",
            [2, 9],
            speaking_style="直接、充满激情",
        ),
        "agent_5": runtime.create_agent(
            "agent_5",
            [8, 4],
            speaking_style="温和、善于调解",
        ),
    }


_USE_DEFAULT_INITIAL_MONEY = object()

DEFAULT_INITIAL_MONEY = {
    "agent_1": 8.0,
    "agent_2": 14.0,
    "agent_3": 18.0,
    "agent_4": 24.0,
    "agent_5": 12.0,
}


def _configure_opinions_and_money(
    agents: dict[str, object],
    initial_money: float | dict[str, float] | None | object = _USE_DEFAULT_INITIAL_MONEY,
) -> None:
    agents["agent_1"].opinion = -0.65
    agents["agent_2"].opinion = -0.20
    agents["agent_3"].opinion = 0.00
    agents["agent_4"].opinion = 0.65
    agents["agent_5"].opinion = 0.25

    if initial_money is _USE_DEFAULT_INITIAL_MONEY:
        initial_money = DEFAULT_INITIAL_MONEY

    if initial_money is not None:
        for agent_id, agent in agents.items():
            target_money = (
                initial_money.get(agent_id, 0.0)
                if isinstance(initial_money, dict)
                else initial_money
            )
            agent.update_satisfaction("money", target_money - agent.satisfaction.get("money", 0.0))


def _configure_trust(agents: dict[str, object]) -> None:
    agents["agent_1"].offline_trust["agent_2"] = 0.75
    agents["agent_1"].offline_trust["agent_3"] = 0.65
    agents["agent_2"].offline_trust["agent_1"] = 0.75
    agents["agent_2"].offline_trust["agent_3"] = 0.70
    agents["agent_3"].offline_trust["agent_1"] = 0.65
    agents["agent_3"].offline_trust["agent_2"] = 0.70
    agents["agent_4"].offline_trust["agent_5"] = 0.80
    agents["agent_5"].offline_trust["agent_4"] = 0.80
    agents["agent_3"].offline_trust["agent_5"] = 0.62
    agents["agent_5"].offline_trust["agent_3"] = 0.62

    agents["agent_1"].online_trust["agent_2"] = 0.55
    agents["agent_1"].online_trust["agent_3"] = 0.60
    agents["agent_2"].online_trust["agent_1"] = 0.50
    agents["agent_2"].online_trust["agent_4"] = 0.35
    agents["agent_3"].online_trust["agent_1"] = 0.55
    agents["agent_3"].online_trust["agent_4"] = 0.55
    agents["agent_3"].online_trust["agent_5"] = 0.60
    agents["agent_4"].online_trust["agent_5"] = 0.65
    agents["agent_4"].online_trust["agent_1"] = 0.30
    agents["agent_5"].online_trust["agent_4"] = 0.60
    agents["agent_5"].online_trust["agent_3"] = 0.65


def _configure_followers(agents: dict[str, object]) -> None:
    agents["agent_1"].add_follower("agent_2")
    agents["agent_1"].add_follower("agent_3")
    agents["agent_2"].add_follower("agent_1")
    agents["agent_2"].add_follower("agent_3")
    agents["agent_2"].add_follower("agent_5")
    agents["agent_3"].add_follower("agent_1")
    agents["agent_3"].add_follower("agent_4")
    agents["agent_3"].add_follower("agent_5")
    agents["agent_4"].add_follower("agent_5")
    agents["agent_4"].add_follower("agent_3")
    agents["agent_5"].add_follower("agent_4")
    agents["agent_5"].add_follower("agent_3")
    agents["agent_5"].add_follower("agent_2")


def _place_default_objects(runtime: SimulationRuntime) -> None:
    world = runtime.world

    bed("bed_1", [2, 2], world)
    bed("bed_2", [2, 4], world)
    bed("bed_3", [2, 6], world)
    bed("bed_4", [2, 8], world)
    bed("bed_5", [2, 10], world)

    company("company_1", [20, 3], world, salary=8, relax_cost=8)
    company("company_2", [22, 6], world, salary=12, relax_cost=12)

    food_shop("shop_1", [2, 18], world, food_num=20, provide=30, price=6)
    food_shop("shop_2", [4, 21], world, food_num=15, provide=20, price=4)

    playground("playground_1", [20, 20], world, provide=18, price=5)

    food("food_1", 3, 20, [10, 8], world)
    food("food_2", 3, 20, [12, 14], world)
    food("food_3", 3, 20, [8, 18], world)
    food("food_4", 3, 20, [15, 5], world)
    food("food_5", 3, 20, [16, 20], world)


def _format_position(pos: list[int]) -> str:
    return f"({pos[0]}, {pos[1]})"


def _format_bounds(bounds: list[int]) -> str:
    return f"({bounds[0]}, {bounds[1]}) 到 ({bounds[2]}, {bounds[3]})"


def _build_map_memory(runtime: SimulationRuntime) -> str:
    world = runtime.world
    design = world.map_design
    object_regions = design["object_regions"]

    region_lines = []
    for region in design["regions"]:
        region_objects = []
        for obj_id, obj in world.objects.items():
            relation = object_regions.get(obj_id)
            if relation and relation["region_id"] == region["id"]:
                region_objects.append(f"{obj_id}{_format_position(obj.position)}")
        object_text = "、".join(region_objects) if region_objects else "无"
        region_lines.append(
            f"{region['name']}({region['id']})范围{_format_bounds(region['bounds'])}："
            f"{region['description']} 对象：{object_text}。"
        )

    road_text = "、".join(f"{road['name']}({road['id']})" for road in design["roads"])
    return (
        "地图信息：坐标格式为[row, col]。"
        f"区域：{''.join(region_lines)}"
        f"道路：{road_text}。"
    )


def _seed_default_memories(runtime: SimulationRuntime) -> None:
    map_info = _build_map_memory(runtime)
    for agent_id in DEFAULT_AGENT_IDS:
        runtime.mem.store_agent_memory(
            agent_id,
            map_info,
            memory_type="semantic",
            task="map_navigation",
            object_id="default_map",
            importance=0.9,
            confidence=1.0,
        )
