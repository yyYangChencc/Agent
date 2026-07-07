from __future__ import annotations

from copy import deepcopy
from typing import Any

from persona.config import AgentConfig
from persona.opinion.scale import OPINION_NEUTRAL, clamp_opinion
from persona.runtime import SimulationRuntime
from world.objects import bed, company, food, food_shop, playground


OBJECT_BUILDERS = {
    "bed": bed,
    "company": company,
    "food": food,
    "food_shop": food_shop,
    "playground": playground,
}


def build_runtime_from_spec(
    spec: dict[str, Any],
    *,
    conversation_max_rounds: int = 2,
    history_recorder=None,
    reset_memory: bool = True,
    config: AgentConfig | None = None,
    llm_client=None,
) -> SimulationRuntime:
    """按场景 spec 创建运行时，避免入口文件继续硬编码初始化逻辑。"""

    runtime = SimulationRuntime.build(
        config=config,
        conversation_max_rounds=conversation_max_rounds,
        llm_client=llm_client,
    )
    if reset_memory:
        runtime.mem.reset_all()

    if spec.get("map_design") is not None:
        runtime.world.map_design = deepcopy(spec["map_design"])
    if spec.get("default_opinion_topic"):
        # 场景可覆盖系统新闻议题，数据集场景复用同一观念评测链路。
        runtime.config.default_opinion_topic = str(spec["default_opinion_topic"])

    agents = _create_agents(runtime, spec.get("agents") or [])
    _apply_initial_agent_state(agents, spec.get("agents") or [], spec.get("default_initial_satisfaction") or {})
    _apply_trust(agents, spec.get("offline_trust") or [], "offline_trust")
    _apply_trust(agents, spec.get("online_trust") or [], "online_trust")
    _apply_follow_edges(agents, spec.get("follow_edges") or [])
    _register_influencers(runtime, spec.get("influencers") or [])
    _place_objects(runtime, spec.get("objects") or [])
    _seed_memories(runtime, spec, agents)

    runtime.world.news_schedule = deepcopy(spec.get("official_news_schedule") or {})
    runtime.world.influencer_schedule = deepcopy(spec.get("influencer_schedule") or {})
    runtime.world.history_recorder = history_recorder
    runtime.scenario_name = spec.get("name", "")
    runtime.world.scenario_name = runtime.scenario_name
    runtime.scenario_spec = deepcopy(spec)
    runtime.scenario_agent_ids = [agent["id"] for agent in spec.get("agents") or []]
    return runtime


def _create_agents(runtime: SimulationRuntime, agent_specs: list[dict[str, Any]]) -> dict[str, object]:
    agents = {}
    for item in agent_specs:
        agent_id = item["id"]
        agents[agent_id] = runtime.create_agent(
            agent_id,
            list(item["position"]),
            speaking_style=str(item.get("speaking_style") or ""),
            salary=float(item.get("salary", 0.0)),
        )
    return agents


def _apply_initial_agent_state(
    agents: dict[str, object],
    agent_specs: list[dict[str, Any]],
    default_satisfaction: dict[str, Any],
) -> None:
    for item in agent_specs:
        agent = agents[item["id"]]
        # opinion 只表示智能体接触系统新闻主题后形成的立场，开局必须保持未知/中立。
        agent.opinion = OPINION_NEUTRAL
        agent.opinion_scores.clear()
        agent.last_opinion_assessment = None
        agent.opinion_assessment_history.clear()
        agent.last_opinion_before_assessment = OPINION_NEUTRAL
        agent.opinion_seen_posts_buffer = []
        agent._last_opinion_assessment_tick = 0
        agent._last_opinion_evidence_signature = ""
        if "initial_opinion" in item:
            # 只有显式数据集场景写入初始观念；普通场景仍保持未知/中立。
            initial_opinion = clamp_opinion(float(item["initial_opinion"]))
            agent.opinion = initial_opinion
            agent.opinion_scores[agent.config.default_opinion_topic] = initial_opinion
            agent.last_opinion_before_assessment = initial_opinion
        # 场景默认满足度先落地，再允许单个智能体用 initial_satisfaction 或 initial_x 覆盖。
        initial_satisfaction = dict(default_satisfaction)
        per_agent_satisfaction = item.get("initial_satisfaction")
        if isinstance(per_agent_satisfaction, dict):
            initial_satisfaction.update(per_agent_satisfaction)
        for need_key in list(agent.satisfaction.keys()):
            field_name = f"initial_{need_key}"
            if field_name in item:
                initial_satisfaction[need_key] = item[field_name]
        for need_key, target_value in initial_satisfaction.items():
            if need_key not in agent.satisfaction:
                continue
            current_value = agent.satisfaction.get(need_key, 0.0)
            agent.update_satisfaction(need_key, float(target_value) - current_value)


def _apply_trust(agents: dict[str, object], edges: list[dict[str, Any]], attr: str) -> None:
    for edge in edges:
        source = agents.get(edge.get("source"))
        if source is None:
            continue
        getattr(source, attr)[str(edge["target"])] = float(edge["value"])


def _apply_follow_edges(agents: dict[str, object], edges: list[dict[str, Any]]) -> None:
    for edge in edges:
        follower = agents.get(edge.get("follower"))
        if follower is not None:
            follower.add_follower(str(edge["author"]))


def _register_influencers(runtime: SimulationRuntime, influencers: list[dict[str, Any]]) -> None:
    for influencer in influencers:
        runtime.platform.add_influencer(influencer)


def _place_objects(runtime: SimulationRuntime, object_specs: list[dict[str, Any]]) -> None:
    for item in object_specs:
        kind = item["kind"]
        params = dict(item.get("params") or {})
        builder = OBJECT_BUILDERS[kind]
        if kind == "food":
            builder(item["id"], params.get("num", 1), params.get("provide", 20), list(item["position"]), runtime.world)
        elif kind == "company":
            builder(
                item["id"],
                list(item["position"]),
                runtime.world,
                salary=params.get("salary", 10),
                relax_cost=params.get("relax_cost", 10),
            )
        elif kind == "food_shop":
            builder(
                item["id"],
                list(item["position"]),
                runtime.world,
                food_num=params.get("food_num", 10),
                provide=params.get("provide", 20),
                price=params.get("price", 5),
            )
        elif kind == "playground":
            builder(
                item["id"],
                list(item["position"]),
                runtime.world,
                provide=params.get("provide", 10),
                price=params.get("price", 3),
            )
        else:
            builder(item["id"], list(item["position"]), runtime.world)


def _seed_memories(runtime: SimulationRuntime, spec: dict[str, Any], agents: dict[str, object]) -> None:
    map_memory = spec.get("map_memory")
    if map_memory:
        for agent_id in agents:
            runtime.mem.store_agent_memory(
                agent_id,
                str(map_memory),
                memory_type="semantic",
                task="map_navigation",
                object_id="scenario_map",
                importance=0.9,
                confidence=1.0,
            )
    facility_memory = _build_facility_memory(spec)
    if facility_memory:
        for agent_id in agents:
            runtime.mem.store_agent_memory(
                agent_id,
                facility_memory,
                memory_type="semantic",
                task="facility_navigation",
                object_id="scenario_facilities",
                importance=0.95,
                confidence=1.0,
            )
    for item in spec.get("memories") or []:
        runtime.mem.store_agent_memory(
            item["agent_id"],
            item["content"],
            memory_type=item.get("memory_type", "system"),
            task=item.get("task"),
            object_id=item.get("object_id"),
            importance=float(item.get("importance", 0.9)),
            confidence=float(item.get("confidence", 1.0)),
        )


def _build_facility_memory(spec: dict[str, Any]) -> str:
    """把场景设施压缩成每个智能体开局可用的导航记忆。"""

    object_regions = {}
    map_design = spec.get("map_design")
    if isinstance(map_design, dict) and isinstance(map_design.get("object_regions"), dict):
        object_regions = map_design["object_regions"]

    lines = ["场景设施记忆：坐标格式为 [row, col]。"]
    for item in spec.get("objects") or []:
        if not isinstance(item, dict):
            continue
        object_id = str(item.get("id") or "")
        kind = str(item.get("kind") or "")
        position = item.get("position")
        params = item.get("params") if isinstance(item.get("params"), dict) else {}
        region = object_regions.get(object_id, {}) if isinstance(object_regions, dict) else {}
        entrance = region.get("entrance") if isinstance(region, dict) else None
        region_id = region.get("region_id") if isinstance(region, dict) else None
        usage = _facility_usage(kind, params)
        detail = f"- {object_id}: kind={kind}, position={position}"
        if entrance is not None:
            detail += f", entrance={entrance}"
        if region_id:
            detail += f", region={region_id}"
        if usage:
            detail += f", usage={usage}"
        lines.append(detail)
    return "\n".join(lines) if len(lines) > 1 else ""


def _facility_usage(kind: str, params: dict[str, Any]) -> str:
    """生成设施用途说明，供 LLM 决策时直接使用。"""

    if kind == "bed":
        return "sleep 可恢复 relax"
    if kind == "company":
        return f"enter_building 后工作赚钱，salary={params.get('salary', 10)}, relax_cost={params.get('relax_cost', 10)}"
    if kind == "food_shop":
        return f"enter_building 后购买食物，price={params.get('price', 5)}, provide={params.get('provide', 20)}"
    if kind == "playground":
        return f"enter_building 后付费放松，price={params.get('price', 3)}, provide={params.get('provide', 10)}"
    if kind == "food":
        return f"eat 后补充 satiety，provide={params.get('provide', 20)}"
    return ""
