import asyncio
import json
import threading
import time
from persona.conversation import ConversationManager
from world.event import Event
from world.map import Map, build_default_map_design
from tools.operator_tools import register_operator_tools, Operator
from world.observer import observe
from persona.logger import get_logger

logger = get_logger(__name__)

class World:
    def __init__(self, opinion_updater=None, history_recorder=None, platform=None):
        self.time = 0
        self.map = Map(25, 25)
        self.map_design = build_default_map_design(self.map.width, self.map.height)
        self.agents = {}
        self.objects = {}
        self.movements = []
        self._world_lock = threading.Lock()
        self.conversation_policy = None   # set externally to enable conversation phase
        self.conversation_max_rounds = 3  # max conversation rounds per time step
        self.conversation_manager = ConversationManager(self)
        self.opinion_updater = opinion_updater
        self.history_recorder = history_recorder
        self.platform = platform          # set externally for news injection
        operator = Operator(self)
        self.tools, self.tools_prompt = register_operator_tools(operator)

    def add_agent(self, agent):
        self.agents[agent.id] = agent
        x, y = agent.position
        self.map.place(x, y, agent.id)

    def add_object(self, obj):
        self.objects[obj.id] = obj
        x, y = obj.position
        self.map.place(x, y, obj.id)

    def interact_inside_building(self, agent, *, record_history: bool = False, source: str = "tick") -> str | None:
        building_id = agent.inside_building_id
        if not building_id:
            return None

        old_satisfaction = agent.satisfaction.copy()
        old_urgency = agent.urgency.copy()
        with self._world_lock:
            target = self.objects.get(building_id)
            if target is None:
                agent.inside_building_id = None
                result = f"building {building_id} no longer exists; cleared inside_building_id"
                logger.warning("[World] agent=%s %s", agent.id, result)
            else:
                from world.objects import building
                if not isinstance(target, building):
                    agent.inside_building_id = None
                    result = f"object {building_id} is not a building; cleared inside_building_id"
                    logger.warning("[World] agent=%s %s", agent.id, result)
                else:
                    result = target.interact(agent)
                    logger.info(
                        "[World] tick=%d source=%s agent=%s building=%s auto_interact=%s",
                        self.time,
                        source,
                        agent.id,
                        building_id,
                        result,
                    )

        if record_history:
            feedback = f"auto_interact building {building_id}: {result}"
            agent.add_history("feedback: ", feedback)
            reward = sum(
                (agent.satisfaction.get(k, 0.0) - old_satisfaction.get(k, 0.0)) * old_urgency.get(k, 0.0)
                for k in agent.satisfaction
            )
            agent.add_history("reward: ", reward)
        return result

    def auto_interact_inside_buildings(self, agents=None) -> None:
        for agent in agents or list(self.agents.values()):
            if agent.sleeping:
                continue
            if agent.inside_building_id:
                self.interact_inside_building(agent, record_history=True, source="tick")

    def step(self):
        """同步入口（CLI 模式兼容），内部委托给 astep()。"""
        return asyncio.run(self.astep())

    async def astep(self):
        step_start = time.perf_counter()
        self.time += 1
        self.movements = []
        self.map.print_map()
        agents = list(self.agents.values())
        phase_start = time.perf_counter()
        self.auto_interact_inside_buildings(agents)
        building_elapsed = time.perf_counter() - phase_start

        # Phase 1: observe + step + execute 并发（所有 agent 的 LLM 调用同时飞行中）
        async def _agent_step_and_execute(agent):
            agent_start = time.perf_counter()
            try:
                if agent.sleeping:
                    agent.sleep_ticks_remaining -= 1
                    agent.tick_satisfaction()
                    if agent.sleep_ticks_remaining <= 0:
                        bed = self.objects.get(agent.sleeping_on_bed_id)
                        agent.wakeup(bed)
                    return
                obs = observe(agent, agent.config.observation_radius)
                action = await agent.astep(obs)
                reward = self.execute(agent, action)
                agent.append_trajectory(obs, action, reward)
            except Exception as e:
                logger.error("[World] agent %s 本轮执行失败: %s", agent.id, e, exc_info=True)
            finally:
                logger.debug(
                    "[Perf] tick=%d agent=%s phase=act elapsed=%.3fs",
                    self.time,
                    agent.id,
                    time.perf_counter() - agent_start,
                )

        phase_start = time.perf_counter()
        await asyncio.gather(*[_agent_step_and_execute(a) for a in agents])
        phase1_elapsed = time.perf_counter() - phase_start

        # Phase 2: reflect + tick_satisfaction 并发（reflect 的 LLM 调用同时飞行中）
        async def _agent_reflect(agent):
            agent_start = time.perf_counter()
            try:
                if not agent.sleeping:
                    await agent.aget_reflect()
                    agent.tick_satisfaction()
            except Exception as e:
                logger.error("[World] agent %s 反思失败: %s", agent.id, e, exc_info=True)
            finally:
                logger.debug(
                    "[Perf] tick=%d agent=%s phase=reflect elapsed=%.3fs",
                    self.time,
                    agent.id,
                    time.perf_counter() - agent_start,
                )

        phase_start = time.perf_counter()
        await asyncio.gather(*[_agent_reflect(a) for a in agents])
        phase2_elapsed = time.perf_counter() - phase_start

        phase_start = time.perf_counter()
        if self.opinion_updater:
            for agent in self.agents.values():
                self.opinion_updater.online_update(agent)
            if self.time % self.opinion_updater.config.offline_update_interval == 0:
                for agent in self.agents.values():
                    self.opinion_updater.offline_update(agent, self.agents)
        opinion_elapsed = time.perf_counter() - phase_start

        phase_start = time.perf_counter()
        if self.conversation_policy:
            self._conversation_phase(agents)
        conversation_elapsed = time.perf_counter() - phase_start

        phase_start = time.perf_counter()
        if self.platform is not None:
            from persona.news_events import NEWS_SCHEDULE
            news = NEWS_SCHEDULE.get(self.time)
            if news:
                self.platform.inject_news(
                    self.time, news["title"], news["content"],
                    news.get("opinion_index", 0.5),
                )
                for agent in self.agents.values():
                    agent._pending_social_notifications.append(
                        f"[新闻] {news['title']}：{news['content']}"
                    )
                    agent.add_history("news", f"{news['title']}：{news['content']}")
                logger.info("[World] tick=%d 新闻已投放，所有智能体自动阅览", self.time)
        news_elapsed = time.perf_counter() - phase_start

        phase_start = time.perf_counter()
        if self.history_recorder:
            self.history_recorder.record(self.time, list(self.agents.values()))
        history_elapsed = time.perf_counter() - phase_start

        logger.info(
            "[Perf] tick=%d world_total=%.3fs building=%.3fs act=%.3fs reflect=%.3fs opinion=%.3fs conversation=%.3fs news=%.3fs history=%.3fs agents=%d",
            self.time,
            time.perf_counter() - step_start,
            building_elapsed,
            phase1_elapsed,
            phase2_elapsed,
            opinion_elapsed,
            conversation_elapsed,
            news_elapsed,
            history_elapsed,
            len(agents),
        )

    def _conversation_phase(self, agents):
        self.conversation_manager.run_phase(
            agents,
            self.conversation_policy,
            self.conversation_max_rounds,
        )

    def execute(self, agent, action_str):
        old_satisfaction = agent.satisfaction.copy()
        old_urgency = agent.urgency.copy()
        if not action_str:
            return
        try:
            data = json.loads(action_str)
        except json.JSONDecodeError:
            logger.error("[World] JSON 解析失败，原始内容: %r", action_str)
            return

        if not data or "tool" not in data:
            logger.debug("[%s] 本轮选择不行动", agent.id)
            return

        tool = self.tools.get(data["tool"])
        if not tool:
            logger.warning("[World] 未知工具: %s", data.get("tool"))
            return

        args = data.get("args", {})
        args["operator_ID"] = agent.id
        try:
            feedback = tool.run(**args)
        except Exception as e:
            logger.error("[World] 工具 %s 执行失败 (agent=%s): %s", data["tool"], agent.id, e, exc_info=True)
            return
        agent.add_history("feedback: ", feedback)
        event = Event(
            type=data["tool"],
            actor=agent.id,
            info=feedback,
            time=self.time,
            position=agent.position
        )

        for other in self.agents.values():
            if other.id != agent.id and other.can_perceive(event):
                other.perceive(event)
                if data["tool"] == "speak":
                    target_str = args.get("ID", "")
                    if target_str == "<all>" or other.id in target_str.split():
                        other.receive_message(
                            agent.id,
                            args.get("content", ""),
                            args.get("response_to"),
                            args.get("session_id"),
                            args.get("intent"),
                            target_str,
                        )
        reward = sum(
            (agent.satisfaction.get(k, 0.0) - old_satisfaction.get(k, 0.0)) * old_urgency.get(k, 0.0)
            for k in agent.satisfaction
        )
        agent.add_history("reward: ", reward)
        return reward
