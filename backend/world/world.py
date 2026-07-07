import asyncio
import json
import threading
import time
from persona.conversation.manager import ConversationManager
from world.event import Event, SpeakingEvent
from world.map import Map, build_default_map_design
from tools.operator_tools import register_operator_tools, Operator
from world.observer import observe
from persona.logger import get_logger
from persona.need_events import apply_passive_need_decay
from persona.opinion.scale import OPINION_NEUTRAL

logger = get_logger(__name__)

class World:
    """仿真世界的调度中心。

    World 只负责推进 tick、管理地图对象和调用各子系统；具体行动决策、
    心理评测、观念评测和社交平台逻辑分别由注入的服务完成。
    """

    def __init__(
        self,
        history_recorder=None,
        platform=None,
        psychological_assessor=None,
        opinion_assessor=None,
    ):
        self.time = 0
        self.map = Map(25, 25)
        self.map_design = build_default_map_design(self.map.width, self.map.height)
        self.agents = {}
        self.objects = {}
        self.movements = []
        # execute 和工具函数都会保护世界状态；可重入锁避免同线程二次加锁卡死。
        self._world_lock = threading.RLock()
        self.conversation_policy = None   # 外部注入后才启用对话阶段
        self.conversation_max_rounds = 3  # 每个 tick 内允许的最大对话轮数
        self.conversation_manager = ConversationManager(self)
        self.history_recorder = history_recorder
        self.platform = platform          # 社交平台由 runtime 注入，也用于新闻投放
        self.psychological_assessor = psychological_assessor
        self.opinion_assessor = opinion_assessor
        self.news_schedule: dict = {}          # 场景注入的官方新闻排期
        self.influencer_schedule: dict = {}    # 场景注入的线上投放者发帖排期
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
        """执行建筑内自动交互。

        智能体进入公司、食品店、游乐场等建筑后，后续 tick 会自动调用建筑交互。
        这里记录交互前的需求和急迫度，用于把需求改善转换为简单 reward。
        """

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
        """推进一个完整异步 tick。

        tick 顺序很重要：先在时间步开头投放新闻通知，再并发执行每个智能体的
        观察、决策、行动、反思和需求衰减；随后统一处理对话，并并发执行心理
        评测和观念评测，最后写历史记录和维护记忆。
        """

        step_start = time.perf_counter()
        self.time += 1
        self.movements = []
        self.map.print_map()
        agents = list(self.agents.values())
        if self.platform is not None:
            self.platform.time = self.time
        for agent in agents:
            # 每个 tick 只记录本轮新产生的动作和社交动作，避免历史导出串用上一轮结果。
            agent.last_action = {}
            agent.last_social_action = {}
            agent.did_move_this_tick = False
            agent.did_work_this_tick = False
        sleeping_at_tick_start = {agent.id for agent in agents if agent.sleeping}

        phase_start = time.perf_counter()
        self._inject_scheduled_news()
        news_elapsed = time.perf_counter() - phase_start

        phase_start = time.perf_counter()
        self.auto_interact_inside_buildings(agents)
        building_elapsed = time.perf_counter() - phase_start

        # 行动与反思合并在同一个 agent task 内；睡眠智能体只推进睡眠恢复，
        # 不 observe，因此新闻通知会一直积压到醒来后的下一次 observe。
        async def _agent_tick(agent):
            agent_start = time.perf_counter()
            try:
                if agent.sleeping:
                    agent.sleep_ticks_remaining -= 1
                    agent.tick_sleep_recovery()
                    if agent.sleep_ticks_remaining <= 0:
                        bed = self.objects.get(agent.sleeping_on_bed_id)
                        agent.wakeup(bed)
                    return
                obs = observe(agent, agent.config.observation_radius)
                action = await agent.astep(obs)
                reward = self.execute(agent, action)
                agent.append_trajectory(obs, action, reward)
                if not agent.sleeping and agent.id not in sleeping_at_tick_start:
                    await agent.aget_reflect()
                    agent.tick_satisfaction()
            except Exception as e:
                logger.error("[World] agent %s 本轮执行失败: %s", agent.id, e, exc_info=True)
            finally:
                logger.debug(
                    "[Perf] tick=%d agent=%s phase=agent_tick elapsed=%.3fs",
                    self.time,
                    agent.id,
                    time.perf_counter() - agent_start,
                )

        phase_start = time.perf_counter()
        await asyncio.gather(*[_agent_tick(a) for a in agents])
        agent_tick_elapsed = time.perf_counter() - phase_start

        phase_start = time.perf_counter()
        if self.conversation_policy:
            self._conversation_phase(agents)
        conversation_elapsed = time.perf_counter() - phase_start

        phase_start = time.perf_counter()
        await self._assess_agents(agents)
        assessment_elapsed = time.perf_counter() - phase_start

        phase_start = time.perf_counter()
        if self.history_recorder:
            self.history_recorder.record(self.time, list(self.agents.values()), platform=self.platform)
        history_elapsed = time.perf_counter() - phase_start

        phase_start = time.perf_counter()
        self._maintain_agent_memories()
        memory_maintenance_elapsed = time.perf_counter() - phase_start

        logger.info(
            "[Perf] tick=%d world_total=%.3fs news=%.3fs building=%.3fs agent_tick=%.3fs conversation=%.3fs assessment=%.3fs history=%.3fs memory_maintenance=%.3fs agents=%d",
            self.time,
            time.perf_counter() - step_start,
            news_elapsed,
            building_elapsed,
            agent_tick_elapsed,
            conversation_elapsed,
            assessment_elapsed,
            history_elapsed,
            memory_maintenance_elapsed,
            len(agents),
        )

    def _conversation_phase(self, agents):
        self.conversation_manager.run_phase(
            agents,
            self.conversation_policy,
            self.conversation_max_rounds,
        )

    def _inject_scheduled_news(self) -> None:
        """在 tick 开头投放系统新闻，只写入通知，不强制智能体观看。"""

        if self.platform is None:
            return

        news_items = self._scheduled_items(self.news_schedule.get(self.time))
        for news in news_items:
            self._inject_one_official_news(news)
        if news_items:
            logger.info("[World] tick=%d 系统新闻已投放为 observe 通知 count=%d", self.time, len(news_items))

        influencer_items = self._scheduled_items(self.influencer_schedule.get(self.time), label="投放者排期")
        self._inject_influencer_posts(influencer_items)

    def _inject_one_official_news(self, news: dict) -> None:
        """投放一条官方新闻，并把通知放入所有实体智能体的观察队列。"""

        topic = str(news.get("topic") or "")
        self.platform.inject_news(
            self.time,
            news["title"],
            news["content"],
            topic=topic,
            opinion_index=news.get("opinion_index", OPINION_NEUTRAL),
        )
        notification = f"[系统新闻][{topic or '未指定主题'}] {news['title']}：{news['content']}"
        for agent in self.agents.values():
            if topic:
                agent.current_focus = topic
            agent._pending_social_notifications.append(notification)

    def _inject_influencer_posts(self, posts: list[dict]) -> None:
        """投放场景中的无实体账号帖子，不产生 observe 系统通知。"""

        if not posts:
            return
        for item in posts:
            topic = str(item.get("topic") or "")
            self.platform.inject_influencer_post(
                self.time,
                item["author_id"],
                item["content"],
                topic=topic,
                opinion_index=item.get("opinion_index", OPINION_NEUTRAL),
                is_rumor=bool(item.get("is_rumor", False)),
            )
        logger.info("[World] tick=%d 投放者帖子已发布 count=%d", self.time, len(posts))

    def _scheduled_items(self, scheduled, label: str = "系统新闻排期") -> list[dict]:
        """把排期统一成列表，支持单条 dict 和同 tick 多条列表。"""

        if scheduled is None:
            return []
        if isinstance(scheduled, dict):
            return [scheduled]
        if isinstance(scheduled, list):
            return [item for item in scheduled if isinstance(item, dict)]
        logger.warning("[World] tick=%d %s格式无效: %r", self.time, label, scheduled)
        return []

    async def _assess_agents(self, agents) -> None:
        """conversation 后并发评测所有智能体；单个智能体内心理评测先于观念评测。"""

        async def _assess_agent(agent):
            try:
                apply_passive_need_decay(agent, self.time)
                if self.psychological_assessor:
                    self.psychological_assessor.observe_agent_tick(agent, self.time)
                    await self.psychological_assessor.amaybe_assess(agent, self.time)
                if self.opinion_assessor:
                    # 观念评测事件触发；无新证据时不进入 LLM，减少无效评测调用。
                    should_assess = True
                    if hasattr(self.opinion_assessor, "should_assess_agent"):
                        should_assess = self.opinion_assessor.should_assess_agent(agent, self.time)
                    if should_assess:
                        await self.opinion_assessor.aassess_agent(agent, self.time)
            except Exception as exc:
                logger.warning("[World] assessment failed for %s at tick=%d: %s", agent.id, self.time, exc, exc_info=True)

        await asyncio.gather(*[_assess_agent(agent) for agent in agents])

    def _maintain_agent_memories(self) -> None:
        """每个 tick 末执行轻量记忆维护。

        当前维护只做低价值原始事件失效和未解决冲突统计，不在世界主循环里触发
        重型 LLM consolidation。
        """

        for agent in self.agents.values():
            mem = getattr(agent, "mem", None)
            if mem is None or not hasattr(mem, "maintain_agent_memory"):
                continue
            try:
                mem.maintain_agent_memory(agent.id, current_time=self.time)
            except Exception as exc:
                logger.debug("[World] memory maintenance failed for %s: %s", agent.id, exc)

    def execute(self, agent, action_str):
        """解析 LLM 决策 JSON 中的 action 并调用对应工具。

        工具调用失败不会中断整个 tick；成功后会生成事件并通知可感知范围内的智能体。
        记忆写入放在这里，是因为只有 execute 才知道动作是否实际执行、反馈和 reward。
        """

        old_satisfaction = agent.satisfaction.copy()
        old_urgency = agent.urgency.copy()
        if not action_str:
            return
        try:
            decision = json.loads(action_str)
        except json.JSONDecodeError:
            logger.error("[World] JSON 解析失败，原始内容: %r", action_str)
            return

        data, think = self._action_from_decision(decision)
        if think:
            agent.add_history("decision_think", think)
        if not data or "tool" not in data:
            logger.debug("[%s] 本轮选择不行动", agent.id)
            agent.last_action = {
                "tool": "",
                "args": {},
                "think": think,
                "feedback": "",
                "reward": None,
            }
            self._store_action_memory(agent, decision, "", None)
            return

        tool = self.tools.get(data["tool"])
        if not tool:
            logger.warning("[World] 未知工具: %s", data.get("tool"))
            agent.last_action = {
                "tool": data.get("tool", ""),
                "args": data.get("args", {}),
                "think": think,
                "feedback": "tool not found",
                "reward": None,
            }
            self._store_action_memory(agent, decision, "tool not found", None)
            return

        with self._world_lock:
            args = data.get("args", {})
            args["operator_ID"] = agent.id
            try:
                feedback = tool.run(**args)
            except Exception as e:
                logger.error("[World] 工具 %s 执行失败 (agent=%s): %s", data["tool"], agent.id, e, exc_info=True)
                agent.last_action = {
                    "tool": data["tool"],
                    "args": args,
                    "think": think,
                    "feedback": "tool execution failed",
                    "reward": None,
                }
                self._store_action_memory(agent, decision, "tool execution failed", None)
                return
            agent.add_history("feedback: ", feedback)
            acted = self._acted_from_action(data["tool"], args)
            if data["tool"] == "speak":
                event = SpeakingEvent(
                    type=data["tool"],
                    actor=agent.id,
                    info=feedback,
                    time=self.time,
                    response_to=args.get("response_to"),
                    position=agent.position,
                    acted=acted,
                    intent=args.get("intent"),
                    social_valence=args.get("social_valence"),
                    topic=args.get("topic", ""),
                    topic_stance=args.get("topic_stance"),
                )
            else:
                event = Event(
                    type=data["tool"],
                    actor=agent.id,
                    info=feedback,
                    time=self.time,
                    position=agent.position,
                    acted=acted,
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
                                args.get("social_valence"),
                                args.get("topic", ""),
                                args.get("topic_stance"),
                            )
            reward = sum(
                (agent.satisfaction.get(k, 0.0) - old_satisfaction.get(k, 0.0)) * old_urgency.get(k, 0.0)
                for k in agent.satisfaction
            )
            agent.add_history("reward: ", reward)
            agent.last_action = {
                "tool": data["tool"],
                "args": args,
                "think": think,
                "feedback": feedback,
                "reward": reward,
            }
            self._store_action_memory(agent, decision, feedback, reward)
            return reward

    def _store_action_memory(self, agent, decision: dict, feedback, reward: float | None) -> None:
        """把已执行或失败的动作结果写入记忆系统。"""

        mem = getattr(agent, "mem", None)
        if mem is None or not hasattr(mem, "store_action_result"):
            return
        mem.store_action_result(
            agent.id,
            decision=decision,
            feedback=feedback,
            reward=reward,
            world_time=self.time,
        )

    def _acted_from_action(self, tool_name: str, args: dict):
        if tool_name == "social_step":
            return "social_platform"
        return args.get("ID") or args.get("post_id")

    def _action_from_decision(self, decision):
        if not isinstance(decision, dict):
            return {}, ""
        think = str(decision.get("think", "") or "")
        action = decision.get("action", {})
        if not isinstance(action, dict):
            return {}, think
        return action, think
