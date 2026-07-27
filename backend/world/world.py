import asyncio
import json
import threading
import time
from persona.conversation.manager import ConversationManager
from world.event import Event, SpeakingEvent
from world.map import Map, build_default_map_design
from tools.operator_tools import register_operator_tools, Operator
from world.observer import observe
from persona.logger import get_logger, set_log_tick
from persona.need_events import apply_passive_need_decay, consume_pending_need_events
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
        # 新世界从时间步 0 开始，避免 reset 后日志继续沿用旧实验时间步。
        set_log_tick(0)
        self.time = 0
        self.map = Map(25, 25)
        self.map_design = build_default_map_design(self.map.width, self.map.height)
        self.agents = {}
        self.objects = {}
        self.movements = []
        # execute 和工具函数都会保护世界状态；可重入锁避免同线程二次加锁卡死。
        self._world_lock = threading.RLock()
        self._episode_lock = threading.RLock()
        self._next_episode_id = 1
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
        """校验并登记对象占用的全部地图格。"""

        if obj.id in self.objects:
            raise ValueError(f"duplicate object id: {obj.id}")
        if (
            not isinstance(obj.position, list)
            or len(obj.position) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in obj.position)
        ):
            raise ValueError(f"object position must be [row, col]: {obj.id}")
        entrance = getattr(obj, "entrance", None)
        if entrance is not None:
            if (
                not isinstance(entrance, list)
                or len(entrance) != 2
                or any(isinstance(value, bool) or not isinstance(value, int) for value in entrance)
            ):
                raise ValueError(f"object entrance must be [row, col]: {obj.id}")
            if entrance[0] < 0 or entrance[1] < 0 or entrance[0] >= self.map.height or entrance[1] >= self.map.width:
                raise ValueError(f"object entrance out of bounds {entrance}: {obj.id}")
        sprite_key = getattr(obj, "sprite_key", None)
        if sprite_key is not None and (not isinstance(sprite_key, str) or not sprite_key):
            raise ValueError(f"object sprite_key must be None or a non-empty string: {obj.id}")
        footprint = getattr(obj, "footprint", None)
        if not isinstance(footprint, list) or not footprint:
            raise ValueError(f"object footprint must be a non-empty list: {obj.id}")
        normalized: list[list[int]] = []
        seen: set[tuple[int, int]] = set()
        for cell in footprint:
            if (
                not isinstance(cell, list)
                or len(cell) != 2
                or any(isinstance(value, bool) or not isinstance(value, int) for value in cell)
            ):
                raise ValueError(f"object footprint cell must be [row, col]: {obj.id}")
            row, col = cell
            key = (row, col)
            if key in seen:
                raise ValueError(f"object footprint contains duplicate cell {cell}: {obj.id}")
            if row < 0 or col < 0 or row >= self.map.height or col >= self.map.width:
                raise ValueError(f"object footprint cell out of bounds {cell}: {obj.id}")
            if not self.map.is_empty(row, col):
                occupied_id = self.map.get_e(row, col)
                # 旧单格场景允许对象覆盖出生点；显式 footprint 始终执行严格碰撞校验。
                if occupied_id in self.agents and not getattr(obj, "has_explicit_footprint", False):
                    logger.warning(
                        "[World] legacy object %s overwrites agent %s at %s",
                        obj.id,
                        occupied_id,
                        cell,
                    )
                    seen.add(key)
                    normalized.append([row, col])
                    continue
                raise ValueError(
                    f"object footprint cell {cell} is occupied by {occupied_id}: {obj.id}"
                )
            seen.add(key)
            normalized.append([row, col])
        if tuple(obj.position) not in seen:
            raise ValueError(f"object position must be included in footprint: {obj.id}")

        obj.footprint = normalized
        self.objects[obj.id] = obj
        for row, col in normalized:
            self.map.place(row, col, obj.id)

    def remove_object(self, obj) -> None:
        """从对象表和全部占地格中移除指定对象。"""

        for row, col in getattr(obj, "footprint", [obj.position]):
            if self.map.get_e(row, col) == obj.id:
                self.map.remove(row, col)
        self.objects.pop(obj.id, None)

    def find_empty_cell_around_object(self, obj) -> list[int] | None:
        """按入口距离从对象完整占地外围寻找空格。"""

        footprint = {tuple(cell) for cell in getattr(obj, "footprint", [obj.position])}
        reference = tuple(getattr(obj, "entrance", None) or obj.position)
        candidates: set[tuple[int, int]] = set()
        for row, col in footprint:
            for row_offset, col_offset in (
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
                (-1, -1),
                (-1, 1),
                (1, -1),
                (1, 1),
            ):
                cell = (row + row_offset, col + col_offset)
                if cell in footprint:
                    continue
                if 0 <= cell[0] < self.map.height and 0 <= cell[1] < self.map.width:
                    candidates.add(cell)
        for row, col in sorted(
            candidates,
            key=lambda cell: (
                (cell[0] - reference[0]) ** 2 + (cell[1] - reference[1]) ** 2,
                cell[0],
                cell[1],
            ),
        ):
            if self.map.is_empty(row, col):
                return [row, col]
        return None

    def _allocate_episode_id(self) -> str:
        """分配进程内全局递增的经历链标识。"""

        with self._episode_lock:
            episode_id = f"episode-{self._next_episode_id:08d}"
            self._next_episode_id += 1
            return episode_id

    def _ensure_agent_episode(self, agent) -> str:
        """保留调用方已设置的经历标识，否则为当前执行分配新标识。"""

        episode_id = str(getattr(agent, "_current_episode_id", "") or "")
        episode_tick = getattr(agent, "_current_episode_tick", None)
        if episode_id and episode_tick is None:
            # 首次接收调用方指定的经历标识时，将其绑定到当前时间步。
            agent._current_episode_tick = self.time
            return episode_id
        if episode_id and episode_tick == self.time:
            return episode_id
        episode_id = self._allocate_episode_id()
        agent._current_episode_id = episode_id
        agent._current_episode_tick = self.time
        return episode_id

    def _agent_state_snapshot(self, agent) -> dict:
        """生成动作前后均可序列化的智能体状态快照。"""

        position = getattr(agent, "position", None)
        if isinstance(position, (list, tuple)):
            position = list(position)
        return {
            "tick": int(self.time),
            "agent_id": str(getattr(agent, "id", "") or ""),
            "position": position,
            "satisfaction": dict(getattr(agent, "satisfaction", {}) or {}),
            "urgency": dict(getattr(agent, "urgency", {}) or {}),
            "need_gap": dict(getattr(agent, "need_gap", {}) or {}),
            "pressure_memory": dict(getattr(agent, "pressure_memory", {}) or {}),
            "load_saturation": dict(getattr(agent, "load_saturation", {}) or {}),
            "effective_pressure": dict(getattr(agent, "effective_pressure", {}) or {}),
            "task": str(getattr(agent, "task", "") or ""),
            "task_urgency_key": str(getattr(agent, "task_urgency_key", "") or ""),
            "emotion": str(getattr(agent, "emotion", "") or ""),
            "opinion": getattr(agent, "opinion", None),
            "sleeping": bool(getattr(agent, "sleeping", False)),
            "inside_building_id": getattr(agent, "inside_building_id", None),
            "followers": list(getattr(agent, "followers", []) or []),
            "online_trust": dict(getattr(agent, "online_trust", {}) or {}),
        }

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
        set_log_tick(self.time)
        self.movements = []
        self.map.print_map()
        agents = list(self.agents.values())
        if self.platform is not None:
            self.platform.time = self.time
        for agent in agents:
            # 每个 tick 只记录本轮新产生的动作和社交动作，避免历史导出串用上一轮结果。
            agent._current_episode_id = self._allocate_episode_id()
            agent._current_episode_tick = self.time
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
                reward = await self.aexecute(agent, action)
                agent.append_trajectory(obs, action, reward, action_result=dict(agent.last_action))
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
            await self._aconversation_phase(agents)
        conversation_elapsed = time.perf_counter() - phase_start

        phase_start = time.perf_counter()
        await self._assess_agents(agents)
        assessment_elapsed = time.perf_counter() - phase_start

        phase_start = time.perf_counter()
        await self._compact_short_term_memories(agents)
        short_term_compaction_elapsed = time.perf_counter() - phase_start

        phase_start = time.perf_counter()
        self._maintain_agent_memories()
        memory_maintenance_elapsed = time.perf_counter() - phase_start

        phase_start = time.perf_counter()
        if self.history_recorder:
            self.history_recorder.record(self.time, list(self.agents.values()), platform=self.platform)
        history_elapsed = time.perf_counter() - phase_start

        logger.info(
            "[Perf] tick=%d world_total=%.3fs news=%.3fs building=%.3fs agent_tick=%.3fs conversation=%.3fs assessment=%.3fs short_memory=%.3fs history=%.3fs memory_maintenance=%.3fs agents=%d",
            self.time,
            time.perf_counter() - step_start,
            news_elapsed,
            building_elapsed,
            agent_tick_elapsed,
            conversation_elapsed,
            assessment_elapsed,
            short_term_compaction_elapsed,
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

    async def _aconversation_phase(self, agents):
        """真实运行主链使用异步对话阶段。"""

        await self.conversation_manager.arun_phase(
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
        news_post = self.platform.inject_news(
            self.time,
            news["title"],
            news["content"],
            topic=topic,
            opinion_index=news.get("opinion_index", OPINION_NEUTRAL),
        )
        notification = f"[系统新闻][{topic or '未指定主题'}] {news['title']}：{news['content']}"
        details = {
            "title": news["title"],
            "content": news_post.content,
            "topic": topic,
            "opinion_index": news_post.opinion_index,
            "is_news": True,
            "is_rumor": False,
            "source_type": news_post.source_type,
        }
        for agent in self.agents.values():
            if topic:
                agent.current_focus = topic
            episode_id = self._ensure_agent_episode(agent)
            platform_event_id = str(getattr(news_post, "platform_event_id", "") or "")
            agent._pending_social_notifications.append(
                {
                    "schema_version": 1,
                    "type": "notification",
                    "content": notification,
                    "time": int(self.time),
                    "episode_id": episode_id,
                    "event_id": platform_event_id,
                    "event_type": "official_news_created",
                    "platform_event_id": platform_event_id,
                    "feed_request_id": "",
                    "actor_id": "system",
                    "related_agent_id": "system",
                    "post_id": news_post.id,
                    "comment_id": None,
                    "parent_comment_id": None,
                    "root_comment_id": None,
                    "target_agent_id": agent.id,
                    "source_post_id": None,
                    "root_post_id": news_post.root_post_id,
                    "source_author_id": "system",
                    "topic": topic,
                    "details": dict(details),
                }
            )

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
        """先并发完成心理评测，再统一启动全部智能体的观念评测。"""

        async def _assess_psychology(agent):
            try:
                apply_passive_need_decay(agent, self.time)
                if (
                    self.psychological_assessor
                    and getattr(agent.config, "psychological_assessment_enabled", True)
                ):
                    self.psychological_assessor.observe_agent_tick(agent, self.time)
                    assessment = await self.psychological_assessor.amaybe_assess(agent, self.time)
                    mem = getattr(agent, "mem", None)
                    if (
                        isinstance(assessment, dict)
                        and mem is not None
                        and hasattr(mem, "store_psychological_assessment")
                    ):
                        mem.store_psychological_assessment(
                            agent.id,
                            assessment,
                            episode_id=str(getattr(agent, "_current_episode_id", "") or ""),
                        )
            except Exception as exc:
                logger.warning("[World] psychology assessment failed for %s at tick=%d: %s", agent.id, self.time, exc, exc_info=True)

        await asyncio.gather(*[_assess_psychology(agent) for agent in agents])
        if self.opinion_assessor:
            # 运行中只执行 honest belief + FLAN 评测；投票在模拟结束后统一执行。
            await self.opinion_assessor.aassess_all(agents, self.time)

    async def _compact_short_term_memories(self, agents) -> None:
        """评测结束后并发压缩达到 tick 上限的短期记忆。"""

        async def _compact(agent):
            if not hasattr(agent, "acompact_short_term_memory_if_needed"):
                return False
            return await agent.acompact_short_term_memory_if_needed()

        results = await asyncio.gather(*[_compact(agent) for agent in agents], return_exceptions=True)
        for agent, result in zip(agents, results):
            if isinstance(result, Exception):
                logger.warning(
                    "[World] short-term memory compaction failed for %s at tick=%d: %s",
                    agent.id,
                    self.time,
                    result,
                )

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

    def _new_need_events(self, agent, start_index: int, episode_id: str) -> list[dict]:
        """提取本次工具执行期间新增的需求事件。"""

        event_log = list(getattr(agent, "need_event_log", []) or [])
        pending_events = consume_pending_need_events(agent)
        source_events = pending_events or event_log[max(0, start_index):]
        new_events: list[dict] = []
        for event in source_events:
            if not isinstance(event, dict):
                continue
            row = dict(event)
            row.setdefault("episode_id", episode_id)
            new_events.append(row)
        return new_events

    def _finalize_action_result(
        self,
        agent,
        decision: dict,
        action_summary: dict,
        feedback,
        reward: float | None,
        *,
        episode_id: str,
        state_before: dict,
        need_event_start: int,
    ) -> None:
        """统一补齐动作快照，并把真实执行结果写入记忆。"""

        state_after = self._agent_state_snapshot(agent)
        need_events = self._new_need_events(agent, need_event_start, episode_id)
        action_summary.update(
            {
                "episode_id": episode_id,
                "before_satisfaction": dict(state_before.get("satisfaction") or {}),
                "after_satisfaction": dict(state_after.get("satisfaction") or {}),
                "before_urgency": dict(state_before.get("urgency") or {}),
                "after_urgency": dict(state_after.get("urgency") or {}),
                "state_snapshot": state_after,
            }
        )
        agent.last_action = action_summary
        status = str(action_summary.get("execution_status") or "")
        agent.add_history(
            "action_result",
            dict(action_summary),
            action_tool=str(action_summary.get("tool") or ""),
            success=False if status in {"tool_not_found", "tool_exception"} else None,
            metadata={"episode_id": episode_id},
        )
        self._store_action_memory(
            agent,
            decision,
            feedback,
            reward,
            episode_id=episode_id,
            state_before=state_before,
            state_after=state_after,
            need_events=need_events,
        )

    async def aexecute(self, agent, action_str):
        """异步执行包含 LLM 的工具，其余工具继续走同步路径。"""

        try:
            decision = json.loads(action_str) if action_str else {}
        except json.JSONDecodeError:
            return self.execute(agent, action_str)
        data, _ = self._action_from_decision(decision)
        if data.get("tool") != "social_step":
            return self.execute(agent, action_str)
        try:
            feedback = await agent.asocial_step()
        except Exception as exc:
            logger.error("[World] 工具 social_step 执行失败 (agent=%s): %s", agent.id, exc, exc_info=True)
            feedback = "tool execution failed"
        return self.execute(agent, action_str, precomputed_feedback=feedback)

    def execute(self, agent, action_str, *, precomputed_feedback=None):
        """解析 LLM 决策 JSON 中的 action 并调用对应工具。

        工具调用失败不会中断整个 tick；成功后会生成事件并通知可感知范围内的智能体。
        记忆写入放在这里，是因为只有 execute 才知道动作是否实际执行、反馈和 reward。
        """

        if not action_str:
            return
        try:
            decision = json.loads(action_str)
        except json.JSONDecodeError:
            logger.error("[World] JSON 解析失败，原始内容: %r", action_str)
            return

        episode_id = self._ensure_agent_episode(agent)
        state_before = self._agent_state_snapshot(agent)
        old_satisfaction = dict(state_before["satisfaction"])
        old_urgency = dict(state_before["urgency"])
        consume_pending_need_events(agent)
        need_event_start = len(getattr(agent, "need_event_log", []) or [])
        data, think = self._action_from_decision(decision)
        if think:
            agent.add_history("decision_think", think)
        if not data or "tool" not in data:
            logger.debug("[%s] 本轮选择不行动", agent.id)
            action_summary = {
                "tool": "",
                "args": {},
                "think": think,
                "feedback": "",
                "reward": None,
                "execution_status": "no_action",
            }
            self._finalize_action_result(
                agent,
                decision,
                action_summary,
                "",
                None,
                episode_id=episode_id,
                state_before=state_before,
                need_event_start=need_event_start,
            )
            return

        tool = self.tools.get(data["tool"])
        if not tool:
            logger.warning("[World] 未知工具: %s", data.get("tool"))
            action_summary = {
                "tool": data.get("tool", ""),
                "args": data.get("args", {}),
                "think": think,
                "feedback": "tool not found",
                "reward": None,
                "execution_status": "tool_not_found",
            }
            self._finalize_action_result(
                agent,
                decision,
                action_summary,
                "tool not found",
                None,
                episode_id=episode_id,
                state_before=state_before,
                need_event_start=need_event_start,
            )
            return

        with self._world_lock:
            args = data.get("args", {})
            args["operator_ID"] = agent.id
            try:
                # 异步入口已完成 social_step 时，不再重复调用同步工具。
                if data["tool"] == "social_step" and precomputed_feedback is not None:
                    feedback = precomputed_feedback
                else:
                    feedback = tool.run(**args)
            except Exception as e:
                logger.error("[World] 工具 %s 执行失败 (agent=%s): %s", data["tool"], agent.id, e, exc_info=True)
                action_summary = {
                    "tool": data["tool"],
                    "args": args,
                    "think": think,
                    "feedback": "tool execution failed",
                    "reward": None,
                    "execution_status": "tool_exception",
                }
                self._finalize_action_result(
                    agent,
                    decision,
                    action_summary,
                    "tool execution failed",
                    None,
                    episode_id=episode_id,
                    state_before=state_before,
                    need_event_start=need_event_start,
                )
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
            action_summary = {
                "tool": data["tool"],
                "args": args,
                "think": think,
                "feedback": feedback,
                "reward": reward,
                "execution_status": "tool_returned",
            }
            self._finalize_action_result(
                agent,
                decision,
                action_summary,
                feedback,
                reward,
                episode_id=episode_id,
                state_before=state_before,
                need_event_start=need_event_start,
            )
            return reward

    def _store_action_memory(
        self,
        agent,
        decision: dict,
        feedback,
        reward: float | None,
        *,
        episode_id: str,
        state_before: dict,
        state_after: dict,
        need_events: list[dict],
    ) -> None:
        """把已执行或失败的动作结果写入记忆系统。"""

        mem = getattr(agent, "mem", None)
        if mem is None or not hasattr(mem, "store_action_result"):
            return
        decision_payload = dict(decision)
        decision_payload.update(
            {
                "episode_id": episode_id,
                "before_satisfaction": dict(state_before.get("satisfaction") or {}),
                "after_satisfaction": dict(state_after.get("satisfaction") or {}),
                "before_urgency": dict(state_before.get("urgency") or {}),
                "after_urgency": dict(state_after.get("urgency") or {}),
                "state_snapshot": state_after,
            }
        )
        mem.store_action_result(
            agent.id,
            decision=decision_payload,
            feedback=feedback,
            reward=reward,
            world_time=self.time,
            episode_id=episode_id,
            state_before=state_before,
            state_after=state_after,
            need_events=need_events,
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
