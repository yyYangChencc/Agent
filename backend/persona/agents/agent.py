from __future__ import annotations
import math
from typing import TYPE_CHECKING
from persona.logger import get_logger
from persona.opinion.scale import clamp_opinion

if TYPE_CHECKING:
    from world.world import World
    from persona.agents.policy import Policy
    from persona.agent_memory.mem import MultiAgentMemoryManager
    from persona.reflect.reflect import Reflect
    from persona.config import AgentConfig

logger = get_logger(__name__)



class Agent:
    """单个智能体的运行时状态。

    Agent 保存可变状态：需求满足度、急迫度、压力记忆、心理评测结果、
    观念评测结果、记忆接口、社交状态和睡眠状态。真正的 LLM 决策由 policy
    完成，Agent 负责把上下文交给 policy，并在动作后更新自身状态。
    """

    def __init__(
        self,
        agent_id: str,
        position: list[int],
        world: "World",
        policy: "Policy",
        mem: "MultiAgentMemoryManager",
        reflect: "Reflect",
        platform,
        social_policy: "Policy",
        config: "AgentConfig",
        speaking_style: str = "",
        salary: float = 0.0,
    ):
        self.id = agent_id
        self.position = position
        self.world = world
        self.policy = policy
        self.reflect = reflect
        self.platform = platform
        self.social_policy = social_policy
        self.next_action: str | None = None
        self.history: list[str] = []
        self.trajectory_buffer: list[dict] = []  # 当前任务内的 obs/action/reward 轨迹

        # satisfaction 是客观满足度；urgency 是主观急迫度；pressure_* 是需求缺口
        # 的累积压力层，供心理评测器判断是否激活。
        self.satisfaction: dict[str, float] = {"satiety": 0.0, "relax": 0.0, "money": 0.0}
        self.urgency: dict[str, float] = {"satiety": 1.0, "relax": 1.0, "money": 1.0}
        self.satisfaction_threshold: dict[str, float] = {
            "satiety": config.satiety_threshold,
            "relax": config.relax_threshold,
            "money": config.money_threshold,
        }
        self.need_gap: dict[str, float] = {k: 0.0 for k in self.satisfaction}
        self.pressure_memory: dict[str, float] = {k: 0.0 for k in self.satisfaction}
        self.load_saturation: dict[str, float] = {k: 0.0 for k in self.satisfaction}
        self.effective_pressure: dict[str, float] = {k: 0.0 for k in self.satisfaction}
        self.psychological_assessment_window = None
        self.last_need_assessments: dict[str, dict] = {}
        self.last_psychological_assessment: dict | None = None
        self.state: dict = {}
        self.task: str = "none"
        self.task_urgency_key: str = ""  # 当前任务对应的 satisfaction/urgency 键，由 set_task() 设置
        self.observed_events: list = []
        self.mem = mem
        self.inbox: list[dict] = []
        self.conversation_opted_out: bool = False

        self.post_history: list = []
        self.followers: list[str] = []
        self.config = config
        self.update_need_pressure(accumulate=False)
        self.update_urgency_from_satisfaction()

        self.speaking_style: str = speaking_style
        self.emotion: str = "平静"
        # 工资：公司自动交互时每次获得的 money satisfaction 增量（>=0）
        self.salary: float = salary

        self.opinion: float = config.initial_opinion
        self.opinion_scores: dict[str, float] = {}
        self.last_opinion_assessment: dict | None = None
        self.opinion_assessment_history: list[dict] = []
        self.online_trust: dict[str, float] = {}   # 线上信任，主要由点赞/点踩调整
        self.offline_trust: dict[str, float] = {}  # 线下信任，保留给后续线下关系建模
        self._last_seen_posts: list = []            # 上一次 social_step 中可见的帖子

        self.current_focus: str = ""
        self.stuck_ticks: int = 0
        self._prev_task_satisfaction: float | None = None

        self.sleeping: bool = False
        self.sleep_ticks_remaining: int = 0
        self.sleeping_on_bed_id: str | None = None
        self._sleep_start_satisfaction: dict[str, float] = {}
        self._sleep_start_time: int = 0
        self.inside_building_id: str | None = None
        self._pending_social_notifications: list[str] = []  # 待推送的社交通知

        self.world.add_agent(self)

    # ------------------------------------------------------------------
    # Core decision cycle
    # ------------------------------------------------------------------

    def step(self, observation: str) -> str:
        self.add_history("observation", observation)
        mem_info = self.recall(observation)
        action = self.policy.decide(self, observation, mem_info)
        self.add_history("action", action)
        return action

    async def astep(self, observation: str) -> str:
        self.add_history("observation", observation)
        mem_info = await self.arecall(observation)
        action = await self.policy.adecide(self, observation, mem_info)
        self.add_history("action", action)
        return action

    # ------------------------------------------------------------------
    # Social platform
    # ------------------------------------------------------------------

    def social_step(self) -> str | None:
        if self.platform is None or self.social_policy is None:
            return None
        logger.info("[%s] 正在查看帖子...", self.id)
        posts, posts_info = self._receive_post()
        self._last_seen_posts = posts
        logger.debug("[%s] 收到帖子内容: %s", self.id, posts_info)
        mem_info = self.recall(posts_info, context="social")
        raw = self.social_policy.decide(self, posts_info, mem_info)
        feedback = self.platform.execute(self.id, raw)
        logger.debug("[%s] 社交平台反馈: %s", self.id, feedback)
        return feedback

    def _receive_post(self) -> tuple[list, str]:
        posts = self.platform.give_post(self.id)
        if not posts:
            logger.info("[%s] 没有收到任何帖子", self.id)
            return [], f"{self.id}暂时没有收到任何帖子"
        visible_ids = "、".join(str(post.id) for post in posts)
        posts_text = "\n".join([post.show() for post in posts])
        return posts, f"当前可互动帖子ID列表：{visible_ids}\n{posts_text}"

    def get_post_history(self) -> str:
        return "\n".join([post.show() for post in self.post_history])

    def add_follower(self, agent_id: str) -> None:
        if agent_id not in self.followers:
            self.followers.append(agent_id)

    def add_post_history(self, post) -> None:
        self.post_history.append(post)

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    def _memory_top_k_for(self, context: str) -> int:
        """按场景调整记忆检索数量。

        卡住或有当前关注主题时增加检索量；社交和对话场景减少检索量，
        避免短 prompt 被过多长期记忆淹没。
        """

        top_k = self.config.memory_top_k
        if self.stuck_ticks > 0 or self.current_focus:
            top_k += self.config.memory_focus_bonus_k
        if context == "social":
            top_k = max(2, top_k - 1)
        if context == "conversation":
            top_k = max(2, top_k - 2)
        return min(top_k, self.config.memory_max_top_k)

    def recall(self, obs: str, context: str = "world") -> list[str]:
        return self.mem.smart_retrieve(
            self.id, obs, self.task, self.urgency, self.satisfaction_threshold,
            n_results=self._memory_top_k_for(context),
            context=context,
        )

    async def arecall(self, obs: str, context: str = "world") -> list[str]:
        return await self.mem.asmart_retrieve(
            self.id, obs, self.task, self.urgency, self.satisfaction_threshold,
            n_results=self._memory_top_k_for(context),
            context=context,
        )

    def remember(self, info: str, **metadata) -> None:
        self.mem.store_agent_memory(self.id, info, world_time=self.world.time, **metadata)

    async def aremember(self, info: str, **metadata) -> None:
        await self.mem.astore_agent_memory(self.id, info, world_time=self.world.time, **metadata)

    def append_trajectory(self, obs: str, action: str, reward: float | None) -> None:
        """把当前任务中的一步执行结果暂存，等任务结束后再总结入长期记忆。"""

        self.trajectory_buffer.append({
            "step": self.world.time,
            "obs": obs,
            "action": action,
            "reward": reward,
        })

    def flush_trajectory(self, task: str, need_key: str = "") -> None:
        """同步总结当前任务轨迹并写入情景记忆。"""

        if not self.trajectory_buffer:
            return
        lines = [f"任务：{task}"]
        for i, entry in enumerate(self.trajectory_buffer, start=1):
            reward_str = f"{entry['reward']:.3f}" if entry["reward"] is not None else "N/A"
            lines.append(
                f"步骤{i} (t={entry['step']})\n"
                f"  obs:    {entry['obs']}\n"
                f"  action: {entry['action']}\n"
                f"  reward: {reward_str}"
            )
        trajectory_text = "\n".join(lines)
        system, user = self.reflect.prompt.trajectory_summary(trajectory_text, task)
        summary = self.reflect.llm.generate(system, user)
        logger.debug("[%s] 轨迹总结: %s", self.id, summary)
        self.remember(
            summary,
            memory_type="episodic",
            task=task,
            need_key=need_key,
            outcome="completed",
            importance=0.7,
            confidence=0.7,
        )
        logger.debug("[%s] 轨迹总结已存储，共 %d 步", self.id, len(self.trajectory_buffer))
        self.trajectory_buffer.clear()

    async def aflush_trajectory(self, task: str, need_key: str = "") -> None:
        """异步总结当前任务轨迹并写入情景记忆。"""

        if not self.trajectory_buffer:
            return
        lines = [f"任务：{task}"]
        for i, entry in enumerate(self.trajectory_buffer, start=1):
            reward_str = f"{entry['reward']:.3f}" if entry["reward"] is not None else "N/A"
            lines.append(
                f"步骤{i} (t={entry['step']})\n"
                f"  obs:    {entry['obs']}\n"
                f"  action: {entry['action']}\n"
                f"  reward: {reward_str}"
            )
        trajectory_text = "\n".join(lines)
        system, user = self.reflect.prompt.trajectory_summary(trajectory_text, task)
        summary = await self.reflect.llm.agenerate(system, user)
        logger.debug("[%s] 轨迹总结: %s", self.id, summary)
        await self.aremember(
            summary,
            memory_type="episodic",
            task=task,
            need_key=need_key,
            outcome="completed",
            importance=0.7,
            confidence=0.7,
        )
        logger.debug("[%s] 轨迹总结已存储，共 %d 步", self.id, len(self.trajectory_buffer))
        self.trajectory_buffer.clear()

    # ------------------------------------------------------------------
    # Perception & messaging
    # ------------------------------------------------------------------

    def perceive(self, event) -> None:
        self.observed_events.append(event)

    def can_perceive(self, event, radius: int | None = None) -> bool:
        if event.position is None:
            return False
        r = radius if radius is not None else self.config.observation_radius
        from persona.utils.distance import manhattan
        return manhattan(self.position, event.position) <= r

    def receive_message(
        self,
        sender_id: str,
        content: str,
        response_to: str | None = None,
        session_id: str | None = None,
        intent: str | None = None,
        target: str | None = None,
    ) -> None:
        """接收线下对话消息。

        opt-out 后不再接收本 tick 对话，避免智能体已经退出对话后继续被拉回。
        """

        if self.conversation_opted_out:
            return
        self.inbox.append({
            "sender": sender_id,
            "content": content,
            "response_to": response_to,
            "time": self.world.time,
            "session_id": session_id,
            "intent": intent,
            "target": target,
        })

    def conversation_step(
        self,
        policy: "Policy",
        round_n: int,
        max_rounds: int,
        conv_history: list[dict] | None = None,
    ) -> str | None:
        """把 inbox 转为一次对话观察，并交给对话 policy 决定回复。"""

        if not self.inbox:
            return None
        remaining = max_rounds - round_n

        if conv_history:
            lines = []
            for entry in conv_history:
                session_text = f" session={entry.get('session_id')}" if entry.get("session_id") else ""
                intent_text = f" intent={entry.get('intent')}" if entry.get("intent") else ""
                line = (
                    f"  [第{entry['round']}轮{session_text}{intent_text}] "
                    f"{entry['sender']} → {entry['target']}: {entry['content']}"
                )
                if entry.get("response_to"):
                    line += f"（回复: {entry['response_to']}）"
                lines.append(line)
            history_block = "[本时间步对话历史]\n" + "\n".join(lines)
        else:
            history_block = "[本时间步对话历史]\n（暂无）"

        msgs = "\n".join([
            (
                f"{m['sender']} 对你说"
                f"{'（session=' + str(m.get('session_id')) + '）' if m.get('session_id') else ''}"
                f"{'（intent=' + str(m.get('intent')) + '）' if m.get('intent') else ''}: "
                f"{m['content']}"
            ) +
            (f"（回复的是: {m['response_to']}）" if m.get("response_to") else "")
            for m in self.inbox
        ])
        observation = (
            f"{history_block}\n\n"
            f"[对话消息 · 第 {round_n} 轮，还剩 {remaining} 轮]\n"
            f"{msgs}"
        )
        self.inbox.clear()
        self.add_history("conversation", observation)
        mem_info = self.recall(observation, context="conversation")
        action = policy.decide(self, observation, mem_info)
        if action:
            self.add_history("conversation_reply", action)
        else:
            self.conversation_opted_out = True
            logger.info("[%s] 选择结束对话（轮次 %d）", self.id, round_n)
        return action

    # ------------------------------------------------------------------
    # Urgency & reward
    # ------------------------------------------------------------------

    def get_position(self) -> list[int]:
        return self.position

    def update_urgency(self, urgency_key: str, urgency_delta: float) -> None:
        if urgency_key in self.urgency:
            self.urgency[urgency_key] = max(0.0, min(1.0, self.urgency[urgency_key] + urgency_delta))

    def _clamp_urgency(self, value: float, floor: float) -> float:
        return max(floor, min(1.0, value))

    def _urgency_floor(self, satisfaction_key: str) -> float:
        floor = self.config.urgency_floors.get(satisfaction_key, 0.0)
        return max(0.0, min(1.0, floor))

    def _raw_urgency_from_gap(self, satisfaction_key: str) -> float:
        floor = self._urgency_floor(satisfaction_key)
        gap_weight = self.config.urgency_gap_weights.get(satisfaction_key, 1.0)
        raw = floor + gap_weight * self._normalized_need_gap(satisfaction_key)
        return self._clamp_urgency(raw, floor)

    def _is_need_satisfied(self, satisfaction_key: str) -> bool:
        threshold = self.satisfaction_threshold.get(satisfaction_key)
        if not isinstance(threshold, (int, float)):
            return True
        return self.satisfaction.get(satisfaction_key, 0.0) > threshold

    def _is_layer_satisfied(self, layer_keys: list[str]) -> bool:
        return all(self._is_need_satisfied(key) for key in layer_keys)

    def update_urgency_from_satisfaction(self) -> None:
        """根据满足度缺口重新计算急迫度，并应用需求层级封顶规则。

        当前生理层由 satiety/relax 组成；当底层未满足时，高层需求的急迫度
        不能超过底层需求形成的 cap。
        """

        raw_urgency = {
            key: self._raw_urgency_from_gap(key)
            for key in self.urgency
        }
        final_urgency = dict(raw_urgency)

        active_cap: float | None = None
        for layer_name in self.config.need_layer_order:
            layer_keys = [
                key
                for key in self.config.need_layers.get(layer_name, [])
                if key in final_urgency
            ]
            if not layer_keys:
                continue

            if active_cap is not None:
                for key in layer_keys:
                    floor = self._urgency_floor(key)
                    final_urgency[key] = self._clamp_urgency(
                        min(final_urgency[key], active_cap),
                        floor,
                    )

            if active_cap is None and not self._is_layer_satisfied(layer_keys):
                active_cap = max(final_urgency[key] for key in layer_keys)

        for key, urgency in final_urgency.items():
            self.urgency[key] = self._clamp_urgency(urgency, self._urgency_floor(key))

    def update_satisfaction(self, satisfaction_key: str, satisfaction_delta: float) -> None:
        if satisfaction_key in self.satisfaction:
            new_val = self.satisfaction[satisfaction_key] + satisfaction_delta
            # money 无上限（累计金额）；satiety / relax 范围 [0, 100]
            if satisfaction_key == "money":
                self.satisfaction[satisfaction_key] = max(0.0, new_val)
            else:
                self.satisfaction[satisfaction_key] = max(0.0, min(100.0, new_val))
            self.update_need_pressure(accumulate=False)
            self.update_urgency_from_satisfaction()

    def _normalized_need_gap(self, satisfaction_key: str) -> float:
        threshold = self.satisfaction_threshold.get(satisfaction_key)
        if not isinstance(threshold, (int, float)) or threshold <= 0:
            return 0.0
        satisfaction = self.satisfaction.get(satisfaction_key, 0.0)
        return max(0.0, (threshold - satisfaction) / threshold)

    def update_need_pressure(self, accumulate: bool = True) -> None:
        """刷新需求压力层。

        need_gap 表示当前缺口；pressure_memory 表示缺口的时间累积；
        load_saturation 将累积压力压缩到 [0, 1]；effective_pressure 综合瞬时缺口、
        残余负荷和二者交互，是心理评测的激活依据。
        """

        dt = max(0.0, self.config.need_pressure_dt)
        for key in self.satisfaction:
            gap = self._normalized_need_gap(key)
            self.need_gap[key] = gap

            memory = self.pressure_memory.get(key, 0.0)
            if accumulate and dt > 0:
                if gap > 0:
                    memory += gap * dt
                else:
                    recovery_rate = self.config.pressure_recovery_rates.get(
                        key,
                        self.config.pressure_default_recovery_rate,
                    )
                    memory *= max(0.0, 1.0 - recovery_rate * dt)
            memory = max(0.0, memory)
            self.pressure_memory[key] = memory

            kappa = self.config.pressure_load_kappas.get(
                key,
                self.config.pressure_default_load_kappa,
            )
            # 指数饱和避免 pressure_memory 无限增长后直接支配心理评测。
            load = 0.0 if kappa <= 0 else 1.0 - math.exp(-memory / kappa)
            load = max(0.0, min(1.0, load))
            self.load_saturation[key] = load

            current_weight = self.config.pressure_current_weights.get(
                key,
                self.config.pressure_default_current_weight,
            )
            residual_weight = self.config.pressure_residual_weights.get(
                key,
                self.config.pressure_default_residual_weight,
            )
            amplification_weight = self.config.pressure_amplification_weights.get(
                key,
                self.config.pressure_default_amplification_weight,
            )
            pressure = (
                current_weight * gap
                + residual_weight * load
                + amplification_weight * gap * load
            )
            self.effective_pressure[key] = max(0.0, pressure)

    def update_opinion(self, delta: float) -> None:
        self.opinion = clamp_opinion(self.opinion + delta)


    # ------------------------------------------------------------------
    # History & task
    # ------------------------------------------------------------------

    def add_history(self, type_: str, record: str) -> None:
        self.history.append(f"{type_}: {record}")
        if len(self.history) > self.config.max_history:
            self.history = self.history[-self.config.max_history:]

    def set_next_action(self, action: str) -> None:
        self.next_action = action

    def get_next_action(self) -> str | None:
        res = self.next_action
        self.next_action = None
        return res

    def set_task(self, task: str, urgency_key: str = "") -> None:
        """设置当前任务及其对应的 satisfaction/urgency 键。
        task="none" 时 urgency_key 忽略；其余任务应传入合法的 satisfaction 键（satiety/relax/money）。
        """
        valid_keys = set(self.satisfaction.keys()) | {""}
        if task != "none" and urgency_key not in valid_keys:
            logger.warning("[%s] 无效 urgency_key: %s，合法值为 %s", self.id, urgency_key, valid_keys)
            return
        self.task = task
        self.task_urgency_key = urgency_key if task != "none" else ""
        self.current_focus = ""
        self.stuck_ticks = 0
        self._prev_task_satisfaction = None

    def update_emotion(self, new_emotion: str) -> None:
        self.emotion = new_emotion

    def tick_satisfaction(self) -> None:
        """普通清醒状态下的每 tick 需求变化。"""

        if self.sleeping:
            return
        self.update_satisfaction("satiety", -self.config.satiety_decay_rate)
        self.update_satisfaction("relax", -self.config.relax_decay_rate)
        if self.task == "none":
            self.update_satisfaction("relax", self.config.relax_increase_rate)
        self.update_need_pressure()
        self.update_urgency_from_satisfaction()

    def tick_sleep_recovery(self) -> None:
        """睡眠期间的每 tick 恢复。

        睡眠不走普通需求衰减，也不累积新的 pressure_memory；
        update_satisfaction 内部只刷新当前压力和急迫度。
        """

        if not self.sleeping:
            return
        sleep_time = max(1, self.config.sleep_time)
        self.update_satisfaction("relax", self.config.sleep_relax_recover / sleep_time)

    def wakeup(self, bed) -> None:
        """结束睡眠并把智能体从床位置移回周围空格。"""

        elapsed = self.world.time - self._sleep_start_time
        changes = {
            k: round(self.satisfaction.get(k, 0) - self._sleep_start_satisfaction.get(k, 0), 2)
            for k in self.satisfaction
        }
        change_str = "，".join(f"{k} {'+' if v >= 0 else ''}{v}" for k, v in changes.items())
        self.sleeping = False
        self.sleep_ticks_remaining = 0
        self.sleeping_on_bed_id = None
        self.task = "none"
        if bed is not None:
            bed.exit_bed(self)
        # 在床周围找空格放置智能体
        bx, by = self.position  # 当前位置 = 床的位置
        placed = False
        with self.world._world_lock:
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                nx, ny = bx + dx, by + dy
                if 0 <= nx < self.world.map.height and 0 <= ny < self.world.map.width:
                    if self.world.map.is_empty(nx, ny):
                        self.world.map.place(nx, ny, self.id)
                        self.position = [nx, ny]
                        placed = True
                        break
            if not placed:
                if self.world.map.is_empty(bx, by):
                    self.world.map.place(bx, by, self.id)
                else:
                    logger.warning("[%s] 睡眠结束但床周围无空位，暂留当前位置 %s", self.id, self.position)
        self.add_history("sleep_summary", f"睡眠结束，共经过 {elapsed} 步，期间需求变化：{change_str}")
        logger.info("[%s] 睡眠结束，经过 %d 步，需求变化 %s", self.id, elapsed, changes)

    def get_reflect(self) -> None:
        self.reflect.step(self)

    async def aget_reflect(self) -> None:
        await self.reflect.astep(self)

    def sleep_status(self, bed_id: str) -> None:
        """进入睡眠状态，并记录睡眠开始时的需求快照。"""

        self.sleep_ticks_remaining = max(1, self.config.sleep_time)
        self.sleeping = True
        self.sleeping_on_bed_id = bed_id
        self._sleep_start_satisfaction = dict(self.satisfaction)
        self._sleep_start_time = self.world.time
        logger.info("[%s] 开始睡觉，预计睡眠 %d tick", self.id, self.config.sleep_time)
    
