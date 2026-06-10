from __future__ import annotations
from typing import TYPE_CHECKING
from persona.logger import get_logger

if TYPE_CHECKING:
    from world.world import World
    from persona.agents.policy import Policy
    from persona.agent_memory.mem import MultiAgentMemoryManager
    from persona.reflect.reflect import Reflect
    from persona.config import AgentConfig

logger = get_logger(__name__)



class Agent:
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
        role: str = "",
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
        self.trajectory_buffer: list[dict] = []  # (obs, action, reward) per step within current task

        self.satisfaction: dict[str, float] = {"satiety": 0.0, "relax": 0.0, "money": 0.0}
        self.urgency: dict[str, float] = {"satiety": 1.0, "relax": 1.0, "money": 1.0}
        self.satisfaction_threshold: dict[str, float] = {
            "satiety": config.satiety_threshold,
            "relax": config.relax_threshold,
            "money": config.money_threshold,
        }
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

        self.role: str = role
        self.speaking_style: str = speaking_style
        self.emotion: str = "平静"
        # 工资：公司自动交互时每次获得的 money satisfaction 增量（>=0）
        self.salary: float = salary

        self.opinion: float = config.initial_opinion
        self.online_trust: dict[str, float] = {}   # {agent_id: τ}
        self.offline_trust: dict[str, float] = {}  # {agent_id: T}
        self._last_seen_posts: list = []            # posts seen during last social_step

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
        mem_info = self.recall(posts_info)
        raw = self.social_policy.decide(self, posts_info, mem_info)
        feedback = self.platform.execute(self.id, raw)
        logger.debug("[%s] 社交平台反馈: %s", self.id, feedback)
        return feedback

    def _receive_post(self) -> tuple[list, str]:
        posts = self.platform.give_post(self.id)
        if not posts:
            logger.info("[%s] 没有收到任何帖子", self.id)
            return [], f"{self.id}暂时没有收到任何帖子"
        return posts, "\n".join([post.show() for post in posts])

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

    def recall(self, obs: str) -> list[str]:
        return self.mem.smart_retrieve(
            self.id, obs, self.task, self.urgency, self.satisfaction_threshold,
            n_results=self.config.memory_top_k,
        )

    async def arecall(self, obs: str) -> list[str]:
        return await self.mem.asmart_retrieve(
            self.id, obs, self.task, self.urgency, self.satisfaction_threshold,
            n_results=self.config.memory_top_k,
        )

    def remember(self, info: str, **metadata) -> None:
        self.mem.store_agent_memory(self.id, info, world_time=self.world.time, **metadata)

    async def aremember(self, info: str, **metadata) -> None:
        await self.mem.astore_agent_memory(self.id, info, world_time=self.world.time, **metadata)

    def append_trajectory(self, obs: str, action: str, reward: float | None) -> None:
        self.trajectory_buffer.append({
            "step": self.world.time,
            "obs": obs,
            "action": action,
            "reward": reward,
        })

    def flush_trajectory(self, task: str) -> None:
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
        self.remember(summary, type="trajectory", task=task)
        logger.debug("[%s] 轨迹总结已存储，共 %d 步", self.id, len(self.trajectory_buffer))
        self.trajectory_buffer.clear()

    async def aflush_trajectory(self, task: str) -> None:
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
        await self.aremember(summary, type="trajectory", task=task)
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
        self, sender_id: str, content: str, response_to: str | None = None
    ) -> None:
        if self.conversation_opted_out:
            return
        self.inbox.append({
            "sender": sender_id,
            "content": content,
            "response_to": response_to,
            "time": self.world.time,
        })

    def conversation_step(
        self,
        policy: "Policy",
        round_n: int,
        max_rounds: int,
        conv_history: list[dict] | None = None,
    ) -> str | None:
        if not self.inbox:
            return None
        remaining = max_rounds - round_n

        if conv_history:
            lines = []
            for entry in conv_history:
                line = f"  [第{entry['round']}轮] {entry['sender']} → {entry['target']}: {entry['content']}"
                if entry.get("response_to"):
                    line += f"（回复: {entry['response_to']}）"
                lines.append(line)
            history_block = "[本时间步对话历史]\n" + "\n".join(lines)
        else:
            history_block = "[本时间步对话历史]\n（暂无）"

        msgs = "\n".join([
            f"{m['sender']} 对你说: {m['content']}" +
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
        mem_info = self.recall(observation)
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
            self.urgency[urgency_key] = max(0.0, self.urgency[urgency_key] + urgency_delta)

    def update_satisfaction(self, satisfaction_key: str, satisfaction_delta: float) -> None:
        if satisfaction_key in self.satisfaction:
            new_val = self.satisfaction[satisfaction_key] + satisfaction_delta
            # money 无上限（累计金额）；satiety / relax 范围 [0, 100]
            if satisfaction_key == "money":
                self.satisfaction[satisfaction_key] = max(0.0, new_val)
            else:
                self.satisfaction[satisfaction_key] = max(0.0, min(100.0, new_val))

    def update_opinion(self, delta: float) -> None:
        self.opinion = max(0.0, min(1.0, self.opinion + delta))


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
        if self.sleeping:
            return
        self.update_satisfaction("satiety", -self.config.satiety_decay_rate)
        self.update_satisfaction("relax", -self.config.relax_decay_rate)
        if self.task == "none":
            self.update_satisfaction("relax", self.config.relax_increase_rate)

    def wakeup(self, bed) -> None:
        elapsed = self.world.time - self._sleep_start_time
        changes = {
            k: round(self.satisfaction.get(k, 0) - self._sleep_start_satisfaction.get(k, 0), 2)
            for k in self.satisfaction
        }
        change_str = "，".join(f"{k} {'+' if v >= 0 else ''}{v}" for k, v in changes.items())
        self.sleeping = False
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
                self.world.map.place(bx, by, self.id)
        self.add_history("sleep_summary", f"睡眠结束，共经过 {elapsed} 步，期间需求变化：{change_str}")
        logger.info("[%s] 睡眠结束，经过 %d 步，需求变化 %s", self.id, elapsed, changes)

    def get_reflect(self) -> None:
        self.reflect.step(self)

    async def aget_reflect(self) -> None:
        await self.reflect.astep(self)

    def sleep_status(self, bed_id: str) -> None:
        self.update_satisfaction("relax", self.config.sleep_relax_recover)
        self.sleep_ticks_remaining = self.config.sleep_time
        self.sleeping = True
        self.sleeping_on_bed_id = bed_id
        self._sleep_start_satisfaction = dict(self.satisfaction)
        self._sleep_start_time = self.world.time
        self.sleeping_on_bed_id = bed_id
        logger.info("[%s] 开始睡觉，预计睡眠 %d tick", self.id, self.config.sleep_time)
    
