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

        self.need: dict[str, float] = {"satiety": 0.0, "relax": 0.0}
        self.demand: dict[str, float] = {"satiety": 1, "relax": 1}
        self.demand_threshold: dict[str, float] = {
            "satiety": config.satiety_threshold,
            "relax": config.relax_threshold
        }
        self.state: dict = {}
        self.task: str = "none"
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

        self.opinion: float = config.initial_opinion
        self.online_trust: dict[str, float] = {}   # {agent_id: τ}
        self.offline_trust: dict[str, float] = {}  # {agent_id: T}
        self._last_seen_posts: list = []            # posts seen during last social_step

        self.current_focus: str = ""
        self.stuck_ticks: int = 0
        self._prev_task_need: float | None = None

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
            self.id, obs, self.task, self.demand, self.demand_threshold,
            n_results=self.config.memory_top_k,
        )

    def remember(self, info: str, **metadata) -> None:
        self.mem.store_agent_memory(self.id, info, **metadata)

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
    # Demand & reward
    # ------------------------------------------------------------------

    def get_position(self) -> list[int]:
        return self.position

    def update_demand(self, demand_key: str, demand_delta: float) -> None:
        if demand_key in self.demand:
            self.demand[demand_key] = max(0.0, self.demand[demand_key] + demand_delta)

    def update_need(self, need_key: str, need_delta: float) -> None:
        if need_key in self.need:
            self.need[need_key] = max(0.0, min(1.0, self.need[need_key] + need_delta))

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

    def set_task(self, task: str) -> None:
        from persona.reflect.reflect import _TASK_DEMAND_MAP, _TASK_INITIAL_FOCUS
        valid = {"none"} | set(_TASK_DEMAND_MAP.keys())
        if task in valid:
            self.task = task
            self.current_focus = _TASK_INITIAL_FOCUS.get(task, "")
            self.stuck_ticks = 0
            self._prev_task_need = None
        else:
            logger.warning("无效任务: %s，合法任务为 %s", task, valid)

    def update_emotion(self, new_emotion: str) -> None:
        self.emotion = new_emotion

    def get_reflect(self) -> None:
        self.reflect.step(self)
