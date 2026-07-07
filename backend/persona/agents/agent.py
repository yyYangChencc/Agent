from __future__ import annotations
import json
import math
from typing import TYPE_CHECKING
from persona.logger import get_logger
from persona.need_events import apply_need_delta
from persona.opinion.scale import clamp_opinion

if TYPE_CHECKING:
    from world.world import World
    from persona.agents.policy import Policy
    from persona.agent_memory.mem import MultiAgentMemoryManager
    from persona.reflect import Reflect
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
        memory_planner,
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
        self.memory_planner = memory_planner
        self.next_action: str | None = None
        self.history: list[str] = []
        self.trajectory_buffer: list[dict] = []  # 当前任务内的 obs/action/reward 轨迹

        # satisfaction 是客观满足度；urgency 是主观急迫度；pressure_* 是需求缺口
        # 的累积压力层，供心理评测器判断是否激活。
        self.satisfaction: dict[str, float] = {
            "satiety": 0.0,
            "relax": 0.0,
            "money": 0.0,
            "belonging": 55.0,
            "esteem": 55.0,
            "self_actualization": 55.0,
        }
        self.urgency: dict[str, float] = {key: 1.0 for key in self.satisfaction}
        self.satisfaction_threshold: dict[str, float] = {
            "satiety": config.satiety_threshold,
            "relax": config.relax_threshold,
            "money": config.money_threshold,
            "belonging": config.belonging_threshold,
            "esteem": config.esteem_threshold,
            "self_actualization": config.self_actualization_threshold,
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
        self.conversation_event_log: list[dict] = []  # 已落地的线下对话消息，供历史与评测追踪
        self.need_event_log: list[dict] = []          # 需求满足度变化事件，供实验解释链追踪
        self.visited_building_ids: set[str] = set()   # 已进入过的建筑 ID，用于自我实现探索奖励
        self.visited_building_kinds: set[str] = set() # 已进入过的建筑类型，用于避免重复奖励
        self.known_social_contacts: set[str] = set()  # 已发生过线上互动的真实智能体
        self.expressed_opinion_topics: set[str] = set() # 已围绕系统主题原创发帖的记录
        self.last_positive_belonging_tick: int = 0
        self.last_positive_esteem_tick: int = 0
        self.last_self_actualization_tick: int = 0
        self.recent_action_tools: list[str] = []       # 近期动作类型，用于自我实现被动衰减

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
        self.last_opinion_before_assessment: float = self.opinion
        self.opinion_seen_posts_buffer: list[dict] = []  # 当前观念评测周期内实际看过的当前主题帖子。
        self._last_opinion_assessment_tick: int = 0      # 记录上次观念评测时间，用于间隔兜底。
        self._last_opinion_evidence_signature: str = ""  # 已评测证据签名，避免同一证据重复触发 LLM。
        self.online_trust: dict[str, float] = {}   # 线上信任，主要由点赞/点踩调整
        self.offline_trust: dict[str, float] = {}  # 线下信任，保留给后续线下关系建模
        self._last_seen_posts: list = []            # 上一次 social_step 中可见的帖子
        self.last_action: dict = {}                 # 最近一次世界动作摘要，供实验日志记录。
        self.last_social_action: dict = {}          # 最近一次社交动作摘要，供实验日志记录。
        self.did_move_this_tick: bool = False       # 本 tick 是否真实发生移动，用于 relax 被动恢复。
        self.did_work_this_tick: bool = False       # 本 tick 是否真实发生工作，用于 relax 被动恢复。

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

    def step(self, observation: str | dict | list | None) -> str:
        """同步决策入口。

        observe 和社交平台现在返回结构化 JSON；Agent 只在送入 prompt 前转成字符串，
        原始结构仍交给记忆系统分类写入。
        """

        observation_text = self._format_structured_context(observation)
        self.add_history("observation", observation_text)
        self._store_observation_memory(observation)
        mem_info = self._planned_recall(observation, observation_text, context="world")
        action = self.policy.decide(self, observation_text, mem_info)
        self.add_history("action", action)
        return action

    async def astep(self, observation: str | dict | list | None) -> str:
        """异步决策入口，与同步路径保持相同的结构化观察处理。"""

        observation_text = self._format_structured_context(observation)
        self.add_history("observation", observation_text)
        self._store_observation_memory(observation)
        mem_info = await self._aplanned_recall(observation, observation_text, context="world")
        action = await self.policy.adecide(self, observation_text, mem_info)
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
        self._record_opinion_seen_posts(posts_info)
        # 社交浏览结果先以 JSON 写入记忆，再转成 prompt 字符串交给社交 policy。
        self._store_social_browse_memory(posts_info)
        posts_text = self._format_structured_context(posts_info)
        logger.debug("[%s] 收到帖子内容: %s", self.id, posts_text)
        mem_info = self._planned_recall(posts_info, posts_text, context="social")
        raw = self.social_policy.decide(self, posts_text, mem_info)
        feedback = self.platform.execute(self.id, raw)
        self._store_social_feedback_memory(feedback)
        logger.debug("[%s] 社交平台反馈: %s", self.id, feedback)
        return self._format_structured_context(feedback)

    def _receive_post(self) -> tuple[list, dict]:
        posts = self.platform.get_visible_posts(self.id)
        return posts, self.platform.give_post(self.id)

    def _record_opinion_seen_posts(self, posts_info: dict) -> None:
        """只记录本周期实际看过且 topic 等于当前观念主题的帖子。"""

        if not isinstance(posts_info, dict):
            return
        posts = posts_info.get("posts") if isinstance(posts_info.get("posts"), list) else []
        if not posts:
            return
        topic = str(self.config.default_opinion_topic or "")
        latest_by_id = {
            str(post.get("id")): dict(post)
            for post in self.opinion_seen_posts_buffer
            if isinstance(post, dict) and post.get("id") is not None
        }
        for post in posts:
            # 观念证据只由帖子 topic 决定，不读取正文或评论内容。
            if not isinstance(post, dict) or not self._post_matches_opinion_topic(post, topic):
                continue
            post_id = post.get("id")
            if post_id is None:
                continue
            latest_by_id[str(post_id)] = self._opinion_post_snapshot(post)
        self.opinion_seen_posts_buffer = list(latest_by_id.values())

    def _post_matches_opinion_topic(self, post: dict, topic: str) -> bool:
        """按帖子 topic 判断是否属于当前观念主题。"""

        return bool(topic) and str(post.get("topic") or "") == topic

    def _opinion_post_snapshot(self, post: dict) -> dict:
        """只保留评测所需字段，避免把无关平台状态带入观念 prompt。"""

        return {
            "id": post.get("id"),
            "author_id": post.get("author_id"),
            "topic": post.get("topic"),
            "content": post.get("content"),
            "time": post.get("time"),
            "likes": post.get("likes"),
            "dislikes": post.get("dislikes"),
            "comments_count": post.get("comments_count"),
            "opinion_index": post.get("opinion_index"),
            "is_news": post.get("is_news"),
            "is_rumor": post.get("is_rumor"),
            "source_type": post.get("source_type"),
            "comments": post.get("comments") if isinstance(post.get("comments"), list) else [],
        }

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

    def _memory_timeout_seconds(self) -> float | None:
        """读取记忆检索超时预算；非正数表示不启用超时。"""

        value = getattr(self.config, "memory_retrieval_timeout_seconds", 0.0)
        try:
            timeout = float(value)
        except (TypeError, ValueError):
            return None
        return timeout if timeout > 0 else None

    def _observation_has_direct_need_target(self, obs: str | dict | list | None, context: str) -> bool:
        """当前观察已经给出可行动目标时跳过记忆查询，避免重复召回同一最新状态。"""

        if not getattr(self.config, "memory_planner_skip_when_observation_sufficient", True):
            return False
        if context != "world" or not isinstance(obs, dict):
            return False
        if not self._is_basic_need_task():
            return False
        useful_kinds = self._useful_object_kinds_for_current_task()
        if not useful_kinds:
            return False
        for obj in obs.get("objects") or []:
            if not isinstance(obj, dict):
                continue
            if str(obj.get("kind") or "") in useful_kinds:
                return True
        return False

    def _rule_based_memory_plan(self, obs: str | dict | list | None, context: str) -> str | None:
        """低信号场景用结构化规则查询替代 LLM planner，减少无意义 planner 调用。"""

        if not isinstance(obs, dict):
            return None
        if context == "world" and self._is_low_signal_observation(obs):
            kinds = sorted(self._useful_object_kinds_for_current_task())
            if not kinds:
                return None
            plan = {
                "context": context,
                "think": "规则短路：当前无新闻、社交、人物互动或失败反馈，只查询当前任务需要的实体状态。",
                "queries": [
                    {
                        "type": "entity_state",
                        "intent": "find_need_target",
                        "entity_type": "object",
                        "kinds": kinds,
                        "exclude_visible": True,
                        "limit": 5,
                    }
                ],
            }
            return json.dumps(plan, ensure_ascii=False)
        return None

    def _is_basic_need_task(self) -> bool:
        """判断当前任务是否属于生理/安全需求。"""

        return self.task_urgency_key in {"satiety", "relax", "money"}

    def _useful_object_kinds_for_current_task(self) -> set[str]:
        """把需求键映射到可用物品/建筑 kind。"""

        if self.task_urgency_key == "satiety":
            return {"food", "food_shop"}
        if self.task_urgency_key == "relax":
            return {"bed", "playground"}
        if self.task_urgency_key == "money":
            return {"company"}
        return set()

    def _is_low_signal_observation(self, obs: dict) -> bool:
        """没有新闻、社交通知、人物互动或失败反馈时，不需要调用 LLM planner。"""

        social = obs.get("social") if isinstance(obs.get("social"), dict) else {}
        notifications = social.get("notifications") if isinstance(social.get("notifications"), list) else []
        if notifications or obs.get("people") or obs.get("actions"):
            return False
        recent_history = "\n".join(str(item) for item in self.history[-4:])
        return "failed" not in recent_history.lower() and "error" not in recent_history.lower() and "失败" not in recent_history

    def recall(self, obs: str | dict | list | None, context: str = "world") -> list[str]:
        """按场景召回记忆，返回 list[str] 以保持现有 prompt 兼容。"""

        top_k = self._memory_top_k_for(context)
        if hasattr(self.mem, "retrieve_context"):
            return self.mem.retrieve_context(
                self.id, obs, self.task, self.urgency, self.satisfaction_threshold,
                n_results=top_k,
                context=context,
                timeout_seconds=self._memory_timeout_seconds(),
            )
        obs_text = self._format_structured_context(obs)
        return self.mem.smart_retrieve(
            self.id, obs_text, self.task, self.urgency, self.satisfaction_threshold,
            n_results=top_k,
            context=context,
        )

    async def arecall(self, obs: str | dict | list | None, context: str = "world") -> list[str]:
        """异步记忆召回；优先使用结构化控制层，旧接口作为兜底。"""

        top_k = self._memory_top_k_for(context)
        if hasattr(self.mem, "aretrieve_context"):
            return await self.mem.aretrieve_context(
                self.id, obs, self.task, self.urgency, self.satisfaction_threshold,
                n_results=top_k,
                context=context,
                timeout_seconds=self._memory_timeout_seconds(),
            )
        obs_text = self._format_structured_context(obs)
        return await self.mem.asmart_retrieve(
            self.id, obs_text, self.task, self.urgency, self.satisfaction_threshold,
            n_results=top_k,
            context=context,
        )

    def _planned_recall(self, obs: str | dict | list | None, observation_text: str, context: str = "world") -> list[str]:
        """先由 LLM planner 生成查询计划，再执行受控记忆召回。"""

        planner = getattr(self, "memory_planner", None)
        if self._observation_has_direct_need_target(obs, context):
            # 当前 observe 已提供可直接行动的目标时，不调用 planner 和 Chroma，直接让 action LLM 决策。
            return []
        rule_plan = self._rule_based_memory_plan(obs, context)
        if rule_plan is not None and hasattr(self.mem, "execute_query_plan"):
            return self.mem.execute_query_plan(
                self.id,
                rule_plan,
                obs,
                self.task,
                self.urgency,
                self.satisfaction_threshold,
                n_results=self._memory_top_k_for(context),
                context=context,
                timeout_seconds=self._memory_timeout_seconds(),
            )
        if (
            not getattr(self.config, "memory_planner_enabled", True)
            or planner is None
            or not hasattr(planner, "plan")
            or not hasattr(self.mem, "execute_query_plan")
        ):
            return self.recall(obs, context=context)
        try:
            plan = planner.plan(self, observation_text, context=context)
            self.add_history("memory_query", self._memory_query_history_summary(plan))
            return self.mem.execute_query_plan(
                self.id,
                plan,
                obs,
                self.task,
                self.urgency,
                self.satisfaction_threshold,
                n_results=self._memory_top_k_for(context),
                context=context,
                timeout_seconds=self._memory_timeout_seconds(),
            )
        except Exception as exc:
            logger.warning("[%s] 记忆查询计划执行失败，退回自动召回: %s", self.id, exc)
            return self.recall(obs, context=context)

    async def _aplanned_recall(self, obs: str | dict | list | None, observation_text: str, context: str = "world") -> list[str]:
        """异步 planner 召回；失败时保留旧召回路径。"""

        planner = getattr(self, "memory_planner", None)
        if self._observation_has_direct_need_target(obs, context):
            # 当前观察足够完成生理/安全任务时，跳过记忆查询，减少无效 LLM/embedding 调用。
            return []
        rule_plan = self._rule_based_memory_plan(obs, context)
        if rule_plan is not None and hasattr(self.mem, "aexecute_query_plan"):
            return await self.mem.aexecute_query_plan(
                self.id,
                rule_plan,
                obs,
                self.task,
                self.urgency,
                self.satisfaction_threshold,
                n_results=self._memory_top_k_for(context),
                context=context,
                timeout_seconds=self._memory_timeout_seconds(),
            )
        if (
            not getattr(self.config, "memory_planner_enabled", True)
            or planner is None
            or not hasattr(planner, "aplan")
            or not hasattr(self.mem, "aexecute_query_plan")
        ):
            return await self.arecall(obs, context=context)
        try:
            plan = await planner.aplan(self, observation_text, context=context)
            self.add_history("memory_query", self._memory_query_history_summary(plan))
            return await self.mem.aexecute_query_plan(
                self.id,
                plan,
                obs,
                self.task,
                self.urgency,
                self.satisfaction_threshold,
                n_results=self._memory_top_k_for(context),
                context=context,
                timeout_seconds=self._memory_timeout_seconds(),
            )
        except Exception as exc:
            logger.warning("[%s] 异步记忆查询计划执行失败，退回自动召回: %s", self.id, exc)
            return await self.arecall(obs, context=context)

    def _memory_query_history_summary(self, plan: str) -> str:
        """短期 history 只记录本轮查询摘要，避免完整 planner JSON 挤占历史窗口。"""

        try:
            data = json.loads(plan)
        except (TypeError, json.JSONDecodeError):
            return str(plan)[:180]
        queries = data.get("queries") if isinstance(data.get("queries"), list) else []
        types = []
        for query in queries[:5]:
            if isinstance(query, dict) and query.get("type"):
                types.append(str(query.get("type")))
        think = str(data.get("think") or "")[:120]
        return f"context={data.get('context', 'world')} query_types={','.join(types) or 'none'} think={think}"

    def _format_structured_context(self, value: str | dict | list | None) -> str:
        """把结构化 JSON 转成 prompt 文本；拼接工作集中在 Agent 层完成。"""

        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, indent=2)

    def remember(self, info: str, **metadata) -> None:
        self.mem.store_agent_memory(self.id, info, world_time=self.world.time, **metadata)

    async def aremember(self, info: str, **metadata) -> None:
        await self.mem.astore_agent_memory(self.id, info, world_time=self.world.time, **metadata)

    def _store_observation_memory(self, observation: str | dict | list | None) -> None:
        """把 observe 原始 JSON 交给记忆控制层，Agent 不在这里拆字段。"""

        if hasattr(self.mem, "store_observation"):
            self.mem.store_observation(self.id, observation)

    def _store_social_browse_memory(self, posts_info: dict) -> None:
        """把社交平台返回的 JSON 浏览结果写入记忆系统。"""

        if hasattr(self.mem, "store_social_browse"):
            self.mem.store_social_browse(self.id, posts_info)

    def _store_social_feedback_memory(self, feedback: dict | str | None) -> None:
        """把社交平台执行反馈写入记忆系统，时间以当前 world.time 为准。"""

        if hasattr(self.mem, "store_social_feedback"):
            self.mem.store_social_feedback(self.id, feedback, world_time=self.world.time)

    def _store_conversation_memory(
        self,
        *,
        messages: list[dict] | None = None,
        reply: str | None = None,
        observation: str = "",
    ) -> None:
        """把对话消息和回复作为结构化 payload 交给记忆系统。"""

        if hasattr(self.mem, "store_conversation"):
            self.mem.store_conversation(
                self.id,
                messages=messages,
                reply=reply,
                observation=observation,
                world_time=self.world.time,
            )

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
        if hasattr(self.mem, "store_action_result"):
            # 轨迹总结本身是任务完成证据，也写入结构化事件供后续 world 检索使用。
            self.mem.store_action_result(
                self.id,
                decision={
                    "think": "task trajectory completed",
                    "action": {"tool": "task_summary", "args": {"task": task, "need_key": need_key}},
                },
                feedback=summary,
                reward=None,
                world_time=self.world.time,
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
        if hasattr(self.mem, "store_action_result"):
            # 异步轨迹总结与同步路径保持一致，也写入结构化任务完成事件。
            self.mem.store_action_result(
                self.id,
                decision={
                    "think": "task trajectory completed",
                    "action": {"tool": "task_summary", "args": {"task": task, "need_key": need_key}},
                },
                feedback=summary,
                reward=None,
                world_time=self.world.time,
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
        # 感知距离使用曼哈顿距离，避免为一个简单计算保留单独工具模块。
        return abs(self.position[0] - event.position[0]) + abs(self.position[1] - event.position[1]) <= r

    def receive_message(
        self,
        sender_id: str,
        content: str,
        response_to: str | None = None,
        session_id: str | None = None,
        intent: str | None = None,
        target: str | None = None,
        social_valence=None,
        topic: str = "",
        topic_stance=None,
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
            "social_valence": social_valence,
            "topic": topic,
            "topic_stance": topic_stance,
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
                metadata_parts = []
                if entry.get("social_valence") is not None:
                    metadata_parts.append(f"social_valence={entry.get('social_valence')}")
                if entry.get("topic"):
                    metadata_parts.append(f"topic={entry.get('topic')}")
                if entry.get("topic_stance") is not None:
                    metadata_parts.append(f"topic_stance={entry.get('topic_stance')}")
                line = (
                    f"  [第{entry['round']}轮{session_text}{intent_text}] "
                    f"{entry['sender']} → {entry['target']}: {entry['content']}"
                )
                if metadata_parts:
                    line += " [" + " ".join(metadata_parts) + "]"
                if entry.get("response_to"):
                    line += f"（回复: {entry['response_to']}）"
                lines.append(line)
            history_block = "[本时间步对话历史]\n" + "\n".join(lines)
        else:
            history_block = "[本时间步对话历史]\n（暂无）"

        inbox_messages = list(self.inbox)
        msgs = "\n".join([
            (
                f"{m['sender']} 对你说"
                f"{'（session=' + str(m.get('session_id')) + '）' if m.get('session_id') else ''}"
                f"{'（intent=' + str(m.get('intent')) + '）' if m.get('intent') else ''}: "
                f"{m['content']}"
            ) +
            (f"（回复的是: {m['response_to']}）" if m.get("response_to") else "")
            for m in inbox_messages
        ])
        metadata_lines = []
        for m in inbox_messages:
            metadata_lines.append(
                "sender={sender} session={session} intent={intent} social_valence={valence} topic={topic} topic_stance={stance}".format(
                    sender=m.get("sender"),
                    session=m.get("session_id") or "",
                    intent=m.get("intent") or "",
                    valence=m.get("social_valence"),
                    topic=m.get("topic") or "",
                    stance=m.get("topic_stance"),
                )
            )
        metadata_block = "[对话元数据]\n" + "\n".join(metadata_lines)
        observation = (
            f"{history_block}\n\n"
            f"[对话消息 · 第 {round_n} 轮，还剩 {remaining} 轮]\n"
            f"{msgs}\n\n{metadata_block}"
        )
        self.inbox.clear()
        self.add_history("conversation", observation)
        mem_info = self._planned_recall(observation, observation, context="conversation")
        action = policy.decide(self, observation, mem_info)
        self._store_conversation_memory(messages=inbox_messages, reply=action, observation=observation)
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
            self.urgency[urgency_key] = self._clamp_unit(self.urgency[urgency_key] + urgency_delta)

    @staticmethod
    def _clamp_unit(value: float) -> float:
        """把普通比例值限制在 [0, 1]。"""

        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _clamp_satisfaction(value: float) -> float:
        """非 money 满足度限制在 [0, 100]。"""

        return max(0.0, min(100.0, float(value)))

    def _clamp_urgency(self, value: float, floor: float) -> float:
        return max(floor, min(1.0, float(value)))

    def _urgency_floor(self, satisfaction_key: str) -> float:
        floor = self.config.urgency_floors.get(satisfaction_key, 0.0)
        return self._clamp_unit(floor)

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
            # money 无上限（累计金额）；其他满足度限制在 [0, 100]。
            if satisfaction_key == "money":
                self.satisfaction[satisfaction_key] = max(0.0, new_val)
            else:
                self.satisfaction[satisfaction_key] = self._clamp_satisfaction(new_val)
            self.update_need_pressure(accumulate=False)
            self.update_urgency_from_satisfaction()

    def _need_threshold(self, satisfaction_key: str) -> float | None:
        """读取需求阈值，非法阈值返回 None。"""

        threshold = self.satisfaction_threshold.get(satisfaction_key)
        if not isinstance(threshold, (int, float)) or threshold <= 0:
            return None
        return float(threshold)

    def _normalized_need_gap(self, satisfaction_key: str) -> float:
        threshold = self._need_threshold(satisfaction_key)
        if threshold is None:
            return 0.0
        satisfaction = self.satisfaction.get(satisfaction_key, 0.0)
        return max(0.0, (threshold - satisfaction) / threshold)

    def _updated_pressure_memory(self, key: str, gap: float, memory: float, dt: float, accumulate: bool) -> float:
        """根据当前缺口更新压力记忆；睡眠等场景可关闭累积。"""

        if accumulate and dt > 0:
            if gap > 0:
                memory += gap * dt
            else:
                recovery_rate = self.config.pressure_recovery_rates.get(
                    key,
                    self.config.pressure_default_recovery_rate,
                )
                memory *= max(0.0, 1.0 - recovery_rate * dt)
        return max(0.0, memory)

    def _pressure_load_from_memory(self, key: str, memory: float) -> float:
        """把压力记忆压缩为 [0, 1] 的负荷饱和值。"""

        kappa = self.config.pressure_load_kappas.get(
            key,
            self.config.pressure_default_load_kappa,
        )
        load = 0.0 if kappa <= 0 else 1.0 - math.exp(-memory / kappa)
        return self._clamp_unit(load)

    def _effective_pressure_value(self, key: str, gap: float, load: float) -> float:
        """合成瞬时缺口、残余负荷和放大项。"""

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
        return max(0.0, pressure)

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
            memory = self._updated_pressure_memory(key, gap, memory, dt, accumulate)
            self.pressure_memory[key] = memory

            # 指数饱和避免 pressure_memory 无限增长后直接支配心理评测。
            load = self._pressure_load_from_memory(key, memory)
            self.load_saturation[key] = load

            self.effective_pressure[key] = self._effective_pressure_value(key, gap, load)

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
        apply_need_delta(
            self,
            "satiety",
            -self.config.satiety_decay_rate,
            source="physiological",
            reason="清醒状态下饱腹度自然消耗",
            evidence={"rate": self.config.satiety_decay_rate},
        )
        if not self.did_move_this_tick and not self.did_work_this_tick:
            apply_need_delta(
                self,
                "relax",
                self.config.relax_increase_rate,
                source="physiological",
                reason="未移动且未工作时自然恢复 relax",
                evidence={"rate": self.config.relax_increase_rate},
            )
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
        apply_need_delta(
            self,
            "relax",
            self.config.sleep_relax_recover / sleep_time,
            source="physiological",
            reason="睡眠期间恢复 relax",
            evidence={
                "bed_id": self.sleeping_on_bed_id,
                "sleep_time": sleep_time,
            },
        )

    def remember_action_tool(self, tool_name: str) -> None:
        """记录近期动作类型，供高层需求被动衰减判断。"""

        self.recent_action_tools.append(str(tool_name or ""))
        max_len = max(30, self.config.self_actualization_repetition_window_ticks * 2)
        if len(self.recent_action_tools) > max_len:
            self.recent_action_tools = self.recent_action_tools[-max_len:]

    def _sleep_satisfaction_changes(self) -> dict[str, float]:
        """计算本次睡眠期间各需求满足度的变化。"""

        return {
            key: round(self.satisfaction.get(key, 0) - self._sleep_start_satisfaction.get(key, 0), 2)
            for key in self.satisfaction
        }

    def _place_near_sleep_position(self) -> bool:
        """睡醒后把智能体放回床周围空格；无空格时保留当前位置。"""

        bx, by = self.position
        with self.world._world_lock:
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                nx, ny = bx + dx, by + dy
                if 0 <= nx < self.world.map.height and 0 <= ny < self.world.map.width:
                    if self.world.map.is_empty(nx, ny):
                        self.world.map.place(nx, ny, self.id)
                        self.position = [nx, ny]
                        return True
            if self.world.map.is_empty(bx, by):
                self.world.map.place(bx, by, self.id)
                return True
        return False

    def wakeup(self, bed) -> None:
        """结束睡眠并把智能体从床位置移回周围空格。"""

        elapsed = self.world.time - self._sleep_start_time
        changes = self._sleep_satisfaction_changes()
        change_str = "，".join(f"{k} {'+' if v >= 0 else ''}{v}" for k, v in changes.items())
        self.sleeping = False
        self.sleep_ticks_remaining = 0
        self.sleeping_on_bed_id = None
        self.task = "none"
        if bed is not None:
            bed.exit_bed(self)
        if not self._place_near_sleep_position():
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
    
