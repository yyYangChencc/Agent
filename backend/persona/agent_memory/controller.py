from __future__ import annotations

import json
from typing import Any

from persona.agent_memory.structured_store import StructuredMemoryStore
from persona.logger import get_logger

logger = get_logger(__name__)


WORLD_STRUCTURED_TYPES = ["semantic", "procedural", "episodic", "reflective", "social"]
SOCIAL_STRUCTURED_TYPES = ["social", "semantic", "episodic", "reflective"]
CONVERSATION_STRUCTURED_TYPES = ["social", "episodic", "reflective", "semantic"]
OPINION_STRUCTURED_TYPES = ["social", "semantic", "episodic", "reflective"]


def _as_dict(value: Any) -> dict[str, Any]:
    """把上游传入的 JSON 字符串或 dict 统一成 dict；无法解析时放弃结构化写入。"""

    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _as_text(value: Any) -> str:
    """把结构化观察转成稳定文本，供 Chroma 语义检索继续使用。"""

    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _text_id(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _short(text: str, limit: int = 180) -> str:
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


class MemoryController:
    """结构化记忆控制层。

    业务模块只提交 observe、action、social、conversation、opinion 等 payload；
    本层负责拆分写入 SQLite，并在查询时把结构化命中和 Chroma 语义召回合并为 list[str]。
    """

    def __init__(self, store: StructuredMemoryStore, semantic_manager):
        self.store = store
        self.semantic_manager = semantic_manager

    def store_observation(self, agent_id: str, observation: Any) -> None:
        """写入 observe 结果。

        observe 已按 people、objects、actions、social 分类；这里分别落到场景快照、
        实体当前状态、原始事件和关系表。
        """

        data = _as_dict(observation)
        if not data:
            return
        world_time = int(data.get("time") or 0)
        self.store.record_scene_snapshot(agent_id, data)
        # 场景快照先作为一次低重要性原始事件保存，便于回放完整观察。
        self.store.record_event(
            agent_id,
            memory_type="episodic",
            source_type="observe",
            event_type="scene_snapshot",
            world_time=world_time,
            payload=data,
            summary=self._scene_summary(data),
            importance=0.35,
            confidence=0.8,
        )

        for person in data.get("people") or []:
            if not isinstance(person, dict):
                continue
            person_id = person.get("id")
            # 人物基础信息进入 person_profiles，替换旧的 person entity_state 记忆方式。
            self.store.upsert_person_profile_seen(
                agent_id,
                target_agent_id=person_id,
                name=_text_id(person.get("name") or person_id),
                position=person.get("position"),
                region=person.get("region"),
                world_time=world_time,
                confidence=0.85,
            )
            self.store.record_event(
                agent_id,
                memory_type="episodic",
                source_type="observe",
                event_type="person_seen",
                world_time=world_time,
                related_agent_id=person_id,
                entity_id=person_id,
                payload=person,
                summary=self._person_summary(person, world_time),
                importance=0.45,
                confidence=0.85,
            )

        for obj in data.get("objects") or []:
            if not isinstance(obj, dict):
                continue
            object_id = obj.get("id")
            # 物品和建筑按 object 统一建模，位置变化交给状态表和冲突表处理。
            self.store.upsert_entity_state(
                agent_id,
                entity_type="object",
                entity_id=object_id,
                name=_text_id(obj.get("name") or object_id),
                position=obj.get("position"),
                region=obj.get("region"),
                source_type="observe",
                world_time=world_time,
                payload=obj,
                importance=0.6,
                confidence=0.85,
            )
            self.store.record_event(
                agent_id,
                memory_type="semantic",
                source_type="observe",
                event_type="object_seen",
                world_time=world_time,
                object_id=object_id,
                entity_id=object_id,
                payload=obj,
                summary=self._object_summary(obj, world_time),
                importance=0.5,
                confidence=0.85,
            )

        for action in data.get("actions") or []:
            if not isinstance(action, dict):
                continue
            actor_id = action.get("actor_id")
            acted_id = action.get("acted_id")
            action_time = int(action.get("time") or world_time)
            # 动作既是情景事件，也会形成 actor -> target 的近期关系。
            self.store.record_event(
                agent_id,
                memory_type="episodic",
                source_type="observe",
                event_type=_text_id(action.get("type") or "action_seen"),
                world_time=action_time,
                related_agent_id=actor_id,
                object_id=acted_id,
                entity_id=actor_id or acted_id,
                payload=action,
                summary=self._action_summary(action),
                importance=0.65,
                confidence=0.8,
            )
            self.store.upsert_relation(
                agent_id,
                subject_id=actor_id,
                relation_type=_text_id(action.get("type") or "acted_on"),
                object_id=acted_id or "unknown",
                source_type="observe",
                world_time=action_time,
                payload=action,
                importance=0.55,
                confidence=0.75,
            )

        social = data.get("social") if isinstance(data.get("social"), dict) else {}
        for note in social.get("notifications") or []:
            if not isinstance(note, dict):
                continue
            self.store.record_event(
                agent_id,
                memory_type="social",
                source_type="observe",
                event_type="notification",
                world_time=int(note.get("time") or world_time),
                payload=note,
                summary=f"t={note.get('time', world_time)} social notification: {_short(note.get('content', ''))}",
                importance=0.5,
                confidence=0.75,
            )

    def store_action_result(
        self,
        agent_id: str,
        *,
        decision: Any,
        feedback: Any,
        reward: float | None,
        world_time: int,
    ) -> None:
        """写入 World.execute 后的决策结果。

        action 的真实反馈和 reward 只有执行阶段才知道，因此动作记忆在 World.execute
        之后写入，避免记录未发生的计划。
        """

        decision_data = _as_dict(decision)
        action = decision_data.get("action") if isinstance(decision_data.get("action"), dict) else {}
        tool = _text_id(action.get("tool"))
        args = action.get("args") if isinstance(action.get("args"), dict) else {}
        payload = {
            "decision": decision_data,
            "feedback": feedback,
            "reward": reward,
        }
        acted = args.get("ID") or args.get("post_id")
        self.store.record_event(
            agent_id,
            memory_type="episodic",
            source_type="action_result",
            event_type=tool or "no_action",
            world_time=world_time,
            entity_id=acted,
            object_id=acted,
            post_id=args.get("post_id"),
            payload=payload,
            summary=self._action_result_summary(agent_id, decision_data, feedback, reward, world_time),
            importance=0.6 if tool else 0.25,
            confidence=0.75,
        )
        if acted:
            self.store.upsert_relation(
                agent_id,
                subject_id=agent_id,
                relation_type=tool or "selected_no_action",
                object_id=acted,
                source_type="action_result",
                world_time=world_time,
                payload=payload,
                importance=0.55,
                confidence=0.75,
            )

    def store_social_browse(self, agent_id: str, browse_payload: Any) -> None:
        """写入一次社交平台浏览结果，包括可见帖子列表和帖子快照。"""

        data = _as_dict(browse_payload)
        if not data:
            return
        world_time = int(data.get("time") or 0)
        visible_ids = data.get("visible_post_ids") or []
        self.store.record_event(
            agent_id,
            memory_type="social",
            source_type="social_browse",
            event_type="browse_posts",
            world_time=world_time,
            payload=data,
            summary=f"t={world_time} browsed social posts: {', '.join(str(x) for x in visible_ids)}",
            importance=0.45,
            confidence=0.8,
        )
        for post in data.get("posts") or []:
            if not isinstance(post, dict):
                continue
            self.store.upsert_social_post(
                agent_id,
                post,
                source_type="social_browse",
                world_time=world_time,
                importance=0.65 if post.get("is_news") else 0.5,
                confidence=0.85,
            )
            self._store_social_people_profiles(agent_id, post, world_time)
            self.store.upsert_relation(
                agent_id,
                subject_id=_text_id(post.get("author_id")) or "unknown",
                relation_type="authored_post",
                object_id=post.get("id"),
                source_type="social_browse",
                world_time=world_time,
                payload=post,
                importance=0.5,
                confidence=0.8,
            )

    def store_social_feedback(self, agent_id: str, feedback_payload: Any, *, world_time: int | None = None) -> None:
        """写入发帖、点赞、点踩、评论等社交操作反馈。"""

        data = _as_dict(feedback_payload)
        if not data:
            return
        post = data.get("post") if isinstance(data.get("post"), dict) else None
        post_id = data.get("post_id") or (post or {}).get("id")
        event_time = int(world_time if world_time is not None else data.get("time") or 0)
        if post is not None:
            self.store.upsert_social_post(
                agent_id,
                post,
                source_type="social_feedback",
                world_time=event_time,
                importance=0.65,
                confidence=0.85,
            )
            self._store_social_people_profiles(agent_id, post, event_time)
            author_id = post.get("author_id")
        else:
            author_id = None
        action = _text_id(data.get("action") or "social_no_action")
        self.store.record_event(
            agent_id,
            memory_type="social",
            source_type="social_feedback",
            event_type=action,
            world_time=event_time,
            post_id=post_id,
            related_agent_id=author_id,
            entity_id=author_id or post_id,
            payload=data,
            summary=self._social_feedback_summary(data, event_time),
            importance=0.7 if data.get("ok") else 0.45,
            confidence=0.8,
        )
        if post_id:
            self.store.upsert_relation(
                agent_id,
                subject_id=agent_id,
                relation_type=action,
                object_id=post_id,
                source_type="social_feedback",
                world_time=event_time,
                payload=data,
                importance=0.6,
                confidence=0.8,
            )

    def store_conversation(
        self,
        agent_id: str,
        *,
        messages: list[dict[str, Any]] | None = None,
        reply: Any = None,
        observation: str = "",
        world_time: int = 0,
    ) -> None:
        """写入线下对话消息和本智能体回复。

        发送者以 related_agent_id/entity_id 建索引，后续 conversation 检索可以优先召回
        这个人的历史消息和关系。
        """

        payload = {
            "messages": messages or [],
            "reply": _as_dict(reply) if isinstance(reply, str) else reply,
            "observation": observation,
        }
        for message in messages or []:
            if not isinstance(message, dict):
                continue
            sender = message.get("sender")
            self.store.record_event(
                agent_id,
                memory_type="social",
                source_type="conversation",
                event_type="message_received",
                world_time=int(message.get("time") or world_time),
                related_agent_id=sender,
                entity_id=sender,
                payload=message,
                summary=f"t={message.get('time', world_time)} {sender} said: {_short(message.get('content', ''))}",
                importance=0.65,
                confidence=0.8,
            )
            self.store.upsert_relation(
                agent_id,
                subject_id=sender,
                relation_type="sent_message_to",
                object_id=agent_id,
                source_type="conversation",
                world_time=int(message.get("time") or world_time),
                payload=message,
                importance=0.55,
                confidence=0.75,
            )
        if reply:
            self.store.record_event(
                agent_id,
                memory_type="social",
                source_type="conversation",
                event_type="conversation_reply",
                world_time=world_time,
                payload=payload,
                summary=f"t={world_time} conversation reply: {_short(reply)}",
                importance=0.6,
                confidence=0.75,
            )

    def store_opinion_assessment(self, agent_id: str, assessment: Any) -> None:
        """写入观念评测结果。

        观念评测属于 reflective 证据；同时写入 derived_memories，便于后续评测查询到
        之前的 reason、evidence 和 confidence。
        """

        data = _as_dict(assessment)
        if not data:
            return
        world_time = int(data.get("tick") or 0)
        topic = _text_id(data.get("topic"))
        summary = (
            f"t={world_time} opinion topic={topic} score={data.get('score')} "
            f"reason={_short(data.get('reason', ''))}"
        )
        self.store.record_event(
            agent_id,
            memory_type="reflective",
            source_type="opinion_assessment",
            event_type="opinion_assessment",
            world_time=world_time,
            entity_id=topic,
            payload=data,
            summary=summary,
            importance=0.75,
            confidence=float(data.get("confidence") or 0.5),
        )
        self._record_derived_without_chroma(
            agent_id=agent_id,
            memory_type="reflective",
            source_type="opinion_assessment",
            world_time=world_time,
            summary=summary,
            payload=data,
            entity_id=topic,
            importance=0.7,
            confidence=float(data.get("confidence") or 0.5),
        )

    def record_derived_memory(
        self,
        agent_id: str,
        memory_id: str,
        memory_text: str,
        world_time: int,
        metadata: dict[str, Any],
    ) -> None:
        """接收 Chroma 长期记忆写入后的双写通知。"""

        memory_type = _text_id(metadata.get("memory_type") or metadata.get("type") or "episodic")
        payload = {
            "text": memory_text,
            "metadata": metadata,
        }
        self.store.record_derived_memory(
            memory_id=memory_id,
            agent_id=agent_id,
            memory_type=memory_type,
            source_type=_text_id(metadata.get("source_type") or "manual_memory"),
            world_time=world_time,
            summary=memory_text,
            payload=payload,
            entity_id=metadata.get("entity_id"),
            related_agent_id=metadata.get("related_agent_id"),
            object_id=metadata.get("object_id"),
            post_id=metadata.get("post_id"),
            task=metadata.get("task"),
            need_key=metadata.get("need_key"),
            importance=float(metadata.get("importance") or 0.5),
            confidence=float(metadata.get("confidence") or 0.5),
            valid=bool(metadata.get("valid", True)),
        )

    def retrieve_context(
        self,
        agent_id: str,
        observation: Any,
        task: str,
        urgency: dict,
        satisfaction_threshold: dict,
        *,
        n_results: int,
        context: str,
    ) -> list[str]:
        """同步检索入口：先按场景查 SQLite，再用剩余预算查 Chroma。"""

        payload = _as_dict(observation)
        query_text = observation if isinstance(observation, str) else _as_text(observation)
        structured = self._retrieve_structured(agent_id, payload, query_text, task, context, n_results)
        semantic_budget = max(1, n_results - len(structured))
        semantic = self.semantic_manager.smart_retrieve(
            agent_id,
            query_text,
            task,
            urgency,
            satisfaction_threshold,
            n_results=semantic_budget,
            context=context,
        )
        return self._dedupe(structured + semantic)[:n_results]

    async def aretrieve_context(
        self,
        agent_id: str,
        observation: Any,
        task: str,
        urgency: dict,
        satisfaction_threshold: dict,
        *,
        n_results: int,
        context: str,
    ) -> list[str]:
        """异步检索入口；结构化查询是本地 SQLite，语义查询走异步 Chroma 路径。"""

        payload = _as_dict(observation)
        query_text = observation if isinstance(observation, str) else _as_text(observation)
        structured = self._retrieve_structured(agent_id, payload, query_text, task, context, n_results)
        semantic_budget = max(1, n_results - len(structured))
        semantic = await self.semantic_manager.asmart_retrieve(
            agent_id,
            query_text,
            task,
            urgency,
            satisfaction_threshold,
            n_results=semantic_budget,
            context=context,
        )
        return self._dedupe(structured + semantic)[:n_results]

    def execute_query_plan(
        self,
        agent_id: str,
        query_plan: Any,
        observation: Any,
        task: str,
        urgency: dict,
        satisfaction_threshold: dict,
        *,
        n_results: int,
        context: str,
    ) -> list[str]:
        """执行 LLM planner 生成的受控记忆查询计划。

        planner 只表达信息需求；本函数负责把查询计划映射到固定 SQLite/Chroma
        查询函数，并强制过滤当前 observe 已经看见的物品最新状态。
        """

        plan = _as_dict(query_plan)
        payload = _as_dict(observation)
        query_text = observation if isinstance(observation, str) else _as_text(observation)
        if not plan or not plan.get("queries"):
            return self.retrieve_context(
                agent_id,
                observation,
                task,
                urgency,
                satisfaction_threshold,
                n_results=n_results,
                context=context,
            )
        return self._execute_query_plan_sync(
            agent_id,
            plan,
            payload,
            query_text,
            task,
            urgency,
            satisfaction_threshold,
            n_results=n_results,
            context=context,
        )

    async def aexecute_query_plan(
        self,
        agent_id: str,
        query_plan: Any,
        observation: Any,
        task: str,
        urgency: dict,
        satisfaction_threshold: dict,
        *,
        n_results: int,
        context: str,
    ) -> list[str]:
        """异步执行受控记忆查询计划；语义查询走异步 Chroma。"""

        plan = _as_dict(query_plan)
        payload = _as_dict(observation)
        query_text = observation if isinstance(observation, str) else _as_text(observation)
        if not plan or not plan.get("queries"):
            return await self.aretrieve_context(
                agent_id,
                observation,
                task,
                urgency,
                satisfaction_threshold,
                n_results=n_results,
                context=context,
            )
        return await self._execute_query_plan_async(
            agent_id,
            plan,
            payload,
            query_text,
            task,
            urgency,
            satisfaction_threshold,
            n_results=n_results,
            context=context,
        )

    def list_structured_memory(self, agent_id: str) -> dict[str, list[dict[str, Any]]]:
        """返回 API 可展示的结构化记忆分表数据。"""

        return self.store.list_agent_memory(agent_id)

    def update_person_profiles_from_reflection(self, agent_id: str, *, current_time: int, llm=None) -> dict[str, Any]:
        """在 reflect 阶段用新增事实更新人物档案印象。"""

        updated = 0
        profiles = self.store.get_profiles_pending_reflection(agent_id)
        for profile in profiles:
            target_id = _text_id(profile.get("target_agent_id"))
            if not target_id:
                continue
            facts = self.store.get_person_profile_facts(
                agent_id,
                target_id,
                since_time=int(profile.get("last_reflected_at") or 0),
            )
            impressions = self._summarize_person_profile(profile, facts, llm=llm)
            self.store.update_person_profile_impressions(
                agent_id,
                target_id,
                actions_impression=impressions["actions_impression"],
                opinion_impression=impressions["opinion_impression"],
                relationship_impression=impressions["relationship_impression"],
                reflected_at=current_time,
                confidence=impressions["confidence"],
            )
            updated += 1
        return {"agent_id": agent_id, "updated_person_profiles": updated}

    async def aupdate_person_profiles_from_reflection(self, agent_id: str, *, current_time: int, llm=None) -> dict[str, Any]:
        """异步 reflect 阶段的人物档案更新，避免 astep 中阻塞事件循环。"""

        updated = 0
        profiles = self.store.get_profiles_pending_reflection(agent_id)
        for profile in profiles:
            target_id = _text_id(profile.get("target_agent_id"))
            if not target_id:
                continue
            facts = self.store.get_person_profile_facts(
                agent_id,
                target_id,
                since_time=int(profile.get("last_reflected_at") or 0),
            )
            impressions = await self._asummarize_person_profile(profile, facts, llm=llm)
            self.store.update_person_profile_impressions(
                agent_id,
                target_id,
                actions_impression=impressions["actions_impression"],
                opinion_impression=impressions["opinion_impression"],
                relationship_impression=impressions["relationship_impression"],
                reflected_at=current_time,
                confidence=impressions["confidence"],
            )
            updated += 1
        return {"agent_id": agent_id, "updated_person_profiles": updated}

    def maintain_agent_memory(self, agent_id: str, *, current_time: int, raw_event_retention: int = 50) -> dict[str, Any]:
        """执行轻量维护：低价值原始事件失效化，并统计未解决冲突。"""

        prune_before = max(0, int(current_time) - max(1, int(raw_event_retention)))
        pruned_events = self.store.prune_low_value_events(agent_id, before_time=prune_before)
        unresolved_conflicts = self.store.get_unresolved_conflicts(agent_id)
        return {
            "agent_id": agent_id,
            "pruned_low_value_events": pruned_events,
            "unresolved_conflicts": len(unresolved_conflicts),
        }

    def _retrieve_structured(
        self,
        agent_id: str,
        payload: dict[str, Any],
        query_text: str,
        task: str,
        context: str,
        n_results: int,
    ) -> list[str]:
        """按使用场景路由结构化查询。

        world 重视实体状态、近期动作和任务轨迹；social 重视当前可见帖子和互动；
        conversation 重视发送者人物记忆和对话事件；opinion_assessment 重视社交、
        对话和历史观念证据。
        """

        world_time = int(payload.get("time") or 0) if payload else 0
        entity_ids = self._extract_entity_ids(payload, query_text)
        post_ids = self._extract_post_ids(payload, query_text)
        rows: list[tuple[str, str, Any]] = []

        if context == "social":
            # 社交决策优先看当前可见帖子，再补充近期社交反馈和浏览记录。
            for row in self.store.get_person_profiles(agent_id, entity_ids, limit=8):
                rows.append(("person_profile", self._format_person_profile(row), row.get("target_agent_id")))
            for row in self.store.get_social_posts(agent_id, post_ids, limit=8):
                rows.append(("social_post", self._format_social_post(row), row.get("post_id")))
            for row in self.store.get_recent_events(
                agent_id,
                source_types=["social_feedback", "social_browse", "opinion_assessment"],
                memory_types=SOCIAL_STRUCTURED_TYPES,
                entity_ids=entity_ids,
                post_ids=post_ids,
                limit=8,
            ):
                rows.append(("event", self._format_event(row), row.get("id")))
            for row in self.store.get_recent_social_posts(agent_id, limit=5):
                rows.append(("social_post", self._format_social_post(row), row.get("post_id")))
        elif context == "conversation":
            # 对话决策优先看发言者或目标人物的状态、关系和历史消息。
            for row in self.store.get_person_profiles(agent_id, entity_ids, limit=8):
                rows.append(("person_profile", self._format_person_profile(row), row.get("target_agent_id")))
            for row in self.store.get_recent_relations(agent_id, subject_ids=entity_ids, object_ids=entity_ids, limit=8):
                rows.append(("relation", self._format_relation(row), row.get("id")))
            for row in self.store.get_recent_events(
                agent_id,
                source_types=["conversation", "observe"],
                memory_types=CONVERSATION_STRUCTURED_TYPES,
                entity_ids=entity_ids,
                limit=8,
            ):
                rows.append(("event", self._format_event(row), row.get("id")))
        elif context == "opinion_assessment":
            # 观念评测不依赖单个实体 id，侧重近期社交/对话/评测证据。
            for row in self.store.get_recent_events(
                agent_id,
                source_types=["opinion_assessment", "social_feedback", "social_browse", "conversation"],
                memory_types=OPINION_STRUCTURED_TYPES,
                limit=12,
            ):
                rows.append(("event", self._format_event(row), row.get("id")))
            for row in self.store.get_recent_social_posts(agent_id, limit=8):
                rows.append(("social_post", self._format_social_post(row), row.get("post_id")))
        else:
            # 默认世界行动查询：当前实体状态、近期经历、长期衍生记忆共同参与。
            for row in self.store.get_person_profiles(agent_id, entity_ids, limit=8):
                rows.append(("person_profile", self._format_person_profile(row), row.get("target_agent_id")))
            for row in self.store.get_entity_states(agent_id, entity_ids, limit=10):
                rows.append(("entity", self._format_entity_state(row), row.get("entity_id")))
            for row in self.store.get_recent_events(
                agent_id,
                source_types=["observe", "action_result", "conversation"],
                memory_types=WORLD_STRUCTURED_TYPES,
                entity_ids=entity_ids,
                limit=8,
            ):
                rows.append(("event", self._format_event(row), row.get("id")))
            for row in self.store.get_recent_derived(
                agent_id,
                memory_types=["procedural", "episodic", "reflective", "semantic"],
                task=task,
                limit=6,
            ):
                rows.append(("derived", self._format_derived(row), row.get("memory_id")))
            if len(rows) < max(2, n_results // 2):
                # 观察里没有明确实体时，补充最近看到的状态，避免结构化结果为空。
                for row in self.store.get_recent_person_profiles(agent_id, limit=4):
                    rows.append(("person_profile", self._format_person_profile(row), row.get("target_agent_id")))
                for row in self.store.get_recent_entity_states(agent_id, limit=6):
                    rows.append(("entity", self._format_entity_state(row), row.get("entity_id")))

        formatted: list[str] = []
        for kind, text, ref in rows:
            if not text:
                continue
            formatted.append(text)
            self.store.record_access(
                agent_id=agent_id,
                context=context,
                memory_kind=kind,
                memory_ref=ref or text,
                world_time=world_time,
                query_text=query_text,
            )
            if len(formatted) >= max(n_results, 1):
                break
        return formatted

    def _execute_query_plan_sync(
        self,
        agent_id: str,
        plan: dict[str, Any],
        payload: dict[str, Any],
        query_text: str,
        task: str,
        urgency: dict,
        satisfaction_threshold: dict,
        *,
        n_results: int,
        context: str,
    ) -> list[str]:
        """同步执行 planner 查询计划。"""

        rows = []
        visible = self._visible_refs(payload)
        world_time = int(payload.get("time") or 0) if payload else 0
        for query in self._limited_queries(plan):
            if query.get("type") == "semantic":
                semantic_query = self._semantic_query_text(query, query_text, task)
                results = self.semantic_manager.smart_retrieve(
                    agent_id,
                    semantic_query,
                    task,
                    urgency,
                    satisfaction_threshold,
                    n_results=self._limit(query),
                    context=context,
                    memory_types=self._list_value(query.get("memory_types")),
                )
                for text in results:
                    rows.append(("semantic", text, semantic_query))
                continue
            for kind, text, ref in self._execute_structured_query(agent_id, query, visible, task, context):
                rows.append((kind, text, ref))

        return self._mix_plan_results(agent_id, rows, context=context, world_time=world_time, query_text=query_text, n_results=n_results)

    async def _execute_query_plan_async(
        self,
        agent_id: str,
        plan: dict[str, Any],
        payload: dict[str, Any],
        query_text: str,
        task: str,
        urgency: dict,
        satisfaction_threshold: dict,
        *,
        n_results: int,
        context: str,
    ) -> list[str]:
        """异步执行 planner 查询计划，保留 Chroma 语义召回入口。"""

        rows = []
        visible = self._visible_refs(payload)
        world_time = int(payload.get("time") or 0) if payload else 0
        for query in self._limited_queries(plan):
            if query.get("type") == "semantic":
                semantic_query = self._semantic_query_text(query, query_text, task)
                results = await self.semantic_manager.asmart_retrieve(
                    agent_id,
                    semantic_query,
                    task,
                    urgency,
                    satisfaction_threshold,
                    n_results=self._limit(query),
                    context=context,
                    memory_types=self._list_value(query.get("memory_types")),
                )
                for text in results:
                    rows.append(("semantic", text, semantic_query))
                continue
            for kind, text, ref in self._execute_structured_query(agent_id, query, visible, task, context):
                rows.append((kind, text, ref))

        return self._mix_plan_results(agent_id, rows, context=context, world_time=world_time, query_text=query_text, n_results=n_results)

    def _execute_structured_query(
        self,
        agent_id: str,
        query: dict[str, Any],
        visible: dict[str, list[str]],
        task: str,
        context: str,
    ) -> list[tuple[str, str, Any]]:
        """把单条受控 query 映射到 SQLite 查询函数。"""

        query_type = query.get("type")
        limit = self._limit(query)
        out: list[tuple[str, str, Any]] = []
        if query_type == "person_profile":
            ids = self._query_ids(query, "target_agent_ids", "target_agent_id")
            if not ids:
                ids = visible["people"]
            for row in self.store.get_person_profiles(agent_id, ids, limit=limit):
                out.append(("person_profile", self._format_person_profile(row), row.get("target_agent_id")))
        elif query_type == "entity_state":
            ids = self._query_ids(query, "entity_ids", "entity_id")
            exclude_visible = bool(query.get("exclude_visible", True))
            if ids:
                rows = self.store.get_entity_states(agent_id, ids, limit=limit)
                if exclude_visible:
                    rows = [row for row in rows if _text_id(row.get("entity_id")) not in set(visible["objects"])]
            else:
                rows = self.store.search_entity_states(
                    agent_id,
                    entity_type=_text_id(query.get("entity_type")),
                    kinds=query.get("kinds") if isinstance(query.get("kinds"), list) else [],
                    exclude_entity_ids=visible["objects"] if exclude_visible else [],
                    limit=limit,
                )
            for row in rows:
                out.append(("entity", self._format_entity_state(row), row.get("entity_id")))
        elif query_type == "event_history":
            for row in self.store.get_recent_events(
                agent_id,
                source_types=self._list_value(query.get("source_types")),
                memory_types=self._list_value(query.get("memory_types")) or self._event_types_for_context(context),
                entity_ids=self._query_ids(query, "entity_ids", "entity_id"),
                post_ids=self._query_ids(query, "post_ids", "post_id"),
                limit=limit,
            ):
                out.append(("event", self._format_event(row), row.get("id")))
        elif query_type == "social_post":
            post_ids = self._query_ids(query, "post_ids", "post_id")
            author_ids = self._query_ids(query, "author_ids", "author_id")
            rows = []
            if post_ids:
                rows.extend(self.store.get_social_posts(agent_id, post_ids, limit=limit))
            if author_ids:
                rows.extend(self.store.get_social_posts_by_authors(agent_id, author_ids, limit=limit))
            if not rows and context == "social":
                rows.extend(self.store.get_recent_social_posts(agent_id, limit=limit))
            for row in rows[:limit]:
                out.append(("social_post", self._format_social_post(row), row.get("post_id")))
        elif query_type == "derived_memory":
            for row in self.store.get_recent_derived(
                agent_id,
                memory_types=self._list_value(query.get("memory_types")),
                task=_text_id(query.get("task") or task),
                limit=limit,
            ):
                out.append(("derived", self._format_derived(row), row.get("memory_id")))
        return out

    def _mix_plan_results(
        self,
        agent_id: str,
        rows: list[tuple[str, str, Any]],
        *,
        context: str,
        world_time: int,
        query_text: str,
        n_results: int,
    ) -> list[str]:
        """按类型配额融合结果，给 Chroma semantic 保留进入 prompt 的空间。"""

        quotas = self._plan_quotas(context, n_results)
        grouped: dict[str, list[tuple[str, Any]]] = {}
        for kind, text, ref in rows:
            if not text:
                continue
            grouped.setdefault(kind, []).append((text, ref))

        ordered_kinds = ["person_profile", "entity", "event", "social_post", "derived", "semantic"]
        selected: list[tuple[str, str, Any]] = []
        for kind in ordered_kinds:
            quota = quotas.get(kind, 0)
            for text, ref in grouped.get(kind, [])[:quota]:
                selected.append((kind, text, ref))
        if len(selected) < n_results:
            for kind in ordered_kinds:
                for text, ref in grouped.get(kind, []):
                    if any(existing == text for _, existing, _ in selected):
                        continue
                    selected.append((kind, text, ref))
                    if len(selected) >= n_results:
                        break
                if len(selected) >= n_results:
                    break

        formatted = []
        for kind, text, ref in selected:
            if text in formatted:
                continue
            formatted.append(text)
            self.store.record_access(
                agent_id=agent_id,
                context=context,
                memory_kind=kind,
                memory_ref=ref or text,
                world_time=world_time,
                query_text=query_text,
            )
            if len(formatted) >= n_results:
                break
        return formatted

    def _visible_refs(self, payload: dict[str, Any]) -> dict[str, list[str]]:
        """提取当前输入中已经可见的人、物品和帖子，执行层据此避免重复召回。"""

        people = []
        objects = []
        posts = []
        for person in payload.get("people") or []:
            if isinstance(person, dict):
                people.append(_text_id(person.get("id")))
        for obj in payload.get("objects") or []:
            if isinstance(obj, dict):
                objects.append(_text_id(obj.get("id")))
        posts.extend(_text_id(value) for value in payload.get("visible_post_ids") or [])
        for post in payload.get("posts") or []:
            if isinstance(post, dict):
                posts.append(_text_id(post.get("id")))
        return {
            "people": self._unique(people),
            "objects": self._unique(objects),
            "posts": self._unique(posts),
        }

    def _limited_queries(self, plan: dict[str, Any]) -> list[dict[str, Any]]:
        queries = plan.get("queries") if isinstance(plan.get("queries"), list) else []
        return [query for query in queries[:5] if isinstance(query, dict)]

    def _limit(self, query: dict[str, Any]) -> int:
        try:
            number = int(query.get("limit", 3))
        except (TypeError, ValueError):
            number = 3
        return max(1, min(5, number))

    def _query_ids(self, query: dict[str, Any], plural_key: str, single_key: str) -> list[str]:
        values = []
        plural = query.get(plural_key)
        if isinstance(plural, list):
            values.extend(_text_id(value) for value in plural)
        elif plural is not None:
            values.append(_text_id(plural))
        if query.get(single_key) is not None:
            values.append(_text_id(query.get(single_key)))
        return self._unique(values)

    def _list_value(self, value: Any) -> list[str]:
        if isinstance(value, list):
            return self._unique([_text_id(item) for item in value])
        if value is None:
            return []
        return [_text_id(value)]

    def _event_types_for_context(self, context: str) -> list[str]:
        if context == "social":
            return SOCIAL_STRUCTURED_TYPES
        if context == "conversation":
            return CONVERSATION_STRUCTURED_TYPES
        if context == "opinion_assessment":
            return OPINION_STRUCTURED_TYPES
        return WORLD_STRUCTURED_TYPES

    def _semantic_query_text(self, query: dict[str, Any], fallback_query: str, task: str) -> str:
        text = _text_id(query.get("query")).strip()
        if text:
            return text
        intent = _text_id(query.get("intent"))
        return " | ".join(part for part in [fallback_query, f"task={task}" if task else "", intent] if part)

    def _plan_quotas(self, context: str, n_results: int) -> dict[str, int]:
        base = {
            "person_profile": 2,
            "entity": 3,
            "event": 3,
            "social_post": 2,
            "derived": 2,
            "semantic": 2,
        }
        if context == "social":
            base.update({"person_profile": 3, "social_post": 4, "event": 3, "semantic": 2})
        elif context == "conversation":
            base.update({"person_profile": 1, "event": 4, "semantic": 2})
        if n_results <= 3:
            base["semantic"] = 1
        return base

    def _record_derived_without_chroma(
        self,
        *,
        agent_id: str,
        memory_type: str,
        source_type: str,
        world_time: int,
        summary: str,
        payload: Any,
        entity_id: Any = None,
        importance: float = 0.5,
        confidence: float = 0.5,
    ) -> None:
        """写入不需要 embedding 的衍生记忆，例如本地生成的观念评测证据。"""

        memory_id = f"{agent_id}:{source_type}:{world_time}:{abs(hash(summary))}"
        self.store.record_derived_memory(
            memory_id=memory_id,
            agent_id=agent_id,
            memory_type=memory_type,
            source_type=source_type,
            world_time=world_time,
            summary=summary,
            payload=payload,
            entity_id=entity_id,
            importance=importance,
            confidence=confidence,
        )

    def _store_social_people_profiles(self, agent_id: str, post: dict[str, Any], world_time: int) -> None:
        """把帖子作者和评论作者登记到人物档案。"""

        author_id = _text_id(post.get("author_id"))
        if author_id and author_id != "system":
            self.store.upsert_person_profile_social(
                agent_id,
                target_agent_id=author_id,
                name=author_id,
                world_time=world_time,
                recent_post_summary=self._post_profile_summary(post),
                confidence=0.8,
            )
        for comment in post.get("comments") or []:
            if not isinstance(comment, dict):
                continue
            comment_author = _text_id(comment.get("author_id"))
            if not comment_author or comment_author == "system":
                continue
            self.store.upsert_person_profile_social(
                agent_id,
                target_agent_id=comment_author,
                name=comment_author,
                world_time=int(comment.get("time") or world_time),
                recent_post_summary=self._comment_profile_summary(post, comment),
                confidence=0.75,
            )

    def _summarize_person_profile(self, profile: dict[str, Any], facts: list[dict[str, Any]], *, llm=None) -> dict[str, Any]:
        """把新增事实压缩为人物档案印象，LLM 失败时使用规则摘要。"""

        fallback = self._rule_person_profile_summary(profile, facts)
        if llm is None or not facts:
            return fallback
        system = (
            "你是沙盒智能体的记忆整理模块。请根据某个目标人物的新事实，更新当前智能体对他的档案印象。"
            "只输出 JSON，不要输出额外文字。不要输出 evidence_event_ids。"
        )
        user = json.dumps(
            {
                "target_agent_id": profile.get("target_agent_id"),
                "old_profile": {
                    "actions_impression": profile.get("actions_impression", ""),
                    "opinion_impression": profile.get("opinion_impression", ""),
                    "relationship_impression": profile.get("relationship_impression", ""),
                    "recent_post_summary": profile.get("recent_post_summary", ""),
                },
                "new_facts": [self._fact_text(row) for row in facts],
                "output_schema": {
                    "actions_impression": "此人做过什么的中文短总结",
                    "opinion_impression": "此人持有什么观念或立场的中文短总结",
                    "relationship_impression": "当前智能体对他的总体印象",
                    "confidence": "float in [0,1]",
                },
            },
            ensure_ascii=False,
        )
        try:
            raw = llm.generate(system, user)
            data = self._parse_json_object(raw)
            return {
                "actions_impression": _short(data.get("actions_impression") or fallback["actions_impression"], 240),
                "opinion_impression": _short(data.get("opinion_impression") or fallback["opinion_impression"], 240),
                "relationship_impression": _short(data.get("relationship_impression") or fallback["relationship_impression"], 240),
                "confidence": self._clamp01(data.get("confidence", fallback["confidence"])),
            }
        except Exception as exc:
            logger.debug("person profile LLM summary failed: %s", exc)
            return fallback

    async def _asummarize_person_profile(self, profile: dict[str, Any], facts: list[dict[str, Any]], *, llm=None) -> dict[str, Any]:
        """异步压缩人物档案印象；LLM 不支持异步时使用规则摘要。"""

        fallback = self._rule_person_profile_summary(profile, facts)
        if llm is None or not facts or not hasattr(llm, "agenerate"):
            return fallback
        system = (
            "你是沙盒智能体的记忆整理模块。请根据某个目标人物的新事实，更新当前智能体对他的档案印象。"
            "只输出 JSON，不要输出额外文字。不要输出 evidence_event_ids。"
        )
        user = json.dumps(
            {
                "target_agent_id": profile.get("target_agent_id"),
                "old_profile": {
                    "actions_impression": profile.get("actions_impression", ""),
                    "opinion_impression": profile.get("opinion_impression", ""),
                    "relationship_impression": profile.get("relationship_impression", ""),
                    "recent_post_summary": profile.get("recent_post_summary", ""),
                },
                "new_facts": [self._fact_text(row) for row in facts],
                "output_schema": {
                    "actions_impression": "此人做过什么的中文短总结",
                    "opinion_impression": "此人持有什么观念或立场的中文短总结",
                    "relationship_impression": "当前智能体对他的总体印象",
                    "confidence": "float in [0,1]",
                },
            },
            ensure_ascii=False,
        )
        try:
            raw = await llm.agenerate(system, user)
            data = self._parse_json_object(raw)
            return {
                "actions_impression": _short(data.get("actions_impression") or fallback["actions_impression"], 240),
                "opinion_impression": _short(data.get("opinion_impression") or fallback["opinion_impression"], 240),
                "relationship_impression": _short(data.get("relationship_impression") or fallback["relationship_impression"], 240),
                "confidence": self._clamp01(data.get("confidence", fallback["confidence"])),
            }
        except Exception as exc:
            logger.debug("async person profile LLM summary failed: %s", exc)
            return fallback

    def _rule_person_profile_summary(self, profile: dict[str, Any], facts: list[dict[str, Any]]) -> dict[str, Any]:
        """规则兜底：用事实类型和摘要拼出人物档案印象。"""

        target_id = _text_id(profile.get("target_agent_id"))
        action_texts: list[str] = []
        opinion_texts: list[str] = []
        relation_texts: list[str] = []
        for row in facts:
            source = row.get("source_type")
            event_type = row.get("event_type")
            summary = _short(row.get("summary", ""), 120)
            payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
            if source in {"observe", "action_result", "conversation"}:
                action_texts.append(summary or f"{target_id} 参与过 {event_type}")
            if source in {"social_browse", "social_feedback"}:
                opinion = payload.get("opinion_index")
                content = payload.get("content") or payload.get("feedback") or summary
                opinion_texts.append(f"{_short(content, 80)} opinion_index={opinion}")
            if source == "conversation":
                relation_texts.append(summary)
        if not action_texts and profile.get("actions_impression"):
            action_texts.append(profile["actions_impression"])
        if not opinion_texts and profile.get("opinion_impression"):
            opinion_texts.append(profile["opinion_impression"])
        if not relation_texts and profile.get("relationship_impression"):
            relation_texts.append(profile["relationship_impression"])
        return {
            "actions_impression": _short("；".join(action_texts[:4]) or f"暂未形成对 {target_id} 行为的明确印象", 240),
            "opinion_impression": _short("；".join(opinion_texts[:4]) or f"暂未形成对 {target_id} 观念的明确印象", 240),
            "relationship_impression": _short("；".join(relation_texts[:3]) or f"对 {target_id} 的总体印象仍不明确", 240),
            "confidence": 0.65 if facts else 0.5,
        }

    def _parse_json_object(self, raw: str) -> dict[str, Any]:
        """兼容纯 JSON 和 fenced JSON block。"""

        text = str(raw or "").strip()
        if text.startswith("```"):
            text = text.removeprefix("```json").removeprefix("```").strip()
            if text.endswith("```"):
                text = text[:-3].strip()
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("person profile summary is not a JSON object")
        return data

    def _clamp01(self, value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = 0.5
        return max(0.0, min(1.0, number))

    def _fact_text(self, row: dict[str, Any]) -> str:
        return (
            f"t={row.get('world_time')} source={row.get('source_type')} "
            f"type={row.get('event_type')} summary={row.get('summary')}"
        )

    def _post_profile_summary(self, post: dict[str, Any]) -> str:
        return (
            f"posted: {_short(post.get('content', ''), 120)} "
            f"opinion_index={post.get('opinion_index')}"
        )

    def _comment_profile_summary(self, post: dict[str, Any], comment: dict[str, Any]) -> str:
        return (
            f"commented on post {post.get('id')}: "
            f"{_short(comment.get('content', ''), 120)}"
        )

    def _extract_entity_ids(self, payload: dict[str, Any], query_text: str) -> list[str]:
        """从结构化 payload 和文本查询里提取实体索引。"""

        ids: list[str] = []
        if payload:
            for person in payload.get("people") or []:
                if isinstance(person, dict):
                    ids.append(_text_id(person.get("id")))
            for obj in payload.get("objects") or []:
                if isinstance(obj, dict):
                    ids.append(_text_id(obj.get("id")))
            for action in payload.get("actions") or []:
                if isinstance(action, dict):
                    ids.extend([_text_id(action.get("actor_id")), _text_id(action.get("acted_id"))])
            if payload.get("receiver_id"):
                ids.append(_text_id(payload.get("receiver_id")))
            for post in payload.get("posts") or []:
                if isinstance(post, dict):
                    ids.append(_text_id(post.get("author_id")))
                    for comment in post.get("comments") or []:
                        if isinstance(comment, dict):
                            ids.append(_text_id(comment.get("author_id")))
        for token in query_text.replace('"', " ").replace("'", " ").replace(",", " ").split():
            if token.startswith("agent_") or token.startswith("food_") or token.startswith("post_"):
                ids.append(token.strip(" ,.:;()[]{}"))
        return self._unique(ids)

    def _extract_post_ids(self, payload: dict[str, Any], query_text: str) -> list[str]:
        """从社交 payload 和文本查询里提取帖子索引。"""

        ids: list[str] = []
        if payload:
            ids.extend(_text_id(value) for value in payload.get("visible_post_ids") or [])
            if payload.get("post_id") is not None:
                ids.append(_text_id(payload.get("post_id")))
            for post in payload.get("posts") or []:
                if isinstance(post, dict):
                    ids.append(_text_id(post.get("id")))
        for token in query_text.replace('"', " ").replace("'", " ").replace(",", " ").split():
            cleaned = token.strip(" ,.:;()[]{}")
            if cleaned.isdigit():
                ids.append(cleaned)
        return self._unique(ids)

    def _unique(self, values: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for value in values:
            if not value or value in seen:
                continue
            seen.add(value)
            out.append(value)
        return out

    def _dedupe(self, values: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for value in values:
            if not value or value in seen:
                continue
            seen.add(value)
            out.append(value)
        return out

    def _scene_summary(self, data: dict[str, Any]) -> str:
        return (
            f"t={data.get('time')} scene at {data.get('position')} "
            f"people={len(data.get('people') or [])} objects={len(data.get('objects') or [])} "
            f"actions={len(data.get('actions') or [])}"
        )

    def _person_summary(self, person: dict[str, Any], world_time: int) -> str:
        return f"t={world_time} person {person.get('id')} at {person.get('position')} region={person.get('region')}"

    def _object_summary(self, obj: dict[str, Any], world_time: int) -> str:
        return (
            f"t={world_time} object {obj.get('id')} kind={obj.get('kind')} "
            f"at {obj.get('position')} region={obj.get('region')}"
        )

    def _action_summary(self, action: dict[str, Any]) -> str:
        return (
            f"t={action.get('time')} {action.get('actor_id')} did {action.get('type')} "
            f"to {action.get('acted_id')}: {_short(action.get('info', ''))}"
        )

    def _action_result_summary(self, agent_id: str, decision: dict[str, Any], feedback: Any, reward: Any, world_time: int) -> str:
        action = decision.get("action") if isinstance(decision.get("action"), dict) else {}
        return (
            f"t={world_time} {agent_id} chose {action.get('tool', 'no_action')} "
            f"think={_short(decision.get('think', ''))} feedback={_short(feedback)} reward={reward}"
        )

    def _social_feedback_summary(self, data: dict[str, Any], world_time: int) -> str:
        post_id = data.get("post_id")
        return (
            f"t={world_time} social action={data.get('action')} post_id={post_id} "
            f"ok={data.get('ok')} feedback={_short(data.get('feedback', ''))}"
        )

    def _format_entity_state(self, row: dict[str, Any]) -> str:
        return (
            f"[state t={row.get('last_seen_at')} confidence={row.get('confidence')}] "
            f"{row.get('entity_type')} {row.get('entity_id')} at {row.get('position')} "
            f"region={row.get('region')}"
        )

    def _format_person_profile(self, row: dict[str, Any]) -> str:
        return (
            f"[person_profile t={max(int(row.get('last_seen_at') or 0), int(row.get('last_social_seen_at') or 0))} "
            f"confidence={row.get('confidence')}] target={row.get('target_agent_id')} "
            f"last_position={row.get('last_position')} region={row.get('last_region')} "
            f"actions={_short(row.get('actions_impression', ''))} "
            f"opinions={_short(row.get('opinion_impression', ''))} "
            f"impression={_short(row.get('relationship_impression', ''))} "
            f"recent_social={_short(row.get('recent_post_summary', ''))}"
        )

    def _format_relation(self, row: dict[str, Any]) -> str:
        return (
            f"[relation t={row.get('last_seen_at')} confidence={row.get('confidence')}] "
            f"{row.get('subject_id')} {row.get('relation_type')} {row.get('object_id')}"
        )

    def _format_social_post(self, row: dict[str, Any]) -> str:
        return (
            f"[social t={row.get('last_seen_at')} post={row.get('post_id')} confidence={row.get('confidence')}] "
            f"author={row.get('author_id')} likes={row.get('likes')} dislikes={row.get('dislikes')} "
            f"comments={row.get('comments_count')} content={_short(row.get('content', ''))}"
        )

    def _format_event(self, row: dict[str, Any]) -> str:
        memory_type = row.get("memory_type") or "episodic"
        return (
            f"[{memory_type} t={row.get('world_time')} source={row.get('source_type')} "
            f"confidence={row.get('confidence')}] {row.get('summary')}"
        )

    def _format_derived(self, row: dict[str, Any]) -> str:
        memory_type = row.get("memory_type") or "episodic"
        task = f" task={row.get('task')}" if row.get("task") else ""
        need = f" need={row.get('need_key')}" if row.get("need_key") else ""
        return (
            f"[{memory_type} t={row.get('world_time')}{task}{need} "
            f"confidence={row.get('confidence')}] {row.get('summary')}"
        )
