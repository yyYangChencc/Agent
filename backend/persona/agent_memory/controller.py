from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import Any

from persona.agent_memory.query_builder import build_semantic_observation_text
from persona.agent_memory.structured_store import StructuredMemoryStore
from persona.llm.interface import JSON_OBJECT_RESPONSE_FORMAT
from persona.llm.debug_trace import annotate_current_llm_trace, trace_llm_call
from persona.llm.json_utils import parse_json_object
from persona.logger import get_logger

logger = get_logger(__name__)


WORLD_STRUCTURED_TYPES = ["semantic", "procedural", "episodic", "reflective", "social"]
SOCIAL_STRUCTURED_TYPES = ["social", "semantic", "episodic", "reflective"]
CONVERSATION_STRUCTURED_TYPES = ["social", "episodic", "reflective", "semantic"]
OPINION_STRUCTURED_TYPES = ["social", "semantic", "episodic", "reflective"]
MAX_QUERY_LIST_ITEMS = 10
MAX_QUERY_FIELD_CHARS = 120


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
        episode_id = _text_id(data.get("episode_id"))
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
            episode_id=episode_id,
            provenance="direct_observation",
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
                episode_id=episode_id,
                provenance="direct_observation",
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
                episode_id=episode_id,
                provenance="direct_observation",
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
                episode_id=episode_id,
                provenance="direct_observation",
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
        notification_events = social.get("notification_events")
        notes = notification_events if isinstance(notification_events, list) else social.get("notifications") or []
        for note in notes:
            if not isinstance(note, dict):
                continue
            self.store.record_event(
                agent_id,
                memory_type="social",
                source_type="observe",
                event_type="notification",
                world_time=int(note.get("time") or world_time),
                related_agent_id=note.get("actor_id") or note.get("source_author_id"),
                entity_id=note.get("actor_id") or note.get("source_author_id") or note.get("post_id"),
                object_id=note.get("source_post_id"),
                post_id=note.get("post_id") or note.get("source_post_id"),
                payload=note,
                summary=f"t={note.get('time', world_time)} social notification: {_short(note.get('content', ''))}",
                episode_id=_text_id(note.get("episode_id")) or episode_id,
                platform_event_id=note.get("platform_event_id") or note.get("event_id"),
                feed_request_id=note.get("feed_request_id"),
                comment_id=note.get("comment_id"),
                parent_comment_id=note.get("parent_comment_id"),
                root_comment_id=note.get("root_comment_id"),
                topic=note.get("topic"),
                provenance="platform_notification",
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
        episode_id: str = "",
        state_before: dict[str, Any] | None = None,
        state_after: dict[str, Any] | None = None,
        outcome: dict[str, Any] | None = None,
        need_events: list[dict[str, Any]] | None = None,
    ) -> None:
        """写入 World.execute 后的决策结果。

        action 的真实反馈和 reward 只有执行阶段才知道，因此动作记忆在 World.execute
        之后写入，避免记录未发生的计划。
        """

        decision_data = _as_dict(decision)
        action = decision_data.get("action") if isinstance(decision_data.get("action"), dict) else {}
        tool = _text_id(action.get("tool"))
        args = action.get("args") if isinstance(action.get("args"), dict) else {}
        feedback_data = _as_dict(feedback)
        need_event_rows = [item for item in need_events or [] if isinstance(item, dict)]
        outcome_data = dict(outcome or {})
        if not outcome_data.get("status"):
            outcome_data["status"] = self._action_status(tool, feedback)
        payload = {
            "decision": decision_data,
            "feedback": feedback,
            "reward": reward,
            "state_before": dict(state_before or {}),
            "state_after": dict(state_after or {}),
            "outcome": outcome_data,
            "need_events": need_event_rows,
        }
        related_agent_id = feedback_data.get("target_agent_id") or feedback_data.get("source_author_id")
        acted = args.get("ID") or args.get("post_id") or related_agent_id
        post = feedback_data.get("post") if isinstance(feedback_data.get("post"), dict) else {}
        event_id = self.store.record_event(
            agent_id,
            memory_type="episodic",
            source_type="action_result",
            event_type=tool or "no_action",
            world_time=world_time,
            entity_id=acted,
            object_id=acted,
            related_agent_id=related_agent_id,
            post_id=feedback_data.get("post_id") or args.get("post_id"),
            payload=payload,
            summary=self._action_result_summary(agent_id, decision_data, feedback, reward, world_time),
            episode_id=episode_id,
            platform_event_id=feedback_data.get("platform_event_id") or feedback_data.get("event_id"),
            feed_request_id=feedback_data.get("feed_request_id"),
            comment_id=feedback_data.get("comment_id"),
            parent_comment_id=feedback_data.get("parent_comment_id"),
            root_comment_id=feedback_data.get("root_comment_id"),
            topic=post.get("topic") or feedback_data.get("post_topic"),
            before_state=state_before,
            after_state=state_after,
            outcome=outcome_data,
            provenance="self_action",
            importance=self._experience_importance(tool, reward, need_event_rows, outcome_data),
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
        self._consolidate_action_experience(
            agent_id,
            event_id=event_id,
            episode_id=episode_id,
            world_time=world_time,
            tool=tool,
            args=args,
            feedback=feedback,
            reward=reward,
            state_before=state_before or {},
            state_after=state_after or {},
            outcome=outcome_data,
            need_events=need_event_rows,
        )

    def store_need_event(self, agent_id: str, event: Any, *, episode_id: str = "") -> None:
        """把需求增减保存为可按需求、人物和经历链查询的原始事件。"""

        data = _as_dict(event)
        if not data:
            return
        evidence = data.get("evidence") if isinstance(data.get("evidence"), dict) else {}
        need_key = _text_id(data.get("need_key"))
        related_agent_id = (
            evidence.get("operator_id")
            or evidence.get("sender")
            or evidence.get("author_id")
            or data.get("sender")
        )
        delta = self._float_value(data.get("delta"), 0.0)
        world_time = int(data.get("tick") or 0)
        self.store.record_event(
            agent_id,
            memory_type="episodic",
            source_type="need_event",
            event_type="need_delta",
            world_time=world_time,
            entity_id=need_key,
            related_agent_id=related_agent_id,
            post_id=evidence.get("post_id"),
            payload=data,
            summary=(
                f"t={world_time} need={need_key} before={data.get('before')} "
                f"after={data.get('after')} delta={delta} reason={_short(data.get('reason', ''))}"
            ),
            episode_id=episode_id or _text_id(data.get("episode_id")),
            platform_event_id=evidence.get("platform_event_id") or evidence.get("event_id"),
            feed_request_id=evidence.get("feed_request_id"),
            topic=evidence.get("topic") or data.get("topic"),
            before_state={"satisfaction": {need_key: data.get("before")}},
            after_state={"satisfaction": {need_key: data.get("after")}},
            outcome={"delta": delta, "source": data.get("source")},
            provenance=_text_id(data.get("source")) or "need_system",
            importance=min(0.9, 0.5 + min(0.4, abs(delta) / 10.0)),
            confidence=0.95,
        )

    def store_psychological_assessment(
        self,
        agent_id: str,
        assessment: Any,
        *,
        episode_id: str = "",
    ) -> None:
        """保存心理评测；其内容是模型推断，不能覆盖直接观察事实。"""

        data = _as_dict(assessment)
        if not data:
            return
        world_time = int(data.get("tick") or 0)
        activated = [str(value) for value in data.get("activated_needs") or []]
        status = _text_id(data.get("status")) or "unknown"
        summary = f"t={world_time} psychological status={status} activated={','.join(activated)}"
        event_id = self.store.record_event(
            agent_id,
            memory_type="reflective",
            source_type="psychological_assessment",
            event_type="psychological_assessment",
            world_time=world_time,
            entity_id=activated[0] if len(activated) == 1 else None,
            payload=data,
            summary=summary,
            episode_id=episode_id,
            provenance="model_assessment",
            importance=0.65 if activated else 0.45,
            confidence=self._clamp01(data.get("confidence", 0.6)),
        )
        if status != "skipped" and (activated or data.get("role_card_delta")):
            self._record_derived_without_chroma(
                agent_id=agent_id,
                memory_type="reflective",
                source_type="psychological_assessment",
                world_time=world_time,
                summary=summary,
                payload=data,
                entity_id=activated[0] if len(activated) == 1 else None,
                episode_id=episode_id,
                evidence_event_ids=[event_id],
                importance=0.65,
                confidence=self._clamp01(data.get("confidence", 0.6)),
            )

    def store_social_browse(self, agent_id: str, browse_payload: Any) -> None:
        """写入一次社交平台浏览结果，包括可见帖子列表和帖子快照。"""

        data = _as_dict(browse_payload)
        if not data:
            return
        world_time = int(data.get("time") or 0)
        episode_id = _text_id(data.get("episode_id"))
        visible_ids = data.get("visible_post_ids") or []
        self.store.record_event(
            agent_id,
            memory_type="social",
            source_type="social_browse",
            event_type="browse_posts",
            world_time=world_time,
            payload=data,
            summary=f"t={world_time} browsed social posts: {', '.join(str(x) for x in visible_ids)}",
            episode_id=episode_id,
            platform_event_id=data.get("platform_event_id") or data.get("event_id"),
            feed_request_id=data.get("feed_request_id"),
            provenance="platform_exposure",
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
        episode_id = _text_id(data.get("episode_id"))
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
        if action == "reply_comment":
            related_agent_id = data.get("target_agent_id") or author_id
        elif action in {"repost_post", "quote_post"}:
            related_agent_id = (post or {}).get("source_author_id") or data.get("source_author_id") or data.get("target_agent_id")
        else:
            related_agent_id = author_id or data.get("target_agent_id") or data.get("source_author_id")
        topic = (post or {}).get("topic") or data.get("post_topic") or data.get("topic")
        comment_id = data.get("comment_id")
        comment = next(
            (
                item
                for item in (post or {}).get("comments") or []
                if isinstance(item, dict) and _text_id(item.get("id")) == _text_id(comment_id)
            ),
            {},
        )
        parent_comment_id = data.get("parent_comment_id") or comment.get("parent_comment_id")
        root_comment_id = data.get("root_comment_id") or comment.get("root_comment_id")
        event_id = self.store.record_event(
            agent_id,
            memory_type="social",
            source_type="social_feedback",
            event_type=action,
            world_time=event_time,
            post_id=post_id,
            related_agent_id=related_agent_id,
            entity_id=related_agent_id or post_id,
            object_id=data.get("source_post_id"),
            payload=data,
            summary=self._social_feedback_summary(data, event_time),
            episode_id=episode_id,
            platform_event_id=data.get("platform_event_id") or data.get("event_id"),
            feed_request_id=data.get("feed_request_id"),
            comment_id=comment_id,
            parent_comment_id=parent_comment_id,
            root_comment_id=root_comment_id,
            topic=topic,
            outcome={
                "ok": bool(data.get("ok")),
                "state_changed": data.get("state_changed"),
                "feedback": data.get("feedback"),
            },
            provenance="self_social_action",
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
        if related_agent_id and related_agent_id not in {agent_id, "system"}:
            self.store.upsert_person_profile_social(
                agent_id,
                target_agent_id=related_agent_id,
                name=_text_id(related_agent_id),
                world_time=event_time,
                recent_post_summary=self._social_feedback_summary(data, event_time),
                confidence=0.8,
            )
        if action in {"follow_author", "unfollow_author"} and related_agent_id and data.get("ok"):
            # 原始事件保留关注变化历史，follows 关系只表达当前边状态。
            self.store.upsert_relation(
                agent_id,
                subject_id=agent_id,
                relation_type="follows",
                object_id=related_agent_id,
                source_type="social_feedback",
                world_time=event_time,
                payload=data,
                importance=0.7,
                confidence=0.95,
                valid=action == "follow_author",
            )
        if action in {"repost_post", "quote_post"} and related_agent_id:
            self.store.upsert_relation(
                agent_id,
                subject_id=agent_id,
                relation_type=f"{action}_author",
                object_id=related_agent_id,
                source_type="social_feedback",
                world_time=event_time,
                payload=data,
                importance=0.65,
                confidence=0.85,
            )
        if related_agent_id and data.get("online_trust_after") is not None:
            self.store.upsert_relation(
                agent_id,
                subject_id=agent_id,
                relation_type="online_trust",
                object_id=related_agent_id,
                source_type="social_feedback",
                world_time=event_time,
                payload={
                    "before": data.get("online_trust_before"),
                    "after": data.get("online_trust_after"),
                    "action": action,
                },
                importance=0.7,
                confidence=0.95,
            )
        self._consolidate_social_experience(
            agent_id,
            event_id=event_id,
            episode_id=episode_id,
            world_time=event_time,
            action=action,
            data=data,
            related_agent_id=_text_id(related_agent_id),
            post_id=post_id,
            topic=_text_id(topic),
        )

    def store_conversation(
        self,
        agent_id: str,
        *,
        messages: list[dict[str, Any]] | None = None,
        reply: Any = None,
        observation: str = "",
        world_time: int = 0,
        episode_id: str = "",
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
        evidence_event_ids: list[int] = []
        related_agent_ids: list[str] = []
        for message in messages or []:
            if not isinstance(message, dict):
                continue
            sender = message.get("sender")
            event_id = self.store.record_event(
                agent_id,
                memory_type="social",
                source_type="conversation",
                event_type="message_received",
                world_time=int(message.get("time") or world_time),
                related_agent_id=sender,
                entity_id=sender,
                payload=message,
                summary=f"t={message.get('time', world_time)} {sender} said: {_short(message.get('content', ''))}",
                episode_id=episode_id,
                topic=message.get("topic"),
                provenance="direct_conversation",
                importance=0.65,
                confidence=0.8,
            )
            evidence_event_ids.append(event_id)
            if sender:
                related_agent_ids.append(_text_id(sender))
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
            reply_data = _as_dict(reply)
            target_id = ((reply_data.get("action") or {}).get("args") or {}).get("ID") if reply_data else None
            event_id = self.store.record_event(
                agent_id,
                memory_type="social",
                source_type="conversation",
                event_type="conversation_reply",
                world_time=world_time,
                related_agent_id=target_id,
                entity_id=target_id,
                payload=payload,
                summary=f"t={world_time} conversation reply: {_short(reply)}",
                episode_id=episode_id,
                topic=(((reply_data.get("action") or {}).get("args") or {}).get("topic") if reply_data else None),
                provenance="self_conversation_action",
                importance=0.6,
                confidence=0.75,
            )
            evidence_event_ids.append(event_id)
            if target_id:
                related_agent_ids.append(_text_id(target_id))
        if evidence_event_ids:
            self._consolidate_conversation_experience(
                agent_id,
                episode_id=episode_id,
                world_time=world_time,
                payload=payload,
                evidence_event_ids=evidence_event_ids,
                related_agent_ids=self._unique(related_agent_ids),
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
        episode_id = _text_id(data.get("episode_id"))
        event_id = self.store.record_event(
            agent_id,
            memory_type="reflective",
            source_type="opinion_assessment",
            event_type="opinion_assessment",
            world_time=world_time,
            entity_id=topic,
            payload=data,
            summary=summary,
            episode_id=episode_id,
            topic=topic,
            provenance="opinion_model_assessment",
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
            episode_id=episode_id,
            evidence_event_ids=[event_id],
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
            episode_id=metadata.get("episode_id"),
            evidence_event_ids=metadata.get("evidence_event_ids"),
            valid_from=metadata.get("valid_from", world_time),
            valid_to=metadata.get("valid_to"),
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
        timeout_seconds: float | None = None,
    ) -> list[str]:
        """同步检索入口：先按场景查 SQLite，再用剩余预算查 Chroma。"""

        started_at = time.perf_counter()
        payload = _as_dict(observation)
        query_text = observation if isinstance(observation, str) else _as_text(observation)
        semantic_query_text = build_semantic_observation_text(observation, context)
        # 至少为相似长期经历保留一个位置，避免结构化结果先占满全部配额。
        structured_limit = n_results if n_results <= 1 else n_results - 1
        structured = self._retrieve_structured(agent_id, payload, query_text, task, context, structured_limit)
        if self._retrieval_timed_out(started_at, timeout_seconds):
            # 结构化 SQLite 已有结果时，超时后不再继续 Chroma，避免慢 embedding 阻塞 tick。
            return self._dedupe(structured)[:n_results]
        semantic_budget = max(1, n_results - len(structured))
        semantic = self.semantic_manager.smart_retrieve(
            agent_id,
            semantic_query_text,
            task,
            urgency,
            satisfaction_threshold,
            n_results=semantic_budget,
            context=context,
            current_time=int(payload.get("time") or 0) if payload else 0,
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
        timeout_seconds: float | None = None,
    ) -> list[str]:
        """异步检索入口；结构化查询是本地 SQLite，语义查询走异步 Chroma 路径。"""

        started_at = time.perf_counter()
        payload = _as_dict(observation)
        query_text = observation if isinstance(observation, str) else _as_text(observation)
        semantic_query_text = build_semantic_observation_text(observation, context)
        # 异步路径使用相同的结构化与语义配额。
        structured_limit = n_results if n_results <= 1 else n_results - 1
        structured = self._retrieve_structured(agent_id, payload, query_text, task, context, structured_limit)
        if self._retrieval_timed_out(started_at, timeout_seconds):
            # 结构化 SQLite 已有结果时，超时后不再继续 Chroma，避免慢 embedding 阻塞 tick。
            return self._dedupe(structured)[:n_results]
        semantic_budget = max(1, n_results - len(structured))
        semantic_call = self.semantic_manager.asmart_retrieve(
            agent_id,
            semantic_query_text,
            task,
            urgency,
            satisfaction_threshold,
            n_results=semantic_budget,
            context=context,
            current_time=int(payload.get("time") or 0) if payload else 0,
        )
        remaining_timeout = self._remaining_timeout(started_at, timeout_seconds)
        try:
            if remaining_timeout is None:
                semantic = await semantic_call
            else:
                semantic = await asyncio.wait_for(semantic_call, timeout=remaining_timeout)
        except asyncio.TimeoutError:
            logger.warning("[%s] memory semantic retrieval timeout context=%s", agent_id, context)
            semantic = []
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
        timeout_seconds: float | None = None,
    ) -> list[str]:
        """执行 LLM planner 生成的受控记忆查询计划。

        planner 只表达信息需求；本函数负责把查询计划映射到固定 SQLite/Chroma
        查询函数，并强制过滤当前 observe 已经看见的物品最新状态。
        """

        plan = _as_dict(query_plan)
        payload = _as_dict(observation)
        query_text = observation if isinstance(observation, str) else _as_text(observation)
        semantic_query_text = build_semantic_observation_text(observation, context)
        if not plan or not plan.get("queries"):
            return self.retrieve_context(
                agent_id,
                observation,
                task,
                urgency,
                satisfaction_threshold,
                n_results=n_results,
                context=context,
                timeout_seconds=timeout_seconds,
            )
        return self._execute_query_plan_sync(
            agent_id,
            plan,
            payload,
            query_text,
            semantic_query_text,
            task,
            urgency,
            satisfaction_threshold,
            n_results=n_results,
            context=context,
            started_at=time.perf_counter(),
            timeout_seconds=timeout_seconds,
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
        timeout_seconds: float | None = None,
    ) -> list[str]:
        """异步执行受控记忆查询计划；语义查询走异步 Chroma。"""

        plan = _as_dict(query_plan)
        payload = _as_dict(observation)
        query_text = observation if isinstance(observation, str) else _as_text(observation)
        semantic_query_text = build_semantic_observation_text(observation, context)
        if not plan or not plan.get("queries"):
            return await self.aretrieve_context(
                agent_id,
                observation,
                task,
                urgency,
                satisfaction_threshold,
                n_results=n_results,
                context=context,
                timeout_seconds=timeout_seconds,
            )
        return await self._execute_query_plan_async(
            agent_id,
            plan,
            payload,
            query_text,
            semantic_query_text,
            task,
            urgency,
            satisfaction_threshold,
            n_results=n_results,
            context=context,
            started_at=time.perf_counter(),
            timeout_seconds=timeout_seconds,
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
            with trace_llm_call(
                llm,
                agent_id=agent_id,
                tick=current_time,
                stage="person_profile_summary",
                metadata={"target_agent_id": target_id},
            ):
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
            with trace_llm_call(
                llm,
                agent_id=agent_id,
                tick=current_time,
                stage="person_profile_summary",
                metadata={"target_agent_id": target_id},
            ):
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

    def maintain_agent_memory(
        self,
        agent_id: str,
        *,
        current_time: int,
        raw_event_retention: int = 100,
        max_prunable_importance: float = 0.5,
        active_event_limit: int = 2000,
        protected_importance: float = 0.7,
    ) -> dict[str, Any]:
        """执行低价值事件失效和活跃事件数量控制。"""

        prune_before = max(0, int(current_time) - max(1, int(raw_event_retention)))
        pruned_events = self.store.prune_low_value_events(
            agent_id,
            before_time=prune_before,
            max_importance=max_prunable_importance,
        )
        pruned_to_limit = self.store.prune_events_to_limit(
            agent_id,
            active_limit=active_event_limit,
            protected_importance=protected_importance,
        )
        unresolved_conflicts = self.store.get_unresolved_conflicts(agent_id)
        return {
            "agent_id": agent_id,
            "pruned_low_value_events": pruned_events,
            "pruned_events_to_limit": pruned_to_limit,
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
        visible_refs = self._visible_refs(payload)
        visible_entity_ids = set(visible_refs["people"] + visible_refs["objects"])
        visible_post_ids = set(visible_refs["posts"])
        rows: list[tuple[str, str, Any]] = []

        if context == "social":
            # 社交决策优先看当前可见帖子，再补充近期社交反馈和浏览记录。
            for row in self.store.get_person_profiles(agent_id, entity_ids, limit=8):
                rows.append(("person_profile", self._format_person_profile(row), row.get("target_agent_id")))
            for row in self.store.get_recent_events(
                agent_id,
                source_types=["social_feedback", "social_browse", "observe"],
                memory_types=SOCIAL_STRUCTURED_TYPES,
                entity_ids=entity_ids,
                post_ids=post_ids,
                limit=8,
            ):
                if row.get("source_type") == "social_browse" and int(row.get("world_time") or 0) == world_time:
                    continue
                rows.append(("event", self._format_event(row), row.get("id")))
            for row in self.store.get_recent_social_posts(agent_id, limit=5):
                if _text_id(row.get("post_id")) in visible_post_ids:
                    continue
                rows.append(("social_post", self._format_social_post(row), row.get("post_id")))
            for row in self.store.get_recent_derived(
                agent_id,
                memory_types=["social", "episodic", "reflective"],
                limit=5,
            ):
                rows.append(("derived", self._format_derived(row), row.get("memory_id")))
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
            for row in self.store.get_recent_derived(
                agent_id,
                memory_types=["social", "episodic", "reflective"],
                limit=4,
            ):
                rows.append(("derived", self._format_derived(row), row.get("memory_id")))
        elif context == "opinion_assessment":
            # 观念评测不依赖单个实体 id，侧重近期社交/对话/评测证据。
            for row in self.store.get_recent_events(
                agent_id,
                source_types=["social_feedback", "social_browse", "conversation"],
                memory_types=OPINION_STRUCTURED_TYPES,
                limit=12,
            ):
                rows.append(("event", self._format_event(row), row.get("id")))
            for row in self.store.get_recent_social_posts(agent_id, limit=8):
                rows.append(("social_post", self._format_social_post(row), row.get("post_id")))
            for row in self.store.get_recent_derived(
                agent_id,
                memory_types=["social", "episodic", "reflective"],
                limit=6,
            ):
                # 旧观念评测不能作为下一轮评测的历史背景，避免自我强化。
                if row.get("source_type") == "opinion_assessment":
                    continue
                rows.append(("derived", self._format_derived(row), row.get("memory_id")))
        else:
            # 默认世界行动查询：当前实体状态、近期经历、长期衍生记忆共同参与。
            for row in self.store.get_person_profiles(agent_id, entity_ids, limit=8):
                rows.append(("person_profile", self._format_person_profile(row), row.get("target_agent_id")))
            for row in self.store.get_entity_states(agent_id, entity_ids, limit=10):
                if _text_id(row.get("entity_id")) in visible_entity_ids:
                    continue
                rows.append(("entity", self._format_entity_state(row), row.get("entity_id")))
            for row in self.store.get_recent_events(
                agent_id,
                source_types=[
                    "observe",
                    "action_result",
                    "conversation",
                    "need_event",
                    "psychological_assessment",
                ],
                memory_types=WORLD_STRUCTURED_TYPES,
                entity_ids=entity_ids,
                limit=8,
            ):
                if row.get("source_type") == "observe" and int(row.get("world_time") or 0) == world_time:
                    continue
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
                    if _text_id(row.get("entity_id")) in visible_entity_ids:
                        continue
                    rows.append(("entity", self._format_entity_state(row), row.get("entity_id")))

        formatted: list[str] = []
        selected_rows = self._select_rows_by_quota(rows, context=context, n_results=n_results)
        for kind, text, ref in selected_rows:
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
        semantic_query_text: str,
        task: str,
        urgency: dict,
        satisfaction_threshold: dict,
        *,
        n_results: int,
        context: str,
        started_at: float | None = None,
        timeout_seconds: float | None = None,
    ) -> list[str]:
        """同步执行 planner 查询计划。"""

        rows = []
        visible = self._visible_refs(payload)
        world_time = int(payload.get("time") or 0) if payload else 0
        started_at = time.perf_counter() if started_at is None else started_at
        for query in self._limited_queries(plan):
            if self._retrieval_timed_out(started_at, timeout_seconds):
                # 超时后保留已经取得的结构化/语义结果，不继续发起新的检索。
                break
            if query.get("type") == "semantic":
                if self._retrieval_timed_out(started_at, timeout_seconds):
                    break
                semantic_query = self._semantic_query_text(query, semantic_query_text)
                results = self.semantic_manager.smart_retrieve(
                    agent_id,
                    semantic_query,
                    task,
                    urgency,
                    satisfaction_threshold,
                    n_results=self._limit(query),
                    context=context,
                    memory_types=self._list_value(query.get("memory_types")),
                    current_time=world_time,
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
        semantic_query_text: str,
        task: str,
        urgency: dict,
        satisfaction_threshold: dict,
        *,
        n_results: int,
        context: str,
        started_at: float | None = None,
        timeout_seconds: float | None = None,
    ) -> list[str]:
        """异步执行 planner 查询计划，保留 Chroma 语义召回入口。"""

        rows = []
        visible = self._visible_refs(payload)
        world_time = int(payload.get("time") or 0) if payload else 0
        started_at = time.perf_counter() if started_at is None else started_at
        for query in self._limited_queries(plan):
            if self._retrieval_timed_out(started_at, timeout_seconds):
                # 超时后保留已经取得的结构化/语义结果，不继续发起新的检索。
                break
            if query.get("type") == "semantic":
                semantic_query = self._semantic_query_text(query, semantic_query_text)
                semantic_call = self.semantic_manager.asmart_retrieve(
                    agent_id,
                    semantic_query,
                    task,
                    urgency,
                    satisfaction_threshold,
                    n_results=self._limit(query),
                    context=context,
                    memory_types=self._list_value(query.get("memory_types")),
                    current_time=world_time,
                )
                remaining_timeout = self._remaining_timeout(started_at, timeout_seconds)
                try:
                    if remaining_timeout is None:
                        results = await semantic_call
                    else:
                        results = await asyncio.wait_for(semantic_call, timeout=remaining_timeout)
                except asyncio.TimeoutError:
                    logger.warning("[%s] memory planner semantic query timeout context=%s", agent_id, context)
                    results = []
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
                    kinds=self._list_value(query.get("kinds")),
                    exclude_entity_ids=visible["objects"] if exclude_visible else [],
                    limit=limit,
                )
            for row in rows:
                out.append(("entity", self._format_entity_state(row), row.get("entity_id")))
        elif query_type == "event_history":
            for row in self.store.get_recent_events(
                agent_id,
                source_types=self._source_types_for_query(query, context),
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

        ordered_kinds = self._memory_kind_order(context)
        selected: list[tuple[str, str, Any]] = []
        semantic_rows = grouped.get("semantic", [])
        semantic_reserve = min(
            quotas.get("semantic", 0),
            len(semantic_rows),
            1 if n_results > 1 else 0,
        )
        structured_budget = max(0, n_results - semantic_reserve)
        for kind in ordered_kinds:
            if kind == "semantic":
                continue
            quota = quotas.get(kind, 0)
            for text, ref in grouped.get(kind, [])[:quota]:
                selected.append((kind, text, ref))
                if len(selected) >= structured_budget:
                    break
            if len(selected) >= structured_budget:
                break
        if len(selected) < structured_budget:
            for kind in ordered_kinds:
                if kind == "semantic":
                    continue
                for text, ref in grouped.get(kind, []):
                    if any(existing == text for _, existing, _ in selected):
                        continue
                    selected.append((kind, text, ref))
                    if len(selected) >= structured_budget:
                        break
                if len(selected) >= structured_budget:
                    break
        for text, ref in semantic_rows[:semantic_reserve]:
            selected.append(("semantic", text, ref))

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

    def _retrieval_timed_out(self, started_at: float, timeout_seconds: float | None) -> bool:
        """记忆检索超过预算后停止追加慢查询，避免单个 tick 被拖住。"""

        if timeout_seconds is None or timeout_seconds <= 0:
            return False
        return time.perf_counter() - started_at >= timeout_seconds

    def _remaining_timeout(self, started_at: float, timeout_seconds: float | None) -> float | None:
        """计算异步 Chroma 查询还能等待多久。"""

        if timeout_seconds is None or timeout_seconds <= 0:
            return None
        return max(0.001, timeout_seconds - (time.perf_counter() - started_at))

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
        return self._bounded_query_values(values)

    def _list_value(self, value: Any) -> list[str]:
        if isinstance(value, list):
            return self._bounded_query_values(value)
        if value is None:
            return []
        return self._bounded_query_values([value])

    def _bounded_query_values(self, values) -> list[str]:
        """限制 planner 列表参数，防止直接调用绕过解析器约束。"""

        bounded = []
        seen = set()
        for value in values:
            text = _text_id(value).strip()
            if not text or len(text) > MAX_QUERY_FIELD_CHARS or text in seen:
                continue
            seen.add(text)
            bounded.append(text)
            if len(bounded) >= MAX_QUERY_LIST_ITEMS:
                break
        return bounded

    def _source_types_for_query(self, query: dict[str, Any], context: str) -> list[str]:
        """观念评测查询固定排除旧评测结果，避免自我强化回音。"""

        values = self._list_value(query.get("source_types"))
        if context == "opinion_assessment":
            allowed = ["social_feedback", "social_browse", "conversation"]
            if not values:
                return allowed
            values = [value for value in values if value in allowed]
            return values or allowed
        return values

    def _event_types_for_context(self, context: str) -> list[str]:
        if context == "social":
            return SOCIAL_STRUCTURED_TYPES
        if context == "conversation":
            return CONVERSATION_STRUCTURED_TYPES
        if context == "opinion_assessment":
            return OPINION_STRUCTURED_TYPES
        return WORLD_STRUCTURED_TYPES

    def _semantic_query_text(self, query: dict[str, Any], fallback_query: str) -> str:
        """优先使用 planner 的聚焦查询，否则使用同场景的精简基础查询。"""

        text = _text_id(query.get("query")).strip()
        if text:
            return text
        intent = _text_id(query.get("intent"))
        return " | ".join(part for part in [fallback_query, intent] if part)

    def _plan_quotas(self, context: str, n_results: int) -> dict[str, int]:
        base = {
            "person_profile": 2,
            "entity": 3,
            "relation": 2,
            "event": 3,
            "social_post": 2,
            "derived": 2,
            "semantic": 2,
        }
        if context == "social":
            base.update({"person_profile": 3, "social_post": 4, "event": 3, "semantic": 2})
        elif context == "conversation":
            base.update({"person_profile": 1, "event": 4, "semantic": 2})
        elif context == "opinion_assessment":
            # 观念评测优先历史社交证据，并保留反思与语义背景。
            base.update({
                "person_profile": 0,
                "entity": 0,
                "relation": 0,
                "event": 2,
                "social_post": 1,
                "derived": 2,
                "semantic": 1,
            })
        if n_results <= 3:
            base["semantic"] = 1
        return base

    def _memory_kind_order(self, context: str) -> list[str]:
        """返回各场景在提示词中的记忆类型优先顺序。"""

        if context == "social":
            return ["social_post", "person_profile", "event", "relation", "derived", "entity", "semantic"]
        if context == "conversation":
            return ["person_profile", "relation", "event", "derived", "social_post", "entity", "semantic"]
        if context == "opinion_assessment":
            return ["event", "social_post", "derived", "semantic", "person_profile", "relation", "entity"]
        return ["entity", "event", "derived", "person_profile", "relation", "social_post", "semantic"]

    def _select_rows_by_quota(
        self,
        rows: list[tuple[str, str, Any]],
        *,
        context: str,
        n_results: int,
    ) -> list[tuple[str, str, Any]]:
        """按场景类型配额选择真正进入提示词的结构化记忆。"""

        grouped: dict[str, list[tuple[str, str, Any]]] = {}
        for kind, text, ref in rows:
            if text:
                grouped.setdefault(kind, []).append((kind, text, ref))
        quotas = self._plan_quotas(context, n_results)
        ordered_kinds = self._memory_kind_order(context)
        selected: list[tuple[str, str, Any]] = []
        for kind in ordered_kinds:
            for row in grouped.get(kind, [])[: quotas.get(kind, 0)]:
                if any(existing[1] == row[1] for existing in selected):
                    continue
                selected.append(row)
                if len(selected) >= n_results:
                    return selected
        for kind in ordered_kinds:
            for row in grouped.get(kind, []):
                if any(existing[1] == row[1] for existing in selected):
                    continue
                selected.append(row)
                if len(selected) >= n_results:
                    return selected
        return selected

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
        related_agent_id: Any = None,
        object_id: Any = None,
        post_id: Any = None,
        task: Any = None,
        need_key: Any = None,
        episode_id: str = "",
        evidence_event_ids: list[Any] | None = None,
        valid_from: int | None = None,
        valid_to: int | None = None,
        memory_id: str = "",
        importance: float = 0.5,
        confidence: float = 0.5,
    ) -> None:
        """写入不需要 embedding 的衍生记忆，例如本地生成的观念评测证据。"""

        if not memory_id:
            digest = hashlib.md5(summary.encode()).hexdigest()[:12]
            memory_id = f"{agent_id}:{source_type}:{world_time}:{digest}"
        self.store.record_derived_memory(
            memory_id=memory_id,
            agent_id=agent_id,
            memory_type=memory_type,
            source_type=source_type,
            world_time=world_time,
            summary=summary,
            payload=payload,
            entity_id=entity_id,
            related_agent_id=related_agent_id,
            object_id=object_id,
            post_id=post_id,
            task=task,
            need_key=need_key,
            episode_id=episode_id,
            evidence_event_ids=evidence_event_ids,
            valid_from=world_time if valid_from is None else valid_from,
            valid_to=valid_to,
            importance=importance,
            confidence=confidence,
        )

    def _consolidate_action_experience(
        self,
        agent_id: str,
        *,
        event_id: int,
        episode_id: str,
        world_time: int,
        tool: str,
        args: dict[str, Any],
        feedback: Any,
        reward: float | None,
        state_before: dict[str, Any],
        state_after: dict[str, Any],
        outcome: dict[str, Any],
        need_events: list[dict[str, Any]],
    ) -> None:
        """将高信息量行动压缩为长期情景经验，同时保留原始事件证据。"""

        status = _text_id(outcome.get("status"))
        meaningful = bool(
            tool in {"task_summary", "task_checkpoint"}
            or need_events
            or status == "failed"
            or (isinstance(reward, (int, float)) and abs(float(reward)) > 0.0)
        )
        if not tool or not meaningful:
            return
        task = _text_id(args.get("task")) or _text_id(state_before.get("task"))
        need_key = _text_id(args.get("need_key")) or _text_id(state_before.get("task_urgency_key"))
        delta_summary = self._state_delta_summary(state_before, state_after)
        summary = (
            f"t={world_time} 执行 {tool}，结果={status}，reward={reward}；"
            f"{delta_summary}反馈={_short(feedback, 240)}"
        )
        stable_episode_id = episode_id or f"{agent_id}:action:{world_time}:{event_id}"
        self._record_derived_without_chroma(
            agent_id=agent_id,
            memory_type="episodic",
            source_type="episode_consolidation",
            world_time=world_time,
            summary=summary,
            payload={
                "tool": tool,
                "args": args,
                "feedback": feedback,
                "reward": reward,
                "state_before": state_before,
                "state_after": state_after,
                "outcome": outcome,
                "need_events": need_events,
            },
            task=task,
            need_key=need_key,
            episode_id=stable_episode_id,
            evidence_event_ids=[event_id],
            memory_id=f"{agent_id}:episode:{stable_episode_id}:{event_id}",
            importance=self._experience_importance(tool, reward, need_events, outcome),
            confidence=0.8,
        )

    def _consolidate_social_experience(
        self,
        agent_id: str,
        *,
        event_id: int,
        episode_id: str,
        world_time: int,
        action: str,
        data: dict[str, Any],
        related_agent_id: str,
        post_id: Any,
        topic: str,
    ) -> None:
        """把一次线上互动压缩为带人物、帖子和传播来源的长期社交经验。"""

        if action == "social_no_action" or data.get("state_changed") is False:
            return
        ok = bool(data.get("ok"))
        summary = (
            f"t={world_time} 社交动作={action} target={related_agent_id or 'none'} "
            f"post={post_id} topic={topic or 'none'} ok={ok}；"
            f"feedback={_short(data.get('feedback', ''), 240)}"
        )
        stable_episode_id = episode_id or f"{agent_id}:social:{world_time}:{event_id}"
        platform_event_id = _text_id(data.get("platform_event_id") or data.get("event_id"))
        memory_suffix = platform_event_id or str(event_id)
        self._record_derived_without_chroma(
            agent_id=agent_id,
            memory_type="episodic",
            source_type="social_consolidation",
            world_time=world_time,
            summary=summary,
            payload=data,
            entity_id=related_agent_id or post_id,
            related_agent_id=related_agent_id,
            object_id=data.get("source_post_id"),
            post_id=post_id,
            episode_id=stable_episode_id,
            evidence_event_ids=[event_id],
            memory_id=f"{agent_id}:social:{memory_suffix}",
            importance=0.72 if ok else 0.62,
            confidence=0.85,
        )

    def _consolidate_conversation_experience(
        self,
        agent_id: str,
        *,
        episode_id: str,
        world_time: int,
        payload: dict[str, Any],
        evidence_event_ids: list[int],
        related_agent_ids: list[str],
    ) -> None:
        """按一次对话决策生成一条长期社交摘要。"""

        messages = [item for item in payload.get("messages") or [] if isinstance(item, dict)]
        first_message = messages[0] if messages else {}
        related_agent_id = related_agent_ids[0] if len(related_agent_ids) == 1 else ""
        summary = (
            f"t={world_time} 与 {','.join(related_agent_ids) or 'unknown'} 对话；"
            f"收到={_short(first_message.get('content', ''), 160)}；"
            f"回复={_short(payload.get('reply', ''), 200)}"
        )
        stable_episode_id = episode_id or f"{agent_id}:conversation:{world_time}:{evidence_event_ids[0]}"
        self._record_derived_without_chroma(
            agent_id=agent_id,
            memory_type="episodic",
            source_type="conversation_consolidation",
            world_time=world_time,
            summary=summary,
            payload=payload,
            related_agent_id=related_agent_id,
            episode_id=stable_episode_id,
            evidence_event_ids=evidence_event_ids,
            memory_id=f"{agent_id}:conversation:{stable_episode_id}",
            importance=0.68,
            confidence=0.8,
        )

    def _action_status(self, tool: str, feedback: Any) -> str:
        """把执行反馈压缩为稳定结果状态。"""

        if not tool:
            return "no_action"
        text = _as_text(feedback).lower()
        if "error:" in text or "failed" in text or "失败" in text:
            return "failed"
        return "completed"

    def _experience_importance(
        self,
        tool: str,
        reward: float | None,
        need_events: list[dict[str, Any]],
        outcome: dict[str, Any],
    ) -> float:
        """按实际结果、需求变化和失败信号计算经历重要性。"""

        score = 0.45 if tool else 0.25
        if isinstance(reward, (int, float)):
            score += min(0.2, abs(float(reward)) / 10.0)
        if need_events:
            max_delta = max(abs(self._float_value(item.get("delta"), 0.0)) for item in need_events)
            score += min(0.2, max_delta / 10.0)
        if outcome.get("status") == "failed":
            score += 0.15
        if tool in {"task_summary", "task_checkpoint"}:
            score += 0.15
        return min(0.95, score)

    def _state_delta_summary(self, before: dict[str, Any], after: dict[str, Any]) -> str:
        """提取需求满足度变化，供长期摘要快速比较行动结果。"""

        before_satisfaction = before.get("satisfaction") if isinstance(before.get("satisfaction"), dict) else {}
        after_satisfaction = after.get("satisfaction") if isinstance(after.get("satisfaction"), dict) else {}
        changes = []
        for key in sorted(set(before_satisfaction) | set(after_satisfaction)):
            old = self._float_value(before_satisfaction.get(key), 0.0)
            new = self._float_value(after_satisfaction.get(key), old)
            if new != old:
                changes.append(f"{key}:{old:.2f}->{new:.2f}")
        return ("需求变化=" + ",".join(changes) + "；") if changes else ""

    def _float_value(self, value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

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
        system, user = self._person_profile_prompt(profile, facts)
        try:
            raw = llm.generate(system, user, response_format=JSON_OBJECT_RESPONSE_FORMAT)
            data = self._parse_json_object(raw)
            return self._person_profile_summary_from_payload(data, fallback)
        except Exception as exc:
            annotate_current_llm_trace(f"{type(exc).__name__}: {exc}")
            logger.debug("person profile LLM summary failed: %s", exc)
            return fallback

    async def _asummarize_person_profile(self, profile: dict[str, Any], facts: list[dict[str, Any]], *, llm=None) -> dict[str, Any]:
        """异步压缩人物档案印象；LLM 不支持异步时使用规则摘要。"""

        fallback = self._rule_person_profile_summary(profile, facts)
        if llm is None or not facts or not hasattr(llm, "agenerate"):
            return fallback
        system, user = self._person_profile_prompt(profile, facts)
        try:
            raw = await llm.agenerate(system, user, response_format=JSON_OBJECT_RESPONSE_FORMAT)
            data = self._parse_json_object(raw)
            return self._person_profile_summary_from_payload(data, fallback)
        except Exception as exc:
            annotate_current_llm_trace(f"{type(exc).__name__}: {exc}")
            logger.debug("async person profile LLM summary failed: %s", exc)
            return fallback

    def _person_profile_prompt(self, profile: dict[str, Any], facts: list[dict[str, Any]]) -> tuple[str, str]:
        """构造人物档案摘要 prompt，供同步和异步路径复用。"""

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
        return system, user

    def _person_profile_summary_from_payload(self, data: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
        """把 LLM 摘要 JSON 标准化为人物档案字段。"""

        return {
            "actions_impression": _short(data.get("actions_impression") or fallback["actions_impression"], 240),
            "opinion_impression": _short(data.get("opinion_impression") or fallback["opinion_impression"], 240),
            "relationship_impression": _short(data.get("relationship_impression") or fallback["relationship_impression"], 240),
            "confidence": self._clamp01(data.get("confidence", fallback["confidence"])),
        }

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
                topic = _text_id(payload.get("topic"))
                content = payload.get("content") or payload.get("feedback") or summary
                topic_text = f"topic={topic} " if topic else ""
                opinion_texts.append(f"{topic_text}{_short(content, 80)} opinion_index={opinion}")
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
        return parse_json_object(raw, context="person profile summary")

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
        topic = _text_id(post.get("topic"))
        topic_text = f" topic={topic}" if topic else ""
        return (
            f"posted:{topic_text} {_short(post.get('content', ''), 120)} "
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
            for key in ("account_ids", "following_ids"):
                ids.extend(_text_id(value) for value in payload.get(key) or [])
            for key in ("actor_id", "target_agent_id", "source_author_id", "related_agent_id"):
                if payload.get(key):
                    ids.append(_text_id(payload.get(key)))
            for post in payload.get("posts") or []:
                if isinstance(post, dict):
                    ids.append(_text_id(post.get("author_id")))
                    ids.append(_text_id(post.get("source_author_id")))
                    for comment in post.get("comments") or []:
                        if isinstance(comment, dict):
                            ids.append(_text_id(comment.get("author_id")))
            # 对话使用结构化 sender/target，避免依赖全文中的 ID 字符串格式。
            for key in ("messages", "conversation_history"):
                for message in payload.get(key) or []:
                    if isinstance(message, dict):
                        ids.extend(
                            [
                                _text_id(message.get("sender")),
                                _text_id(message.get("target")),
                            ]
                        )
            social = payload.get("social") if isinstance(payload.get("social"), dict) else {}
            notes = social.get("notification_events") or social.get("notifications") or []
            for note in notes:
                if isinstance(note, dict):
                    ids.extend(
                        [
                            _text_id(note.get("actor_id")),
                            _text_id(note.get("target_agent_id")),
                            _text_id(note.get("source_author_id")),
                        ]
                    )
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
            if payload.get("source_post_id") is not None:
                ids.append(_text_id(payload.get("source_post_id")))
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
            f"[state t={row.get('last_seen_at')} source={row.get('source_type')} "
            f"confidence={row.get('confidence')}] "
            f"{row.get('entity_type')} {row.get('entity_id')} at {row.get('position')} "
            f"region={row.get('region')}"
        )

    def _format_person_profile(self, row: dict[str, Any]) -> str:
        return (
            f"[person_profile t={max(int(row.get('last_seen_at') or 0), int(row.get('last_social_seen_at') or 0))} "
            f"confidence={row.get('confidence')}] target={row.get('target_agent_id')} "
            f"observed_position={row.get('last_position')} observed_region={row.get('last_region')} "
            f"inferred_actions={_short(row.get('actions_impression', ''))} "
            f"inferred_opinions={_short(row.get('opinion_impression', ''))} "
            f"inferred_relationship={_short(row.get('relationship_impression', ''))} "
            f"recent_social={_short(row.get('recent_post_summary', ''))}"
        )

    def _format_relation(self, row: dict[str, Any]) -> str:
        return (
            f"[relation t={row.get('last_seen_at')} source={row.get('source_type')} "
            f"confidence={row.get('confidence')}] "
            f"{row.get('subject_id')} {row.get('relation_type')} {row.get('object_id')}"
        )

    def _format_social_post(self, row: dict[str, Any]) -> str:
        topic = f" topic={row.get('topic')}" if row.get("topic") else ""
        propagation = ""
        if row.get("repost_of_post_id") or row.get("root_post_id"):
            propagation = (
                f" repost_of={row.get('repost_of_post_id')} root_post={row.get('root_post_id')}"
                f" source_author={row.get('source_author_id')}"
            )
        return (
            f"[social t={row.get('last_seen_at')} post={row.get('post_id')}{topic} confidence={row.get('confidence')}] "
            f"author={row.get('author_id')} likes={row.get('likes')} dislikes={row.get('dislikes')} "
            f"reposts={row.get('reposts')} comments={row.get('comments_count')} seen={row.get('feed_seen_count')} "
            f"opinion_index={row.get('opinion_index')}{propagation} "
            f"content={_short(row.get('content', ''))}"
        )

    def _format_event(self, row: dict[str, Any]) -> str:
        memory_type = row.get("memory_type") or "episodic"
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        refs = []
        for key in (
            "episode_id",
            "related_agent_id",
            "object_id",
            "post_id",
            "comment_id",
            "parent_comment_id",
            "root_comment_id",
            "topic",
            "platform_event_id",
            "feed_request_id",
            "provenance",
        ):
            if row.get(key) not in {None, ""}:
                refs.append(f"{key}={row.get(key)}")
        for key in (
            "target_agent_id",
            "source_post_id",
            "root_post_id",
            "source_author_id",
        ):
            if payload.get(key) not in {None, ""}:
                refs.append(f"{key}={payload.get(key)}")
        refs_text = (" " + " ".join(refs)) if refs else ""
        state_result = ""
        before_state = row.get("before_state") if isinstance(row.get("before_state"), dict) else {}
        after_state = row.get("after_state") if isinstance(row.get("after_state"), dict) else {}
        outcome = row.get("outcome") if isinstance(row.get("outcome"), dict) else {}
        need_events = payload.get("need_events") if isinstance(payload.get("need_events"), list) else []
        if before_state or after_state or outcome or need_events:
            state_result = (
                f" result_before={_short(before_state.get('satisfaction', {}), 160)}"
                f" result_after={_short(after_state.get('satisfaction', {}), 160)}"
                f" outcome={_short(outcome, 180)}"
                f" need_events={_short(need_events, 240)}"
            )
        return (
            f"[{memory_type} t={row.get('world_time')} source={row.get('source_type')} "
            f"confidence={row.get('confidence')}{refs_text}] {row.get('summary')}{state_result}"
        )

    def _format_derived(self, row: dict[str, Any]) -> str:
        memory_type = row.get("memory_type") or "episodic"
        task = f" task={row.get('task')}" if row.get("task") else ""
        need = f" need={row.get('need_key')}" if row.get("need_key") else ""
        episode = f" episode={row.get('episode_id')}" if row.get("episode_id") else ""
        evidence_ids = row.get("evidence_event_ids") or []
        evidence = f" evidence={','.join(str(value) for value in evidence_ids)}" if evidence_ids else ""
        refs = "".join(
            f" {key}={row.get(key)}"
            for key in ("entity_id", "related_agent_id", "object_id", "post_id")
            if row.get(key) not in {None, ""}
        )
        return (
            f"[{memory_type} t={row.get('world_time')} source={row.get('source_type')}"
            f"{task}{need}{episode}{evidence}{refs} "
            f"confidence={row.get('confidence')}] {row.get('summary')}"
        )
