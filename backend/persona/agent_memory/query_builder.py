from __future__ import annotations

import json
from typing import Any


DEFAULT_SEMANTIC_QUERY_MAX_BYTES = 6000
_FALLBACK_TEXT_MAX_CHARS = 1200
_SOCIAL_POST_SUMMARY_LIMIT = 3
_CONVERSATION_MESSAGE_LIMIT = 3
_CONVERSATION_HISTORY_LIMIT = 2


def bound_utf8_text(text: Any, max_bytes: int) -> str:
    """按 UTF-8 字节截断文本，并保证返回值仍是合法字符串。"""

    value = str(text or "")
    limit = max(0, int(max_bytes))
    raw = value.encode("utf-8")
    if len(raw) <= limit:
        return value
    return raw[:limit].decode("utf-8", errors="ignore")


def build_semantic_observation_text(
    observation: Any,
    context: str,
    *,
    max_bytes: int = DEFAULT_SEMANTIC_QUERY_MAX_BYTES,
) -> str:
    """按决策场景投影观察，只保留向量召回需要的语义信息。"""

    context_name = str(context or "world")
    payload = _as_dict(observation)
    if context_name == "social" or payload.get("type") == "social_browse":
        text = _social_query(payload, observation)
    elif context_name == "conversation":
        text = _conversation_query(payload, observation)
    elif context_name == "opinion_assessment" or payload.get("type") == "opinion_assessment":
        text = _opinion_query(payload, observation)
    elif context_name == "world":
        text = _world_query(payload, observation)
    else:
        text = _fallback_query(context_name, payload, observation)
    return bound_utf8_text(text, max_bytes)


def build_memory_planner_observation_text(
    observation: Any,
    context: str,
    *,
    max_bytes: int = DEFAULT_SEMANTIC_QUERY_MAX_BYTES,
) -> str:
    """为 planner 补充有界 ID 引用，使结构化查询仍能精确定位。"""

    context_name = str(context or "world")
    payload = _as_dict(observation)
    parts = [build_semantic_observation_text(observation, context_name)]

    if context_name == "social" and payload:
        post_refs = []
        posts = [item for item in payload.get("posts") or [] if isinstance(item, dict)]
        for post in posts[:10]:
            fields = [
                f"post_id={_text(post.get('id'))}" if post.get("id") is not None else "",
                f"author_id={_text(post.get('author_id'))}" if post.get("author_id") else "",
                f"source_author_id={_text(post.get('source_author_id'))}" if post.get("source_author_id") else "",
            ]
            ref = ",".join(value for value in fields if value)
            if ref:
                post_refs.append(ref)
        if post_refs:
            parts.append("visible_post_refs=" + "; ".join(post_refs))

        # planner 只看少量帖子摘要，用于形成聚焦查询，不复制评论线程。
        post_summaries = []
        for post in posts:
            content = _short(post.get("content"), 180)
            if content:
                post_summaries.append(
                    f"post_id={_text(post.get('id'))},content={content}"
                    if post.get("id") is not None
                    else f"content={content}"
                )
            if len(post_summaries) >= _SOCIAL_POST_SUMMARY_LIMIT:
                break
        if post_summaries:
            parts.append("visible_post_summaries=" + "; ".join(post_summaries))

    if context_name == "world" and payload:
        people_ids = _unique_texts(
            item.get("id")
            for item in payload.get("people") or []
            if isinstance(item, dict)
        )
        if people_ids:
            parts.append("visible_people=" + ",".join(people_ids[:10]))

        object_refs = []
        for item in payload.get("objects") or []:
            if not isinstance(item, dict):
                continue
            object_id = _text(item.get("id")).strip()
            kind = _text(item.get("kind")).strip()
            if object_id:
                object_refs.append(f"{object_id}:{kind}" if kind else object_id)
            if len(object_refs) >= 10:
                break
        if object_refs:
            parts.append("visible_objects=" + ",".join(object_refs))

        social = payload.get("social") if isinstance(payload.get("social"), dict) else {}
        notification_events = social.get("notification_events")
        notes = notification_events if isinstance(notification_events, list) else social.get("notifications") or []
        notification_refs = []
        for note in notes[:3]:
            if not isinstance(note, dict):
                continue
            fields = []
            for key in ("actor_id", "source_author_id", "post_id", "source_post_id", "comment_id"):
                if note.get(key) not in (None, ""):
                    fields.append(f"{key}={note.get(key)}")
            if fields:
                notification_refs.append(",".join(fields))
        if notification_refs:
            parts.append("notification_refs=" + "; ".join(notification_refs))

    return bound_utf8_text(" | ".join(part for part in parts if part), max_bytes)


def _social_query(payload: dict[str, Any], observation: Any) -> str:
    if not payload:
        return _fallback_query("social", payload, observation)

    posts = [item for item in payload.get("posts") or [] if isinstance(item, dict)]
    topics = _unique_texts(item.get("topic") for item in posts)
    parts = ["context=social", "type=social_browse"]
    if topics:
        parts.append("topics=" + "; ".join(_short(topic, 180) for topic in topics))
    else:
        summaries = []
        for post in posts:
            content = _short(post.get("content"), 180)
            if content:
                summaries.append(content)
            if len(summaries) >= _SOCIAL_POST_SUMMARY_LIMIT:
                break
        if summaries:
            parts.append("post_summaries=" + "; ".join(summaries))

    source_types = _unique_texts(item.get("source_type") for item in posts)
    if source_types:
        parts.append("source_types=" + ",".join(source_types))
    if any(bool(item.get("is_news")) for item in posts):
        parts.append("contains_news=true")
    if any(bool(item.get("is_rumor")) for item in posts):
        parts.append("contains_rumor=true")
    return " | ".join(parts)


def _world_query(payload: dict[str, Any], observation: Any) -> str:
    if not payload:
        return _fallback_query("world", payload, observation)

    parts = ["context=world"]
    region = payload.get("region") if isinstance(payload.get("region"), dict) else {}
    region_text = _first_text(region.get("kind"), region.get("name"), region.get("id"))
    if region_text:
        parts.append(f"region={_short(region_text, 120)}")

    object_kinds = _unique_texts(
        item.get("kind")
        for item in payload.get("objects") or []
        if isinstance(item, dict)
    )
    if object_kinds:
        parts.append("visible_object_kinds=" + ",".join(object_kinds[:10]))

    action_summaries = []
    actions = [item for item in payload.get("actions") or [] if isinstance(item, dict)]
    for action in actions[-3:]:
        fields = [
            _text(action.get("type")),
            f"actor={_text(action.get('actor_id'))}" if action.get("actor_id") else "",
            f"target={_text(action.get('acted_id'))}" if action.get("acted_id") else "",
            _short(_stable_text(action.get("info")), 160),
        ]
        summary = " ".join(item for item in fields if item)
        if summary:
            action_summaries.append(summary)
    if action_summaries:
        parts.append("recent_actions=" + "; ".join(action_summaries))

    social = payload.get("social") if isinstance(payload.get("social"), dict) else {}
    notification_events = social.get("notification_events")
    notes = notification_events if isinstance(notification_events, list) else social.get("notifications") or []
    notification_summaries = []
    for note in notes[:3]:
        if not isinstance(note, dict):
            continue
        fields = [
            _text(note.get("event_type")),
            f"actor={_first_text(note.get('actor_id'), note.get('source_author_id'))}"
            if note.get("actor_id") or note.get("source_author_id")
            else "",
            f"topic={_short(note.get('topic'), 160)}" if note.get("topic") else "",
            _short(note.get("content"), 180),
        ]
        summary = " ".join(item for item in fields if item)
        if summary:
            notification_summaries.append(summary)
    if notification_summaries:
        parts.append("notifications=" + "; ".join(notification_summaries))
    return " | ".join(parts)


def _conversation_query(payload: dict[str, Any], observation: Any) -> str:
    if not payload:
        return _fallback_query("conversation", payload, observation)

    parts = ["context=conversation"]
    messages = [item for item in payload.get("messages") or [] if isinstance(item, dict)]
    message_summaries = [
        _conversation_item(item, content_limit=180)
        for item in messages[-_CONVERSATION_MESSAGE_LIMIT:]
    ]
    message_summaries = [item for item in message_summaries if item]
    if message_summaries:
        parts.append("messages=" + "; ".join(message_summaries))

    history = [item for item in payload.get("conversation_history") or [] if isinstance(item, dict)]
    history_summaries = [
        _conversation_item(item, content_limit=120)
        for item in history[-_CONVERSATION_HISTORY_LIMIT:]
    ]
    history_summaries = [item for item in history_summaries if item]
    if history_summaries:
        parts.append("recent_history=" + "; ".join(history_summaries))
    return " | ".join(parts)


def _conversation_item(item: dict[str, Any], *, content_limit: int) -> str:
    fields = [
        f"sender={_text(item.get('sender'))}" if item.get("sender") else "",
        f"target={_text(item.get('target'))}" if item.get("target") else "",
        f"intent={_text(item.get('intent'))}" if item.get("intent") else "",
        f"topic={_short(item.get('topic'), 160)}" if item.get("topic") else "",
        f"content={_short(item.get('content'), content_limit)}" if item.get("content") else "",
        f"response_to={_short(item.get('response_to'), 100)}" if item.get("response_to") else "",
    ]
    return " ".join(value for value in fields if value)


def _opinion_query(payload: dict[str, Any], observation: Any) -> str:
    if not payload:
        return _fallback_query("opinion_assessment", payload, observation)

    parts = ["context=opinion_assessment", "type=opinion_assessment"]
    if payload.get("topic"):
        parts.append(f"topic={_short(payload.get('topic'), 240)}")
    if payload.get("window_start_tick") is not None:
        parts.append(f"window_start_tick={payload.get('window_start_tick')}")
    if payload.get("window_end_tick") is not None:
        parts.append(f"window_end_tick={payload.get('window_end_tick')}")
    for key in ("self_authored_post_ids", "self_authored_comment_ids", "observed_speech_ids"):
        values = payload.get(key) if isinstance(payload.get(key), list) else []
        parts.append(f"{key}_count={len(values)}")
    return " | ".join(parts)


def _fallback_query(context: str, payload: dict[str, Any], observation: Any) -> str:
    if payload:
        compact = {
            "type": payload.get("type"),
            "time": payload.get("time"),
        }
        text = _stable_text(compact)
    else:
        text = _short(observation, _FALLBACK_TEXT_MAX_CHARS)
    return f"context={context} | observation={text}" if text else f"context={context}"


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _unique_texts(values) -> list[str]:
    seen = set()
    result = []
    for value in values:
        text = _text(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _first_text(*values: Any) -> str:
    for value in values:
        text = _text(value).strip()
        if text:
            return text
    return ""


def _stable_text(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return _text(value)


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _short(value: Any, limit: int) -> str:
    text = _stable_text(value).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."
