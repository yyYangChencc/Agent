from __future__ import annotations

from typing import TYPE_CHECKING

from persona.logger import get_logger

if TYPE_CHECKING:
    from persona.agents.agent import Agent
    from persona.conversation.session import ConversationMessage, ConversationSession


logger = get_logger(__name__)


def apply_need_delta(
    agent: "Agent | None",
    need_key: str,
    delta: float,
    *,
    source: str,
    reason: str,
    tick: int | None = None,
    evidence: dict | None = None,
    extra: dict | None = None,
) -> dict | None:
    """按规则更新需求满足度，并把可审计事件写入日志。"""

    if agent is None or need_key not in getattr(agent, "satisfaction", {}):
        return None
    try:
        numeric_delta = float(delta)
    except (TypeError, ValueError):
        return None
    if numeric_delta == 0:
        return None

    before = float(agent.satisfaction.get(need_key, 0.0))
    agent.update_satisfaction(need_key, numeric_delta)
    after = float(agent.satisfaction.get(need_key, 0.0))
    actual_delta = after - before
    if actual_delta == 0:
        if numeric_delta > 0:
            _mark_positive_need_tick(agent, need_key, numeric_delta, _event_tick(agent, tick))
        return None

    event_tick = _event_tick(agent, tick)
    event = {
        "tick": event_tick,
        "agent_id": agent.id,
        "need_key": need_key,
        "delta": round(actual_delta, 4),
        "source": source,
        "reason": reason,
        "evidence": dict(evidence or {}),
        "before": round(before, 4),
        "after": round(after, 4),
    }
    if extra:
        event.update(extra)
    _append_need_event(agent, event)
    _mark_positive_need_tick(agent, need_key, actual_delta, event_tick)
    return event


def record_need_delta_events(
    agent: "Agent | None",
    before_satisfaction: dict,
    *,
    source: str,
    reason: str,
    tick: int | None = None,
    evidence: dict | None = None,
    keys: list[str] | tuple[str, ...] | set[str] | None = None,
    extra: dict | None = None,
) -> list[dict]:
    """把已经发生的需求差值补写为事件，不再次修改满足度。"""

    if agent is None:
        return []
    tracked_keys = keys or getattr(agent, "satisfaction", {}).keys()
    event_tick = _event_tick(agent, tick)
    events: list[dict] = []
    for need_key in tracked_keys:
        if need_key not in getattr(agent, "satisfaction", {}):
            continue
        before = float(before_satisfaction.get(need_key, 0.0))
        after = float(agent.satisfaction.get(need_key, 0.0))
        actual_delta = after - before
        if actual_delta == 0:
            continue
        event = {
            "tick": event_tick,
            "agent_id": agent.id,
            "need_key": need_key,
            "delta": round(actual_delta, 4),
            "source": source,
            "reason": reason,
            "evidence": dict(evidence or {}),
            "before": round(before, 4),
            "after": round(after, 4),
        }
        if extra:
            event.update(extra)
        _append_need_event(agent, event)
        _mark_positive_need_tick(agent, need_key, actual_delta, event_tick)
        events.append(event)
    return events


def apply_passive_need_decay(agent: "Agent", tick: int) -> list[dict]:
    """在心理评测前执行高层需求的被动衰减。"""

    events: list[dict] = []
    events.extend(_apply_social_need_decay(agent, tick))
    event = _apply_self_actualization_decay(agent, tick)
    if event is not None:
        events.append(event)
    return events


def apply_conversation_need_event(
    message: "ConversationMessage",
    session: "ConversationSession",
    agent_by_id: dict[str, "Agent"],
) -> None:
    """根据线下对话消息生成归属和尊重需求事件。"""

    # 延迟导入避免 persona.conversation.__init__ 与本模块形成循环导入。
    from persona.conversation.session import ConversationIntent

    sender = agent_by_id.get(message.sender)
    target = agent_by_id.get(message.target)
    intent = message.intent
    valence = float(message.social_valence or 0.0)

    if intent == ConversationIntent.SOCIAL_BONDING and valence > 0:
        _apply_conversation_delta(target, "belonging", 1.5, message, "收到正向寒暄或关系维持")
        _apply_conversation_delta(sender, "belonging", 0.5, message, "主动发起正向寒暄或关系维持")

    if intent == ConversationIntent.EMOTIONAL_SUPPORT and valence > 0:
        _apply_conversation_delta(target, "belonging", 2.0, message, "收到情绪支持")
        _apply_conversation_delta(sender, "esteem", 0.5, message, "提供情绪支持")

    if intent == ConversationIntent.ANSWER_INFO:
        _apply_conversation_delta(sender, "esteem", 1.0, message, "回答他人问题")
        _apply_conversation_delta(target, "belonging", 0.8, message, "提问得到回应")

    if intent == ConversationIntent.OFFER_HELP and valence >= 0:
        _apply_conversation_delta(sender, "esteem", 1.0, message, "主动提供帮助")
        _apply_conversation_delta(target, "belonging", 0.8, message, "收到帮助意向")

    if intent == ConversationIntent.THANKS:
        _apply_conversation_delta(target, "esteem", 1.2, message, "收到感谢")

    if intent == ConversationIntent.COORDINATION and valence >= 0:
        _apply_conversation_delta(sender, "belonging", 0.6, message, "参与合作协调")
        _apply_conversation_delta(target, "belonging", 0.6, message, "收到合作协调")

    if intent == ConversationIntent.CONFLICT or valence < -0.5:
        _apply_conversation_delta(target, "belonging", -2.0, message, "收到冲突或敌意表达")
        _apply_conversation_delta(target, "esteem", -1.5, message, "收到贬损性互动")

    previous = _previous_conversation_message(session, message)
    if (
        previous is not None
        and previous.intent == ConversationIntent.SELF_DISCLOSURE
        and message.response_to
        and valence > 0
    ):
        disclosed_agent = agent_by_id.get(previous.sender)
        _apply_conversation_delta(disclosed_agent, "belonging", 1.5, message, "自我暴露后得到积极回应")

    if previous is not None and previous.intent == ConversationIntent.APOLOGY and valence >= 0:
        apologizer = agent_by_id.get(previous.sender)
        receiver = agent_by_id.get(message.sender)
        _apply_conversation_delta(apologizer, "belonging", 0.6, message, "道歉得到非负回应")
        _apply_conversation_delta(receiver, "belonging", 0.4, message, "接受关系修复尝试")


def _apply_social_need_decay(agent: "Agent", tick: int) -> list[dict]:
    """归属和尊重长期缺少正向反馈时缓慢下降。"""

    events: list[dict] = []
    cfg = agent.config
    belonging_window = max(1, int(getattr(cfg, "belonging_passive_decay_window_ticks", 10)))
    esteem_window = max(1, int(getattr(cfg, "esteem_passive_decay_window_ticks", 20)))

    if _due_since(getattr(agent, "last_positive_belonging_tick", 0), tick, belonging_window):
        event = apply_need_delta(
            agent,
            "belonging",
            getattr(cfg, "belonging_passive_decay_delta", -0.5),
            source="passive_decay",
            reason="连续多个时间步没有正向归属反馈",
            tick=tick,
            evidence={"window_ticks": belonging_window},
        )
        if event is not None:
            events.append(event)

    if _due_since(getattr(agent, "last_positive_esteem_tick", 0), tick, esteem_window):
        event = apply_need_delta(
            agent,
            "esteem",
            getattr(cfg, "esteem_passive_decay_delta", -0.3),
            source="passive_decay",
            reason="连续多个时间步没有正向尊重反馈",
            tick=tick,
            evidence={"window_ticks": esteem_window},
        )
        if event is not None:
            events.append(event)
    return events


def _apply_self_actualization_decay(agent: "Agent", tick: int) -> dict | None:
    """底层需求稳定但长期重复低层行动时降低自我实现。"""

    cfg = agent.config
    window = max(1, int(getattr(cfg, "self_actualization_repetition_window_ticks", 15)))
    if not _due_since(getattr(agent, "last_self_actualization_tick", 0), tick, window):
        return None
    lower_keys = ("satiety", "relax", "money")
    if not all(agent.satisfaction.get(key, 0.0) > agent.satisfaction_threshold.get(key, 0.0) for key in lower_keys):
        return None
    recent_tools = list(getattr(agent, "recent_action_tools", []) or [])[-window:]
    if len(recent_tools) < window:
        return None
    low_level_tools = set(getattr(cfg, "self_actualization_low_level_tools", []))
    if not recent_tools or any(tool not in low_level_tools for tool in recent_tools):
        return None
    return apply_need_delta(
        agent,
        "self_actualization",
        getattr(cfg, "self_actualization_repetition_decay_delta", -0.5),
        source="passive_decay",
        reason="底层需求稳定但长期重复低层行动",
        tick=tick,
        evidence={"recent_action_tools": recent_tools, "window_ticks": window},
    )


def _previous_conversation_message(
    session: "ConversationSession",
    message: "ConversationMessage",
) -> "ConversationMessage | None":
    """读取同一会话中当前消息前一条消息。"""

    for index, item in enumerate(session.messages):
        if item.message_id == message.message_id and index > 0:
            return session.messages[index - 1]
    return None


def _apply_conversation_delta(
    agent: "Agent | None",
    need_key: str,
    delta: float,
    message: "ConversationMessage",
    reason: str,
) -> None:
    """把对话意图转成统一格式的需求事件。"""

    apply_need_delta(
        agent,
        need_key,
        delta,
        source="conversation",
        reason=reason,
        tick=message.time,
        evidence={
            "session_id": message.session_id,
            "message_id": message.message_id,
            "sender": message.sender,
            "target": message.target,
            "intent": message.intent.value,
            "social_valence": message.social_valence,
            "topic": message.topic,
            "topic_stance": message.topic_stance,
        },
        extra={
            "session_id": message.session_id,
            "message_id": message.message_id,
            "sender": message.sender,
            "target": message.target,
            "intent": message.intent.value,
            "social_valence": message.social_valence,
            "topic": message.topic,
            "topic_stance": message.topic_stance,
        },
    )


def _due_since(last_tick: int, tick: int, window: int) -> bool:
    """判断是否到了按固定窗口衰减的时间点。"""

    elapsed = tick - int(last_tick or 0)
    return elapsed > 0 and elapsed % window == 0


def _event_tick(agent: "Agent", tick: int | None) -> int:
    if tick is not None:
        return int(tick)
    world = getattr(agent, "world", None)
    return int(getattr(world, "time", 0) or 0)


def _append_need_event(agent: "Agent", event: dict) -> None:
    episode_id = str(getattr(agent, "_current_episode_id", "") or "")
    if episode_id and not event.get("episode_id"):
        event["episode_id"] = episode_id
    agent.need_event_log.append(event)
    if len(agent.need_event_log) > 500:
        agent.need_event_log = agent.need_event_log[-500:]

    pending = getattr(agent, "_pending_need_events", None)
    if not isinstance(pending, list):
        pending = []
        agent._pending_need_events = pending
    pending.append(dict(event))
    if len(pending) > 500:
        del pending[:-500]

    mem = getattr(agent, "mem", None)
    if mem is None or not hasattr(mem, "store_need_event"):
        return
    try:
        mem.store_need_event(agent.id, event, episode_id=episode_id)
    except Exception as exc:
        # 记忆持久化失败不能回滚已经发生的需求变化。
        logger.warning("[%s] 需求事件写入记忆失败: %s", agent.id, exc, exc_info=True)


def consume_pending_need_events(agent: "Agent") -> list[dict]:
    """读取并清空尚未被动作经历消费的需求事件。"""

    pending = getattr(agent, "_pending_need_events", None)
    if not isinstance(pending, list) or not pending:
        return []
    events = [dict(event) for event in pending if isinstance(event, dict)]
    pending.clear()
    return events


def _mark_positive_need_tick(agent: "Agent", need_key: str, delta: float, tick: int) -> None:
    """正向事件会刷新对应高层需求的最近满足时间。"""

    if delta <= 0:
        return
    if need_key == "belonging":
        agent.last_positive_belonging_tick = tick
    elif need_key == "esteem":
        agent.last_positive_esteem_tick = tick
    elif need_key == "self_actualization":
        agent.last_self_actualization_tick = tick
