from __future__ import annotations

import json
from typing import TYPE_CHECKING

from persona.conversation.session import (
    ConversationIntent,
    ConversationMessage,
    ConversationSession,
    normalize_conversation_intent,
    normalize_social_valence,
    normalize_topic_stance,
)
from persona.logger import get_logger
from persona.need_events import apply_conversation_need_event

if TYPE_CHECKING:
    from persona.agents.agent import Agent
    from persona.agents.policy import Policy
    from world.world import World

logger = get_logger(__name__)


class ConversationManager:
    def __init__(self, world: "World"):
        self.world = world
        self.sessions: dict[str, ConversationSession] = {}
        self._message_counter = 0
        self._session_counter = 0

    def run_phase(self, agents: list["Agent"], policy: "Policy", max_rounds: int) -> None:
        for agent in agents:
            agent.conversation_opted_out = False

        active_sessions = self._seed_sessions_from_inboxes(agents, max_rounds)
        self._resolve_mutual_speaks(agents, active_sessions)

        conv_history = self._history_from_sessions(active_sessions)
        for round_n in range(1, max_rounds + 1):
            selected = self._select_speakers(active_sessions, agents)
            if not selected:
                self._mark_open_sessions_resolved(active_sessions, "no_respondents")
                break
            logger.info(
                "[Conversation] tick=%d round=%d/%d selected_speakers=%d sessions=%d",
                self.world.time,
                round_n,
                max_rounds,
                len(selected),
                len(active_sessions),
            )

            round_history: list[dict] = []

            for session, agent in selected:
                try:
                    action = agent.conversation_step(policy, round_n, max_rounds, list(conv_history))
                    if not action:
                        if self._session_has_pending_inbox(session, agents):
                            continue
                        if session.status == "active":
                            session.mark_resolved("speaker_opted_out")
                        continue
                    decision = json.loads(action)
                    data = self._action_from_decision(decision)
                    if not data:
                        if self._session_has_pending_inbox(session, agents):
                            continue
                        if session.status == "active":
                            session.mark_resolved("speaker_opted_out")
                        continue
                    if data.get("tool") != "speak":
                        continue
                    self.world.execute(agent, action)
                    args = data.get("args", {})
                    target = str(args.get("ID", ""))
                    content = str(args.get("content", ""))
                    response_to = args.get("response_to")
                    message = self._new_message(
                        session,
                        round_n,
                        agent.id,
                        target,
                        content,
                        response_to,
                        args.get("intent"),
                        args.get("social_valence"),
                        args.get("topic"),
                        args.get("topic_stance"),
                    )
                    self._finalize_message(session, message, agents)
                    round_history.append(message.to_history_entry())
                    logger.info(
                        "[Conversation] session=%s round=%d %s -> %s intent=%s",
                        session.session_id,
                        round_n,
                        agent.id,
                        target,
                        message.intent.value,
                    )
                    reason = self._termination_reason(session)
                    if reason:
                        session.mark_resolved(reason)
                except json.JSONDecodeError:
                    logger.warning("[Conversation] agent=%s returned invalid conversation JSON: %r", agent.id, action)
                except Exception as e:
                    logger.error("[Conversation] agent %s 对话轮次 %d 失败: %s", agent.id, round_n, e, exc_info=True)

            conv_history.extend(round_history)
            for session in active_sessions:
                session.round = round_n

        for session in active_sessions:
            if session.status == "active":
                session.mark_resolved("max_rounds")
                logger.info(
                    "[Conversation] session=%s resolved reason=%s messages=%d",
                    session.session_id,
                    session.termination_reason,
                    len(session.messages),
                )

    def _seed_sessions_from_inboxes(
        self,
        agents: list["Agent"],
        max_rounds: int,
    ) -> list[ConversationSession]:
        seeded: list[ConversationSession] = []
        session_by_pair: dict[tuple[str, ...], ConversationSession] = {}
        for receiver in agents:
            for msg in receiver.inbox:
                sender = str(msg["sender"])
                raw_target = str(msg.get("target") or receiver.id)
                target = raw_target if raw_target == "<all>" else receiver.id
                key = self._session_key(sender, target)
                session = session_by_pair.get(key)
                if session is None:
                    session = self._new_session(
                        sender,
                        target,
                        msg.get("content", ""),
                        msg.get("response_to"),
                        max_rounds,
                        msg.get("intent"),
                    )
                    session_by_pair[key] = session
                    seeded.append(session)
                if not self._session_has_seed_message(session, sender, target, str(msg.get("content", ""))):
                    message = self._new_message(
                        session,
                        0,
                        sender,
                        target,
                        str(msg.get("content", "")),
                        msg.get("response_to"),
                        msg.get("intent"),
                        msg.get("social_valence"),
                        msg.get("topic"),
                        msg.get("topic_stance"),
                    )
                    self._finalize_message(session, message, agents)
                msg["session_id"] = session.session_id
                msg["intent"] = session.intent.value
        if seeded:
            logger.info(
                "[Conversation] tick=%d seeded_sessions=%d seeded_messages=%d",
                self.world.time,
                len(seeded),
                sum(len(session.messages) for session in seeded),
            )
        return seeded

    def _resolve_mutual_speaks(
        self,
        agents: list["Agent"],
        sessions: list[ConversationSession],
    ) -> None:
        conv_history = self._history_from_sessions(sessions)
        main_step_senders = {entry["sender"] for entry in conv_history if entry["round"] == 0}
        for agent in agents:
            if agent.id in main_step_senders and agent.inbox:
                if all(msg["sender"] in main_step_senders for msg in agent.inbox):
                    if all(agent.id < msg["sender"] for msg in agent.inbox):
                        logger.info("[Conversation] 检测到互相说话，清空 %s 的收件箱以避免双向对话循环", agent.id)
                        agent.inbox.clear()

    def _session_for_reply(
        self,
        active_sessions: list[ConversationSession],
        sender: str,
        target: str,
        max_rounds: int,
    ) -> ConversationSession:
        target_ids = set(target.split()) if target and target != "<all>" else set()
        for session in active_sessions:
            participants = set(session.participants)
            if sender in participants and (
                target == "<all>"
                or not target_ids
                or target_ids.issubset(participants)
                or target_ids.intersection(participants)
            ):
                return session
        session = self._new_session(sender, target or "<all>", "", None, max_rounds)
        active_sessions.append(session)
        return session

    def _select_speakers(
        self,
        active_sessions: list[ConversationSession],
        agents: list["Agent"],
    ) -> list[tuple[ConversationSession, "Agent"]]:
        agent_by_id = {agent.id: agent for agent in agents}
        selected: list[tuple[ConversationSession, "Agent"]] = []
        used_speakers: set[str] = set()
        for session in active_sessions:
            if session.status != "active":
                continue
            speaker = self._select_speaker_for_session(session, agent_by_id, used_speakers)
            if speaker is None:
                continue
            selected.append((session, speaker))
            used_speakers.add(speaker.id)
        return selected

    def _select_speaker_for_session(
        self,
        session: ConversationSession,
        agent_by_id: dict[str, "Agent"],
        used_speakers: set[str],
    ) -> "Agent" | None:
        last_message = session.messages[-1] if session.messages else None
        preferred_ids: list[str] = []
        if last_message:
            if last_message.target != "<all>":
                preferred_ids.append(last_message.target)
            preferred_ids.extend(
                agent_id for agent_id in session.participants
                if agent_id != last_message.sender
            )
        else:
            preferred_ids.extend(session.participants)

        for agent_id in preferred_ids:
            if agent_id in used_speakers:
                continue
            agent = agent_by_id.get(agent_id)
            if agent is None or agent.conversation_opted_out or not agent.inbox:
                continue
            if not self._agent_has_session_inbox(agent, session.session_id):
                continue
            return agent
        return None

    def _agent_has_session_inbox(self, agent: "Agent", session_id: str) -> bool:
        return any(msg.get("session_id") == session_id for msg in agent.inbox)

    def _session_has_pending_inbox(self, session: ConversationSession, agents: list["Agent"]) -> bool:
        return any(self._agent_has_session_inbox(agent, session.session_id) for agent in agents)

    def _session_has_seed_message(
        self,
        session: ConversationSession,
        sender: str,
        target: str,
        content: str,
    ) -> bool:
        return any(
            message.round == 0
            and message.sender == sender
            and message.target == target
            and message.content == content
            for message in session.messages
        )

    def _termination_reason(self, session: ConversationSession) -> str | None:
        if session.messages and session.messages[-1].intent == ConversationIntent.END:
            return "explicit_end"
        if session.intent == ConversationIntent.ASK_INFO and session.known_facts:
            return "resolved_intent"
        return None

    def _new_session(
        self,
        sender: str,
        target: str,
        content: str,
        response_to: str | None,
        max_rounds: int,
        intent_value=None,
    ) -> ConversationSession:
        self._session_counter += 1
        session_id = f"conv_{self.world.time}_{self._session_counter}"
        intent = normalize_conversation_intent(intent_value, content, response_to)
        participants = [sender]
        if target == "<all>":
            participants.extend(agent_id for agent_id in self.world.agents if agent_id != sender)
        elif target not in participants:
            participants.append(target)
        session = ConversationSession(
            session_id=session_id,
            participants=participants,
            initiator=sender,
            topic=self._topic_from_content(content),
            intent=intent,
            status="active",
            round=0,
            max_rounds=max_rounds,
            created_at=self.world.time,
            last_updated=self.world.time,
        )
        self.sessions[session_id] = session
        return session

    def _new_message(
        self,
        session: ConversationSession,
        round_n: int,
        sender: str,
        target: str,
        content: str,
        response_to: str | None,
        intent_value=None,
        social_valence=None,
        topic=None,
        topic_stance=None,
    ) -> ConversationMessage:
        self._message_counter += 1
        default_topic = self._default_opinion_topic(sender)
        normalized_topic = str(topic or "").strip()
        if normalized_topic != default_topic:
            normalized_topic = ""
        return ConversationMessage(
            message_id=f"{session.session_id}_m{self._message_counter}",
            session_id=session.session_id,
            round=round_n,
            sender=sender,
            target=target,
            content=content,
            intent=normalize_conversation_intent(intent_value, content, response_to),
            response_to=response_to,
            time=self.world.time,
            social_valence=normalize_social_valence(social_valence),
            topic=normalized_topic,
            topic_stance=normalize_topic_stance(normalized_topic, topic_stance, default_topic),
        )

    def _default_opinion_topic(self, sender: str) -> str:
        """读取发送者配置中的系统新闻主题，用于校验 topic_stance。"""

        agent = self.world.agents.get(sender)
        config = getattr(agent, "config", None)
        return str(getattr(config, "default_opinion_topic", "") or "")

    def _finalize_message(
        self,
        session: ConversationSession,
        message: ConversationMessage,
        agents: list["Agent"],
    ) -> None:
        """统一落地对话消息、历史日志和需求事件。"""

        session.add_message(message)
        agent_by_id = {agent.id: agent for agent in agents}
        self._stamp_pending_inbox(message, agent_by_id)
        self._record_conversation_message(message, agent_by_id)
        apply_conversation_need_event(message, session, agent_by_id)

    def _record_conversation_message(
        self,
        message: ConversationMessage,
        agent_by_id: dict[str, "Agent"],
    ) -> None:
        """把结构化对话消息写入参与者运行时日志，供 HistoryRecorder 导出。"""

        entry = message.to_history_entry()
        receiver_ids: set[str] = {message.sender}
        if message.target == "<all>":
            receiver_ids.update(agent_by_id.keys())
        else:
            receiver_ids.add(message.target)
        for agent_id in receiver_ids:
            agent = agent_by_id.get(agent_id)
            if agent is None:
                continue
            agent.conversation_event_log.append(dict(entry))
            if len(agent.conversation_event_log) > 500:
                agent.conversation_event_log = agent.conversation_event_log[-500:]

    def _stamp_pending_inbox(
        self,
        message: ConversationMessage,
        agent_by_id: dict[str, "Agent"],
    ) -> None:
        """把刚落地的会话元数据回填到接收者 inbox，保证下一轮能按 session 继续。"""

        if message.target == "<all>":
            target_ids = [agent_id for agent_id in agent_by_id if agent_id != message.sender]
        else:
            target_ids = [message.target]
        for agent_id in target_ids:
            agent = agent_by_id.get(agent_id)
            if agent is None:
                continue
            for item in getattr(agent, "inbox", []) or []:
                if item.get("sender") != message.sender or item.get("content") != message.content:
                    continue
                item["session_id"] = message.session_id
                item["intent"] = message.intent.value
                item["social_valence"] = message.social_valence
                item["topic"] = message.topic
                item["topic_stance"] = message.topic_stance
                item["target"] = message.target

    def _history_from_sessions(self, sessions: list[ConversationSession]) -> list[dict]:
        history = []
        for session in sessions:
            history.extend(session.history_entries())
        history.sort(key=lambda entry: (entry["round"], entry["session_id"], entry["sender"], entry["target"]))
        return history

    def _mark_open_sessions_resolved(self, sessions: list[ConversationSession], reason: str) -> None:
        for session in sessions:
            if session.status == "active":
                session.mark_resolved(reason)
                logger.info(
                    "[Conversation] session=%s resolved reason=%s messages=%d",
                    session.session_id,
                    reason,
                    len(session.messages),
                )

    def _session_key(self, sender: str, target: str) -> tuple[str, ...]:
        if target == "<all>":
            return ("<all>", sender)
        return tuple(sorted((sender, target)))

    def _topic_from_content(self, content: str) -> str:
        text = (content or "").strip()
        if len(text) <= 40:
            return text or "未命名话题"
        return text[:40] + "..."

    def _action_from_decision(self, decision) -> dict:
        if not isinstance(decision, dict):
            return {}
        action = decision.get("action", {})
        if not isinstance(action, dict):
            return {}
        return action
