from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class ConversationIntent(StrEnum):
    ASK_INFO = "ask_info"
    ANSWER_INFO = "answer_info"
    SOCIAL_BONDING = "social_bonding"
    PERSUASION = "persuasion"
    COORDINATION = "coordination"
    REACTION = "reaction"
    NOTIFICATION = "notification"
    END = "end"


def infer_conversation_intent(content: str, response_to: str | None = None) -> ConversationIntent:
    text = (content or "").strip()
    if not text:
        return ConversationIntent.REACTION

    lowered = text.lower()
    if any(token in text for token in ("谢谢", "再见", "结束", "不用了", "知道了")):
        return ConversationIntent.END
    if response_to:
        return ConversationIntent.ANSWER_INFO
    if any(token in text for token in ("过得怎么样", "最近怎么样", "心情怎么样", "今天怎么样")):
        return ConversationIntent.SOCIAL_BONDING
    if any(token in text for token in ("?", "？", "吗", "哪里", "在哪", "怎么", "如何", "能否", "可否")):
        return ConversationIntent.ASK_INFO
    if any(token in text for token in ("一起", "去", "前往", "约", "集合", "帮我")):
        return ConversationIntent.COORDINATION
    if any(token in text for token in ("认为", "观点", "支持", "反对", "应该", "不应该")):
        return ConversationIntent.PERSUASION
    if lowered.startswith("[新闻]") or lowered.startswith("[社交通知]"):
        return ConversationIntent.NOTIFICATION
    return ConversationIntent.SOCIAL_BONDING


@dataclass
class ConversationMessage:
    message_id: str
    session_id: str
    round: int
    sender: str
    target: str
    content: str
    intent: ConversationIntent
    time: int
    response_to: str | None = None

    def to_history_entry(self) -> dict:
        return {
            "round": self.round,
            "sender": self.sender,
            "target": self.target,
            "content": self.content,
            "response_to": self.response_to,
            "session_id": self.session_id,
            "intent": self.intent.value,
        }


@dataclass
class ConversationSession:
    session_id: str
    participants: list[str]
    initiator: str
    topic: str
    intent: ConversationIntent
    status: str
    round: int
    max_rounds: int
    created_at: int
    last_updated: int
    messages: list[ConversationMessage] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    known_facts: list[str] = field(default_factory=list)
    termination_reason: str | None = None

    def add_message(self, message: ConversationMessage) -> None:
        self.messages.append(message)
        self.last_updated = message.time
        if message.sender not in self.participants:
            self.participants.append(message.sender)
        if message.target != "<all>" and message.target not in self.participants:
            self.participants.append(message.target)
        if message.intent == ConversationIntent.ASK_INFO and message.content not in self.open_questions:
            self.open_questions.append(message.content)
        elif message.intent == ConversationIntent.ANSWER_INFO and message.content not in self.known_facts:
            self.known_facts.append(message.content)

    def history_entries(self) -> list[dict]:
        return [message.to_history_entry() for message in self.messages]

    def mark_resolved(self, reason: str) -> None:
        self.status = "resolved"
        self.termination_reason = reason
