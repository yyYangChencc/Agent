from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class ConversationIntent(StrEnum):
    ASK_INFO = "ask_info"
    ANSWER_INFO = "answer_info"
    ASK_HELP = "ask_help"
    OFFER_HELP = "offer_help"
    SOCIAL_BONDING = "social_bonding"
    SELF_DISCLOSURE = "self_disclosure"
    EMOTIONAL_SUPPORT = "emotional_support"
    PERSUASION = "persuasion"
    COORDINATION = "coordination"
    DISAGREEMENT = "disagreement"
    CONFLICT = "conflict"
    THANKS = "thanks"
    APOLOGY = "apology"
    REACTION = "reaction"
    NOTIFICATION = "notification"
    END = "end"


def infer_conversation_intent(content: str, response_to: str | None = None) -> ConversationIntent:
    text = (content or "").strip()
    if not text:
        return ConversationIntent.REACTION

    lowered = text.lower()
    if any(token in text for token in ("谢谢", "感谢", "多谢", "谢了")):
        return ConversationIntent.THANKS
    if any(token in text for token in ("抱歉", "对不起", "不好意思")):
        return ConversationIntent.APOLOGY
    if any(token in text for token in ("闭嘴", "蠢", "废物", "滚", "威胁")):
        return ConversationIntent.CONFLICT
    if any(token in text for token in ("我不同意", "不赞成", "反驳", "质疑你的")):
        return ConversationIntent.DISAGREEMENT
    if any(token in text for token in ("帮帮我", "帮我", "能不能帮", "需要帮助", "借我")):
        return ConversationIntent.ASK_HELP
    if any(token in text for token in ("我可以帮", "我来帮", "需要我帮", "我陪你")):
        return ConversationIntent.OFFER_HELP
    if any(token in text for token in ("我有点", "我最近", "我感觉", "我担心", "我压力")):
        return ConversationIntent.SELF_DISCLOSURE
    if any(token in text for token in ("别太担心", "我理解", "辛苦了", "支持你", "可以先休息")):
        return ConversationIntent.EMOTIONAL_SUPPORT
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


def normalize_conversation_intent(
    intent: str | ConversationIntent | None,
    content: str,
    response_to: str | None = None,
) -> ConversationIntent:
    """把 LLM 给出的意图规范化；非法值回退到文本规则判断。"""

    if isinstance(intent, ConversationIntent):
        return intent
    try:
        return ConversationIntent(str(intent or "").strip())
    except ValueError:
        return infer_conversation_intent(content, response_to)


def normalize_social_valence(value) -> float:
    """把社会效价规范化到 [-1, 1]；非法值按中性处理。"""

    if value is None or isinstance(value, bool):
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number < -1.0 or number > 1.0:
        return 0.0
    return number


def normalize_topic_stance(topic: str, value, default_topic: str) -> float | None:
    """只有系统新闻主题允许保留 topic_stance，其他话题统一写入 None。"""

    if not topic or topic != default_topic or value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number < -1.0 or number > 1.0:
        return None
    return number


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
    social_valence: float = 0.0
    topic: str = ""
    topic_stance: float | None = None

    def to_history_entry(self) -> dict:
        return {
            "message_id": self.message_id,
            "round": self.round,
            "sender": self.sender,
            "target": self.target,
            "content": self.content,
            "response_to": self.response_to,
            "session_id": self.session_id,
            "intent": self.intent.value,
            "social_valence": self.social_valence,
            "topic": self.topic,
            "topic_stance": self.topic_stance,
            "time": self.time,
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
