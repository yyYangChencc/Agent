from __future__ import annotations

"""心理评测流水线。

本模块把一段滚动的需求压力窗口转换为心理中介变量和动态角色卡增量。
运行流程如下：

1. AssessmentWindow 记录近期经历、需求满足度变化，以及每个需求的最大/最新有效压力。
2. 只有当某个需求在窗口内的压力超过配置阈值时，才激活对应需求评测器。
3. LLM 模式下默认使用 LLMTheoryCardNeedEvaluator，并以 TheoryCardNeedEvaluator 作为确定性回退。
4. PsychologicalArbiter 将多个被激活需求的评测结果合并为一个角色卡，供后续智能体 prompt 和观念评测使用。
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from persona.llm.interface import JSON_OBJECT_RESPONSE_FORMAT
from persona.llm.json_utils import parse_json_object
from persona.logger import get_logger

if TYPE_CHECKING:
    from persona.agents.agent import Agent
    from persona.config import AgentConfig
    from persona.llm.interface import LLMClient

logger = get_logger(__name__)
THEORY_CARD_PATH = Path(__file__).with_name("theory_cards.json")


# 小型类型转换工具。它们在结果写回智能体之前，保证评测器输出有边界、类型稳定。
def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _as_float(value, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _as_text_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str) and value:
        return [value]
    return []


def _pressure_level(pressure: float) -> str:
    if pressure >= 0.75:
        return "high"
    if pressure >= 0.50:
        return "moderate"
    if pressure >= 0.25:
        return "low"
    return "minimal"


def _merge_unique(target: list[str], source) -> None:
    for item in _as_text_list(source):
        if item not in target:
            target.append(item)


@dataclass
class AssessmentWindow:
    """两次心理评测之间收集的滚动上下文。

    窗口同时保存叙事上下文和数值化需求压力状态。
    每次完成评测或跳过评测后都会重置为新窗口。
    """

    start_tick: int
    end_tick: int | None = None
    experiences: list[str] = field(default_factory=list)
    satisfaction_start: dict[str, float] = field(default_factory=dict)
    satisfaction_end: dict[str, float] = field(default_factory=dict)
    effective_pressure_max: dict[str, float] = field(default_factory=dict)
    effective_pressure_last: dict[str, float] = field(default_factory=dict)

    def add_experience(self, text: str) -> None:
        if text:
            self.experiences.append(text)

    def update_from_agent(self, agent: "Agent") -> None:
        """记录当前需求状态快照，并保留本窗口内出现过的最大压力。"""

        if not self.satisfaction_start:
            self.satisfaction_start = dict(agent.satisfaction)
        self.satisfaction_end = dict(agent.satisfaction)
        for key, pressure in agent.effective_pressure.items():
            self.effective_pressure_last[key] = pressure
            self.effective_pressure_max[key] = max(
                self.effective_pressure_max.get(key, 0.0),
                pressure,
            )

    def need_changes(self) -> dict[str, float]:
        return {
            key: self.satisfaction_end.get(key, 0.0) - start_value
            for key, start_value in self.satisfaction_start.items()
        }


class PsychologicalEvaluator(Protocol):
    """规则评测器、LLM 评测器和占位评测器共用的接口。"""

    need_key: str

    def assess(
        self,
        *,
        agent: "Agent",
        window: AssessmentWindow,
        previous_result: dict | None,
    ) -> dict:
        ...


class TheoryCardRepository:
    """读取理论卡，并按精确的需求键建立索引。"""

    def __init__(self, cards: list[dict]):
        self.cards = cards
        self.by_need_key: dict[str, dict] = {}
        for card in cards:
            need_key = card.get("need_key")
            if isinstance(need_key, str) and need_key:
                self.by_need_key[need_key] = card

    @classmethod
    def load_default(cls) -> "TheoryCardRepository":
        if not THEORY_CARD_PATH.exists():
            logger.warning("[Psychology] theory card file not found: %s", THEORY_CARD_PATH)
            return cls([])
        with THEORY_CARD_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        cards = data.get("cards", [])
        if not isinstance(cards, list):
            logger.warning("[Psychology] theory card file has no cards list: %s", THEORY_CARD_PATH)
            cards = []
        return cls(cards)

    def get(self, need_key: str) -> dict | None:
        return self.by_need_key.get(need_key)


class TheoryCardNeedEvaluator:
    """由单张理论卡驱动的确定性评测器。

    它根据压力、需求下降幅度和上一轮中介变量残余计算新的中介变量。
    同时，它会从理论卡中的压力等级模板生成角色卡增量。
    """

    def __init__(self, card: dict):
        self.card = card
        self.need_key = str(card["need_key"])

    def assess(
        self,
        *,
        agent: "Agent",
        window: AssessmentWindow,
        previous_result: dict | None,
    ) -> dict:
        # 使用窗口内出现过的最高压力作为该需求的严重程度信号。
        pressure_max = window.effective_pressure_max.get(self.need_key, 0.0)
        pressure_last = window.effective_pressure_last.get(self.need_key, 0.0)
        pressure = _clamp01(max(pressure_max, pressure_last))
        need_change = window.need_changes().get(self.need_key, 0.0)
        decline = self._normalized_decline(agent, need_change)
        previous_mediators = {}
        if isinstance(previous_result, dict):
            previous_mediators = previous_result.get("mediators", {}) or {}

        mediators = {}
        for mediator in self.card.get("mediators", []):
            if not isinstance(mediator, dict):
                continue
            mediator_key = mediator.get("key")
            if not isinstance(mediator_key, str) or not mediator_key:
                continue
            previous_value = _as_float(previous_mediators.get(mediator_key), 0.0)
            # 理论卡公式：
            # mediator = base + pressure_weight * pressure
            #          + decline_weight * normalized_decline
            #          + previous_weight * previous_mediator
            value = (
                _as_float(mediator.get("base"), 0.0)
                + _as_float(mediator.get("pressure_weight"), 0.0) * pressure
                + _as_float(mediator.get("decline_weight"), 0.0) * decline
                + _as_float(mediator.get("previous_weight"), 0.0) * previous_value
            )
            mediators[mediator_key] = round(_clamp01(value), 3)

        level = _pressure_level(pressure)
        role_card = self._build_role_card(level, pressure, mediators)
        return {
            "need_key": self.need_key,
            "status": "theory_card_evaluated",
            "theory_card": {
                "id": self.card.get("id"),
                "display_name": self.card.get("display_name"),
                "maslow_need": self.card.get("maslow_need"),
            },
            "window": {
                "start_tick": window.start_tick,
                "end_tick": window.end_tick,
            },
            "experiences": list(window.experiences),
            "need_change": need_change,
            "effective_pressure": {
                "max": pressure_max,
                "last": pressure_last,
                "level": level,
            },
            "previous_result": previous_result,
            "mediators": mediators,
            "cognitive_behavior_templates": role_card,
            "role_card_delta": role_card,
            "source_anchors": self.card.get("source_anchors", []),
            "validation_predictions": self.card.get("validation_predictions", []),
        }

    def _normalized_decline(self, agent: "Agent", need_change: float) -> float:
        threshold = agent.satisfaction_threshold.get(self.need_key)
        if not isinstance(threshold, (int, float)) or threshold <= 0:
            return 0.0
        return _clamp01(max(0.0, -need_change) / float(threshold))

    def _build_role_card(self, level: str, pressure: float, mediators: dict[str, float]) -> dict:
        templates = self.card.get("role_card_templates", {})
        if not isinstance(templates, dict):
            templates = {}
        template = (
            templates.get(level)
            or templates.get("moderate")
            or templates.get("low")
            or templates.get("high")
            or {}
        )
        if not isinstance(template, dict):
            template = {}
        top_mediators = sorted(
            mediators.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:3]
        return {
            "source_need": self.need_key,
            "maslow_need": self.card.get("maslow_need"),
            "severity": level,
            "pressure": round(pressure, 3),
            "summary": template.get("summary", self.card.get("display_name", "")),
            "emotion_tone": template.get("emotion_tone", ""),
            "cognition": _as_text_list(template.get("cognition")),
            "behavior": _as_text_list(template.get("behavior")),
            "social_expression": _as_text_list(template.get("social_expression")),
            "online_behavior": _as_text_list(template.get("online_behavior")),
            "decision_bias": _as_text_list(template.get("decision_bias")),
            "constraints": _as_text_list(template.get("constraints")),
            "mediator_focus": [
                {"key": key, "value": value}
                for key, value in top_mediators
            ],
        }

    def recover(
        self,
        *,
        agent: "Agent",
        window: AssessmentWindow,
        previous_result: dict,
        decay_rate: float,
        clear_threshold: float,
    ) -> dict | None:
        # 恢复阶段不调用 LLM。旧中介变量会持续衰减，直到低于 clear_threshold，
        # 随后该需求评测结果会从智能体状态中移除。
        previous_mediators = previous_result.get("mediators", {}) or {}
        if not isinstance(previous_mediators, dict) or not previous_mediators:
            return None
        keep_ratio = max(0.0, 1.0 - decay_rate)
        mediators = {
            key: round(_clamp01(_as_float(value) * keep_ratio), 3)
            for key, value in previous_mediators.items()
        }
        mediators = {
            key: value
            for key, value in mediators.items()
            if value >= clear_threshold
        }
        if not mediators:
            return None

        pressure_last = window.effective_pressure_last.get(self.need_key, 0.0)
        role_card = self._build_recovery_role_card(pressure_last, mediators)
        return {
            "need_key": self.need_key,
            "status": "recovering",
            "theory_card": {
                "id": self.card.get("id"),
                "display_name": self.card.get("display_name"),
                "maslow_need": self.card.get("maslow_need"),
            },
            "window": {
                "start_tick": window.start_tick,
                "end_tick": window.end_tick,
            },
            "experiences": list(window.experiences),
            "need_change": window.need_changes().get(self.need_key, 0.0),
            "effective_pressure": {
                "max": window.effective_pressure_max.get(self.need_key, 0.0),
                "last": pressure_last,
                "level": "recovery",
            },
            "previous_result": previous_result,
            "mediators": mediators,
            "cognitive_behavior_templates": role_card,
            "role_card_delta": role_card,
            "source_anchors": self.card.get("source_anchors", []),
            "validation_predictions": self.card.get("validation_predictions", []),
        }

    def _build_recovery_role_card(self, pressure: float, mediators: dict[str, float]) -> dict:
        top_mediators = sorted(
            mediators.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:3]
        return {
            "source_need": self.need_key,
            "maslow_need": self.card.get("maslow_need"),
            "severity": "recovery",
            "pressure": round(_clamp01(pressure), 3),
            "summary": f"{self.card.get('display_name', self.need_key)}压力已回落，心理影响正在衰减。",
            "emotion_tone": "逐步恢复，但仍有轻微残余影响",
            "cognition": [
                "相关威胁或缺口线索的注意偏置正在减弱。"
            ],
            "behavior": [
                "行为应逐步回到当前任务和真实需求状态，不再过度受旧压力牵引。"
            ],
            "social_expression": [
                "表达语气逐步恢复平稳。"
            ],
            "online_behavior": [
                "相关主题的表达冲动下降。"
            ],
            "decision_bias": [
                "保留轻微惯性，但降低该需求对决策的权重。"
            ],
            "constraints": [
                "当当前压力已经恢复时，不得继续表现为高压力状态。"
            ],
            "mediator_focus": [
                {"key": key, "value": value}
                for key, value in top_mediators
            ],
        }


class LLMTheoryCardNeedEvaluator:
    """由单张理论卡约束的 LLM 评测器。

    LLM 会接收理论卡、评测窗口、当前需求状态和上一轮评测结果。
    它必须返回结构化 JSON。若输出无效，并且 fallback_to_rule 启用，
    则回退到 TheoryCardNeedEvaluator。
    """

    def __init__(
        self,
        card: dict,
        llm: "LLMClient",
        *,
        fallback_evaluator: TheoryCardNeedEvaluator | None = None,
        fallback_to_rule: bool = True,
    ):
        self.card = card
        self.llm = llm
        self.need_key = str(card["need_key"])
        self.fallback_evaluator = fallback_evaluator or TheoryCardNeedEvaluator(card)
        self.fallback_to_rule = fallback_to_rule

    def assess(
        self,
        *,
        agent: "Agent",
        window: AssessmentWindow,
        previous_result: dict | None,
    ) -> dict:
        # 同步路径，用于单元测试和兼容性调用。
        try:
            return self._assess_with_llm(agent=agent, window=window, previous_result=previous_result)
        except Exception as exc:
            return self._fallback_after_llm_error(agent, window, previous_result, exc, async_mode=False)

    async def aassess(
        self,
        *,
        agent: "Agent",
        window: AssessmentWindow,
        previous_result: dict | None,
    ) -> dict:
        # 异步路径，由 world.astep() 使用，避免 LLM 调用阻塞所有智能体的事件循环。
        try:
            return await self._aassess_with_llm(agent=agent, window=window, previous_result=previous_result)
        except Exception as exc:
            return self._fallback_after_llm_error(agent, window, previous_result, exc, async_mode=True)

    def recover(
        self,
        *,
        agent: "Agent",
        window: AssessmentWindow,
        previous_result: dict,
        decay_rate: float,
        clear_threshold: float,
    ) -> dict | None:
        # 即使处于 LLM 模式，恢复阶段也保持确定性，避免反复要求 LLM “撤销”旧心理影响。
        return self.fallback_evaluator.recover(
            agent=agent,
            window=window,
            previous_result=previous_result,
            decay_rate=decay_rate,
            clear_threshold=clear_threshold,
        )

    def _assess_with_llm(
        self,
        *,
        agent: "Agent",
        window: AssessmentWindow,
        previous_result: dict | None,
    ) -> dict:
        system, user = self._build_prompt(agent, window, previous_result)
        raw = self.llm.generate(system, user, response_format=JSON_OBJECT_RESPONSE_FORMAT)
        return self._build_result_from_raw(agent, window, previous_result, raw)

    async def _aassess_with_llm(
        self,
        *,
        agent: "Agent",
        window: AssessmentWindow,
        previous_result: dict | None,
    ) -> dict:
        system, user = self._build_prompt(agent, window, previous_result)
        raw = await self.llm.agenerate(system, user, response_format=JSON_OBJECT_RESPONSE_FORMAT)
        return self._build_result_from_raw(agent, window, previous_result, raw)

    def _fallback_after_llm_error(
        self,
        agent: "Agent",
        window: AssessmentWindow,
        previous_result: dict | None,
        exc: Exception,
        *,
        async_mode: bool,
    ) -> dict:
        """LLM 输出异常时统一执行规则回退。"""

        if not self.fallback_to_rule:
            raise exc
        mode = "async LLM" if async_mode else "LLM"
        logger.warning(
            "[Psychology] %s assessment failed for %s need=%s: %s; fallback to theory card",
            mode,
            agent.id,
            self.need_key,
            exc,
        )
        return self.fallback_evaluator.assess(
            agent=agent,
            window=window,
            previous_result=previous_result,
        )

    def _build_result_from_raw(
        self,
        agent: "Agent",
        window: AssessmentWindow,
        previous_result: dict | None,
        raw: str,
    ) -> dict:
        """统一解析 LLM 原文并转成标准评测结果。"""

        payload = self._parse_llm_json(raw)
        return self._build_result_from_payload(agent, window, previous_result, payload)

    def _pressure_snapshot(self, window: AssessmentWindow) -> dict:
        """统一生成本需求在窗口内的压力快照。"""

        pressure_max = window.effective_pressure_max.get(self.need_key, 0.0)
        pressure_last = window.effective_pressure_last.get(self.need_key, 0.0)
        pressure = _clamp01(max(pressure_max, pressure_last))
        return {
            "max": pressure_max,
            "last": pressure_last,
            "pressure": pressure,
            "level": _pressure_level(pressure),
        }

    def _build_prompt(
        self,
        agent: "Agent",
        window: AssessmentWindow,
        previous_result: dict | None,
    ) -> tuple[str, str]:
        # prompt 保留在本模块内，以匹配项目已有 LLM 调用风格：
        # system 文本 + JSON user payload，然后调用 llm.generate/agenerate。
        pressure_snapshot = self._pressure_snapshot(window)
        need_change = window.need_changes().get(self.need_key, 0.0)
        recent_history = list(getattr(agent, "history", []) or [])[-8:]
        system = (
            "你是智能体心理评测器。请根据理论卡、评测窗口经历、需求变化、有效压力和上一轮评测结果，"
            "评估该智能体当前由该需求缺口引发的心理中介变量与动态角色卡。"
            "心理评测只能描述认知偏置、情绪倾向、表达风格、社交倾向和行动偏好，"
            "不能替代需求数值、工具规则、地图观测或真实经历。"
            "不得编造经历、对象 ID、资源数量、医学诊断或未给出的事实。"
            "中介变量取值必须在 0 到 1 之间。"
            "只输出 JSON，不要输出额外文字。"
        )
        user = json.dumps(
            {
                "agent_id": agent.id,
                "need_key": self.need_key,
                "current_task": agent.task,
                "current_focus": agent.current_focus,
                "current_need_state": {
                    "satisfaction": dict(agent.satisfaction),
                    "need_gap": dict(agent.need_gap),
                    "pressure_memory": dict(agent.pressure_memory),
                    "load_saturation": dict(agent.load_saturation),
                    "effective_pressure": dict(agent.effective_pressure),
                    "threshold": dict(agent.satisfaction_threshold),
                },
                "assessment_window": {
                    "start_tick": window.start_tick,
                    "end_tick": window.end_tick,
                    "experiences": list(window.experiences),
                    "recent_history": recent_history,
                    "need_change": need_change,
                    "effective_pressure": {
                        "max": pressure_snapshot["max"],
                        "last": pressure_snapshot["last"],
                        "level": pressure_snapshot["level"],
                    },
                },
                "theory_card": self.card,
                "previous_result": previous_result,
                "output_schema": {
                    "mediators": {"mediator_key": "float in [0,1]"},
                    "role_card_delta": {
                        "summary": "short Chinese summary",
                        "emotion_tone": "short Chinese phrase",
                        "cognition": ["short Chinese instruction"],
                        "behavior": ["short Chinese instruction"],
                        "social_expression": ["short Chinese instruction"],
                        "online_behavior": ["short Chinese instruction"],
                        "decision_bias": ["short Chinese instruction"],
                        "constraints": ["short Chinese constraint"],
                    },
                    "reason": "short Chinese explanation",
                    "evidence": ["short evidence strings"],
                    "confidence": "float in [0,1]",
                },
            },
            ensure_ascii=False,
        )
        return system, user

    def _build_result_from_payload(
        self,
        agent: "Agent",
        window: AssessmentWindow,
        previous_result: dict | None,
        payload: dict,
    ) -> dict:
        # 将 LLM 原始 JSON 转换为与确定性理论卡评测器一致的结果结构。
        # 下游代码不需要关心结果来自 LLM 还是规则评测器。
        pressure_snapshot = self._pressure_snapshot(window)
        need_change = window.need_changes().get(self.need_key, 0.0)
        mediators = self._validated_mediators(payload.get("mediators"))
        if not mediators:
            raise ValueError("LLM psychological assessment output has no valid mediators")
        role_card = self._validated_role_card(
            payload.get("role_card_delta"),
            pressure_snapshot["level"],
            pressure_snapshot["pressure"],
            mediators,
        )
        reason = payload.get("reason", "")
        if not isinstance(reason, str) or not reason:
            reason = "LLM 根据理论卡和窗口上下文完成心理评测。"
        evidence = payload.get("evidence", [])
        if not isinstance(evidence, list):
            evidence = [str(evidence)]
        confidence = _clamp01(_as_float(payload.get("confidence"), 0.5))
        return {
            "need_key": self.need_key,
            "status": "llm_theory_card_evaluated",
            "theory_card": {
                "id": self.card.get("id"),
                "display_name": self.card.get("display_name"),
                "maslow_need": self.card.get("maslow_need"),
            },
            "window": {
                "start_tick": window.start_tick,
                "end_tick": window.end_tick,
            },
            "experiences": list(window.experiences),
            "need_change": need_change,
            "effective_pressure": {
                "max": pressure_snapshot["max"],
                "last": pressure_snapshot["last"],
                "level": pressure_snapshot["level"],
            },
            "previous_result": previous_result,
            "mediators": mediators,
            "cognitive_behavior_templates": role_card,
            "role_card_delta": role_card,
            "source_anchors": self.card.get("source_anchors", []),
            "validation_predictions": self.card.get("validation_predictions", []),
            "reason": reason,
            "evidence": [str(item) for item in evidence[:6]],
            "confidence": confidence,
        }

    def _validated_mediators(self, value) -> dict[str, float]:
        # 只接受理论卡中声明过的中介变量键。数值会被限制在 [0, 1]，
        # 以保证角色卡影响有界。
        if not isinstance(value, dict):
            return {}
        allowed = {
            mediator.get("key")
            for mediator in self.card.get("mediators", [])
            if isinstance(mediator, dict) and isinstance(mediator.get("key"), str)
        }
        mediators = {}
        for key, raw_value in value.items():
            if not isinstance(key, str) or (allowed and key not in allowed):
                continue
            mediators[key] = round(_clamp01(_as_float(raw_value, 0.0)), 3)
        return mediators

    def _validated_role_card(
        self,
        value,
        level: str,
        pressure: float,
        mediators: dict[str, float],
    ) -> dict:
        # 将可选角色卡字段标准化为稳定列表。必要约束会始终追加，
        # 因为角色卡只是偏置层，不是世界规则或工具调用的替代品。
        if not isinstance(value, dict):
            value = {}
        top_mediators = sorted(
            mediators.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:3]
        constraints = _as_text_list(value.get("constraints"))
        _merge_unique(
            constraints,
            "心理角色卡只影响认知偏置、表达风格、社交倾向和行动偏好，不直接替代需求目标或工具规则。",
        )
        return {
            "source_need": self.need_key,
            "maslow_need": self.card.get("maslow_need"),
            "severity": level,
            "pressure": round(pressure, 3),
            "summary": str(value.get("summary") or self.card.get("display_name", "")),
            "emotion_tone": str(value.get("emotion_tone") or ""),
            "cognition": _as_text_list(value.get("cognition")),
            "behavior": _as_text_list(value.get("behavior")),
            "social_expression": _as_text_list(value.get("social_expression")),
            "online_behavior": _as_text_list(value.get("online_behavior")),
            "decision_bias": _as_text_list(value.get("decision_bias")),
            "constraints": constraints,
            "mediator_focus": [
                {"key": key, "value": value}
                for key, value in top_mediators
            ],
        }

    def _parse_llm_json(self, raw: str) -> dict:
        return parse_json_object(raw, context="LLM psychological assessment output")


class PlaceholderNeedEvaluator:
    """没有理论卡的需求键使用的兜底评测器。"""

    def __init__(self, need_key: str):
        self.need_key = need_key

    def assess(
        self,
        *,
        agent: "Agent",
        window: AssessmentWindow,
        previous_result: dict | None,
    ) -> dict:
        return {
            "need_key": self.need_key,
            "status": "placeholder",
            "window": {
                "start_tick": window.start_tick,
                "end_tick": window.end_tick,
            },
            "experiences": list(window.experiences),
            "need_change": window.need_changes().get(self.need_key, 0.0),
            "effective_pressure": {
                "max": window.effective_pressure_max.get(self.need_key, 0.0),
                "last": window.effective_pressure_last.get(self.need_key, 0.0),
            },
            "previous_result": previous_result,
            "mediators": {},
            "cognitive_behavior_templates": {},
            "note": "心理评测器接口已触发，具体理论卡/LLM评测逻辑暂未实现。",
        }


class PsychologicalArbiter:
    """将多个被激活需求的评测结果合并为一个角色卡增量。"""

    def integrate(
        self,
        *,
        agent: "Agent",
        evaluator_results: dict[str, dict],
        previous_integrated_result: dict | None,
    ) -> dict:
        # 主导需求取窗口压力最高的激活需求。其他字段合并时会去重。
        activated_needs = list(evaluator_results.keys())
        dominant_need = ""
        dominant_pressure = -1.0
        role_card_delta = {
            "summary": "",
            "dominant_need": "",
            "activated_needs": activated_needs,
            "emotion_tone": "",
            "cognition": [],
            "behavior": [],
            "social_expression": [],
            "online_behavior": [],
            "decision_bias": [],
            "constraints": [],
            "mediator_focus": [],
        }
        summaries = []
        emotion_tones = []
        mediator_focus = []
        merged_mediators = {}

        for need_key, result in evaluator_results.items():
            pressure = _as_float((result.get("effective_pressure") or {}).get("max"), 0.0)
            if pressure > dominant_pressure:
                dominant_need = need_key
                dominant_pressure = pressure
            for mediator_key, value in (result.get("mediators") or {}).items():
                merged_mediators[f"{need_key}.{mediator_key}"] = value

            card = result.get("role_card_delta") or result.get("cognitive_behavior_templates") or {}
            if not isinstance(card, dict):
                continue
            summary = card.get("summary")
            if isinstance(summary, str) and summary:
                summaries.append(summary)
            emotion_tone = card.get("emotion_tone")
            if isinstance(emotion_tone, str) and emotion_tone:
                emotion_tones.append(emotion_tone)
            for field in [
                "cognition",
                "behavior",
                "social_expression",
                "online_behavior",
                "decision_bias",
                "constraints",
            ]:
                _merge_unique(role_card_delta[field], card.get(field))
            for item in card.get("mediator_focus", []) or []:
                if isinstance(item, dict):
                    mediator_focus.append({
                        "need_key": need_key,
                        "key": item.get("key"),
                        "value": item.get("value"),
                    })

        role_card_delta["dominant_need"] = dominant_need
        role_card_delta["summary"] = "；".join(summaries)
        role_card_delta["emotion_tone"] = "；".join(emotion_tones)
        role_card_delta["mediator_focus"] = sorted(
            mediator_focus,
            key=lambda item: _as_float(item.get("value"), 0.0),
            reverse=True,
        )[:5]
        _merge_unique(
            role_card_delta["constraints"],
            "心理角色卡只影响认知偏置、表达风格、社交倾向和行动偏好，不直接替代需求目标或工具规则。",
        )
        return {
            "status": "integrated_theory_cards",
            "activated_needs": activated_needs,
            "dominant_need": dominant_need,
            "evaluator_results": evaluator_results,
            "previous_integrated_result": previous_integrated_result,
            "mediators": merged_mediators,
            "role_card_delta": role_card_delta,
            "note": "多个理论卡评测结果已整合为统一动态角色卡。",
        }


class PsychologicalAssessmentCoordinator:
    """管理所有智能体的心理评测生命周期。

    协调器负责判断窗口何时成熟、哪些需求评测器被激活、是否进入恢复流程，
    以及最终结果如何写回智能体。
    """

    def __init__(
        self,
        config: "AgentConfig",
        llm: "LLMClient | None" = None,
        evaluators: dict[str, PsychologicalEvaluator] | None = None,
        arbiter: PsychologicalArbiter | None = None,
    ):
        self.config = config
        self.llm = llm
        if evaluators is None:
            repository = TheoryCardRepository.load_default()
            self.evaluators = self._build_default_evaluators(repository)
        else:
            self.evaluators = evaluators
        self.arbiter = arbiter or PsychologicalArbiter()

    def _build_default_evaluators(self, repository: TheoryCardRepository) -> dict[str, PsychologicalEvaluator]:
        # 在 LLM 模式下，每张理论卡都会获得一个带确定性回退的 LLM 评测器。
        # 如果没有 LLM 客户端，协调器会自动使用确定性理论卡评测器。
        use_llm = self.config.psychological_assessment_mode == "llm" and self.llm is not None
        if use_llm:
            return {
                need_key: LLMTheoryCardNeedEvaluator(
                    card,
                    self.llm,
                    fallback_to_rule=self.config.psychological_llm_fallback_to_rule,
                )
                for need_key, card in repository.by_need_key.items()
            }
        return {
            need_key: TheoryCardNeedEvaluator(card)
            for need_key, card in repository.by_need_key.items()
        }

    def ensure_agent_state(self, agent: "Agent", tick: int) -> None:
        # 第一次观察时创建一个从当前 tick 开始的评测窗口。
        if agent.psychological_assessment_window is None:
            agent.psychological_assessment_window = AssessmentWindow(start_tick=tick)
            agent.psychological_assessment_window.update_from_agent(agent)

    def observe_agent_tick(self, agent: "Agent", tick: int) -> None:
        # 每个 tick 在 maybe_assess/amaybe_assess 前调用。
        # 它会记录最新压力快照和一条压缩后的经历文本。
        self.ensure_agent_state(agent, tick)
        window = agent.psychological_assessment_window
        window.update_from_agent(agent)
        experience = self._experience_summary(agent, tick)
        if experience:
            window.add_experience(experience)

    def _ready_window(self, agent: "Agent", tick: int) -> AssessmentWindow | None:
        """返回已到评测间隔的窗口；未成熟时返回 None。"""

        self.ensure_agent_state(agent, tick)
        interval = max(1, self.config.psychological_assessment_interval)
        window = agent.psychological_assessment_window
        if tick - window.start_tick + 1 < interval:
            return None
        window.end_tick = tick
        return window

    def _assess_need_sync(self, agent: "Agent", window: AssessmentWindow, need_key: str) -> dict:
        """同步评测单个需求，并写回该需求上一轮结果。"""

        evaluator = self.evaluators.get(need_key) or PlaceholderNeedEvaluator(need_key)
        previous_result = agent.last_need_assessments.get(need_key)
        result = evaluator.assess(
            agent=agent,
            window=window,
            previous_result=previous_result,
        )
        agent.last_need_assessments[need_key] = result
        return result

    async def _assess_need_async(self, agent: "Agent", window: AssessmentWindow, need_key: str) -> dict:
        """异步评测单个需求；不支持异步的评测器走同步接口。"""

        evaluator = self.evaluators.get(need_key) or PlaceholderNeedEvaluator(need_key)
        previous_result = agent.last_need_assessments.get(need_key)
        if hasattr(evaluator, "aassess"):
            result = await evaluator.aassess(
                agent=agent,
                window=window,
                previous_result=previous_result,
            )
        else:
            result = evaluator.assess(
                agent=agent,
                window=window,
                previous_result=previous_result,
            )
        agent.last_need_assessments[need_key] = result
        return result

    def maybe_assess(self, agent: "Agent", tick: int) -> dict | None:
        # 同步兼容路径。实际运行时使用 amaybe_assess_all()。
        window = self._ready_window(agent, tick)
        if window is None:
            return None
        activated_needs = self._activated_needs(window)
        if not activated_needs:
            return self._handle_no_activated_needs(agent, window, tick)

        evaluator_results: dict[str, dict] = {}
        for need_key in activated_needs:
            evaluator_results[need_key] = self._assess_need_sync(agent, window, need_key)

        return self._store_integrated_assessment(agent, window, tick, evaluator_results)

    async def amaybe_assess(self, agent: "Agent", tick: int) -> dict | None:
        # 单个智能体的异步路径。逻辑与 maybe_assess() 对齐；
        # 当评测器支持异步 LLM 调用时使用 aassess()。
        window = self._ready_window(agent, tick)
        if window is None:
            return None
        activated_needs = self._activated_needs(window)
        if not activated_needs:
            return self._handle_no_activated_needs(agent, window, tick)

        evaluator_results: dict[str, dict] = {}
        for need_key in activated_needs:
            evaluator_results[need_key] = await self._assess_need_async(agent, window, need_key)

        return self._store_integrated_assessment(agent, window, tick, evaluator_results)

    async def amaybe_assess_all(self, agents: list["Agent"], tick: int) -> None:
        # 并发运行所有智能体的评测，并隔离单个智能体的失败；
        # 一个坏的 LLM 响应不会中断整个世界 tick。
        import asyncio

        results = await asyncio.gather(
            *(self.amaybe_assess(agent, tick) for agent in agents),
            return_exceptions=True,
        )
        for agent, result in zip(agents, results):
            if isinstance(result, Exception):
                logger.warning(
                    "[Psychology] assessment failed for %s at tick=%d: %s",
                    agent.id,
                    tick,
                    result,
                )

    def _activated_needs(self, window: AssessmentWindow) -> list[str]:
        # 如果某需求在本窗口内的最大有效压力达到配置阈值，则激活该需求。
        activated = []
        for need_key, max_pressure in window.effective_pressure_max.items():
            threshold = self.config.psychological_pressure_thresholds.get(
                need_key,
                self.config.psychological_pressure_default_threshold,
            )
            if max_pressure >= threshold:
                activated.append(need_key)
        return activated

    def _recovering_results(self, agent: "Agent", window: AssessmentWindow) -> dict[str, dict]:
        # 如果没有新的压力超过激活阈值，旧心理影响仍可通过 recover() 衰减。
        recovery_threshold = self.config.psychological_recovery_pressure_threshold
        decay_rate = self.config.psychological_mediator_decay_rate
        clear_threshold = self.config.psychological_mediator_clear_threshold
        recovering: dict[str, dict] = {}
        for need_key, previous_result in list(agent.last_need_assessments.items()):
            if not isinstance(previous_result, dict):
                continue
            last_pressure = window.effective_pressure_last.get(need_key, 0.0)
            max_pressure = window.effective_pressure_max.get(need_key, 0.0)
            if max(last_pressure, max_pressure) >= recovery_threshold:
                continue
            evaluator = self.evaluators.get(need_key)
            if evaluator is None or not hasattr(evaluator, "recover"):
                continue
            result = evaluator.recover(
                agent=agent,
                window=window,
                previous_result=previous_result,
                decay_rate=decay_rate,
                clear_threshold=clear_threshold,
            )
            if result is None:
                agent.last_need_assessments.pop(need_key, None)
                continue
            recovering[need_key] = result
            agent.last_need_assessments[need_key] = result
        return recovering

    def _handle_no_activated_needs(self, agent: "Agent", window: AssessmentWindow, tick: int) -> dict:
        # 没有激活压力时：要么恢复上一轮中介变量残余，要么记录 skipped 并开启新窗口。
        recovering_results = self._recovering_results(agent, window)
        if recovering_results:
            if len(recovering_results) > 1:
                result = self.arbiter.integrate(
                    agent=agent,
                    evaluator_results=recovering_results,
                    previous_integrated_result=agent.last_psychological_assessment,
                )
                result["status"] = "recovering_integrated"
            else:
                need_key, recovery_result = next(iter(recovering_results.items()))
                result = {
                    "status": "recovering_single",
                    "activated_needs": [],
                    "recovering_needs": [need_key],
                    "evaluator_results": {need_key: recovery_result},
                    "role_card_delta": recovery_result.get(
                        "role_card_delta",
                        recovery_result.get("cognitive_behavior_templates", {}),
                    ),
                }
            result["tick"] = tick
            result["window"] = {
                "start_tick": window.start_tick,
                "end_tick": window.end_tick,
            }
            agent.last_psychological_assessment = result
            agent.psychological_assessment_window = self._new_window(agent, tick + 1)
            logger.info(
                "[Psychology] tick=%d agent=%s recovering_needs=%s status=%s",
                tick,
                agent.id,
                list(recovering_results.keys()),
                result.get("status"),
            )
            return result

        result = {
            "tick": tick,
            "status": "skipped",
            "activated_needs": [],
            "window": {
                "start_tick": window.start_tick,
                "end_tick": window.end_tick,
            },
            "reason": "本评测窗口内没有任何有效压力超过阈值。",
        }
        agent.last_psychological_assessment = result
        agent.psychological_assessment_window = self._new_window(agent, tick + 1)
        return result

    def _store_integrated_assessment(
        self,
        agent: "Agent",
        window: AssessmentWindow,
        tick: int,
        evaluator_results: dict[str, dict],
    ) -> dict:
        # 单个评测器结果直接存储；多个激活需求则交给裁判整合。
        # 最终结果会被后续 prompt 使用。
        if len(evaluator_results) > 1:
            integrated = self.arbiter.integrate(
                agent=agent,
                evaluator_results=evaluator_results,
                previous_integrated_result=agent.last_psychological_assessment,
            )
        else:
            need_key, result = next(iter(evaluator_results.items()))
            status = "single_evaluator"
            if result.get("status") == "llm_theory_card_evaluated":
                status = "llm_single_evaluator"
            integrated = {
                "status": status,
                "activated_needs": [need_key],
                "evaluator_results": {need_key: result},
                "role_card_delta": result.get(
                    "role_card_delta",
                    result.get("cognitive_behavior_templates", {}),
                ),
            }

        integrated["tick"] = tick
        integrated["window"] = {
            "start_tick": window.start_tick,
            "end_tick": window.end_tick,
        }
        agent.last_psychological_assessment = integrated
        agent.psychological_assessment_window = self._new_window(agent, tick + 1)
        logger.info(
            "[Psychology] tick=%d agent=%s activated_needs=%s status=%s",
            tick,
            agent.id,
            integrated.get("activated_needs", []),
            integrated.get("status"),
        )
        return integrated

    def _new_window(self, agent: "Agent", start_tick: int) -> AssessmentWindow:
        return AssessmentWindow(start_tick=start_tick)

    def _experience_summary(self, agent: "Agent", tick: int) -> str:
        # 保持窗口紧凑：每个 tick 只追加最新一条 history。
        # LLM prompt 会另外传入 recent_history。
        if not agent.history:
            return ""
        return f"t={tick}: {agent.history[-1]}"
