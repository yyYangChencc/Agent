from __future__ import annotations
import re
import json
from persona.logger import get_logger

logger = get_logger(__name__)


class ActionParser:
    def parse_action(self, text: str) -> str:
        action, error = self.parse_action_with_error(text)
        if error:
            logger.warning("ActionParser: %s，原始输出: %r", error, text)
            return ""
        return action

    def parse_action_with_error(self, text: str) -> tuple[str, str]:
        """返回 (json_decision_str, error_msg)。成功时 error_msg 为空字符串。"""
        content = text.strip()
        if not content:
            return json.dumps({"think": "LLM 输出为空，本轮不行动", "action": {}}, ensure_ascii=False), ""
        if re.search(r"</?Action>", content):
            return "", "输出必须是纯 JSON 对象字符串，不能包含 <Action> 标签"
        if re.search(r"</?Think>", content):
            return "", "输出必须是纯 JSON 对象字符串，不能包含 <Think> 标签"
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            return "", f"输出不是合法 JSON：{e}"
        if not isinstance(data, dict):
            return "", "JSON 顶层必须是对象"
        if "think" not in data:
            return "", "JSON 顶层缺少 think 字段"
        if not isinstance(data["think"], str) or not data["think"].strip():
            return "", "JSON 字段 think 必须是非空字符串"
        if "action" not in data:
            return "", "JSON 顶层缺少 action 字段"
        action = data["action"]
        if not isinstance(action, dict):
            return "", "JSON 字段 action 必须是对象"
        args = action.get("args")
        if args is not None and not isinstance(args, dict):
            return "", "JSON 字段 action.args 必须是对象"
        return json.dumps(
            {
                "think": data["think"].strip(),
                "action": action,
            },
            ensure_ascii=False,
        ), ""


class MemoryQueryPlanParser:
    """校验 LLM 生成的记忆查询计划，禁止 planner 输出任意 SQL 或未知查询类型。"""

    ALLOWED_CONTEXTS = {"world", "social", "conversation", "opinion_assessment"}
    ALLOWED_QUERY_TYPES = {
        "person_profile",
        "entity_state",
        "event_history",
        "social_post",
        "derived_memory",
        "semantic",
    }
    MAX_QUERIES = 5
    MAX_LIMIT = 5

    def parse_plan(self, text: str) -> str:
        plan, error = self.parse_plan_with_error(text)
        if error:
            logger.warning("MemoryQueryPlanParser: %s，原始输出: %r", error, text)
            return self.empty_plan()
        return plan

    def parse_plan_with_error(self, text: str) -> tuple[str, str]:
        """返回 (json_plan_str, error_msg)。成功时 error_msg 为空字符串。"""

        content = str(text or "").strip()
        if not content:
            return self.empty_plan(), ""
        if content.startswith("```"):
            content = content.removeprefix("```json").removeprefix("```").strip()
            if content.endswith("```"):
                content = content[:-3].strip()
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            return "", f"输出不是合法 JSON：{exc}"
        if not isinstance(data, dict):
            return "", "JSON 顶层必须是对象"
        think = data.get("think")
        if not isinstance(think, str) or not think.strip():
            return "", "JSON 字段 think 必须是非空字符串"
        context = str(data.get("context") or "world")
        if context not in self.ALLOWED_CONTEXTS:
            return "", f"context 不支持：{context}"
        queries = data.get("queries")
        if queries is None:
            queries = []
        if not isinstance(queries, list):
            return "", "queries 必须是数组"

        normalized_queries = []
        for index, query in enumerate(queries[: self.MAX_QUERIES]):
            if not isinstance(query, dict):
                return "", f"queries[{index}] 必须是对象"
            query_type = str(query.get("type") or "")
            if query_type not in self.ALLOWED_QUERY_TYPES:
                return "", f"queries[{index}].type 不支持：{query_type}"
            normalized = dict(query)
            normalized["type"] = query_type
            normalized["intent"] = str(normalized.get("intent") or "")
            normalized["limit"] = self._normalize_limit(normalized.get("limit"))
            normalized_queries.append(normalized)

        return json.dumps(
            {
                "think": think.strip(),
                "context": context,
                "queries": normalized_queries,
            },
            ensure_ascii=False,
        ), ""

    def empty_plan(self) -> str:
        return json.dumps(
            {
                "think": "未生成有效记忆查询计划，本轮沿用保守召回",
                "context": "world",
                "queries": [],
            },
            ensure_ascii=False,
        )

    def _normalize_limit(self, value) -> int:
        try:
            number = int(value)
        except (TypeError, ValueError):
            number = 3
        return max(1, min(self.MAX_LIMIT, number))
