from __future__ import annotations
import json
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from persona.llm.interface import JSON_OBJECT_RESPONSE_FORMAT
from persona.llm.json_utils import parse_json_object
from persona.logger import get_logger

if TYPE_CHECKING:
    from persona.agents.agent import Agent
    from persona.llm.interface import LLMClient

logger = get_logger(__name__)

MAX_RETRIES = 2
NO_ACTION_DECISION = '{"think": "本轮没有得到可执行动作，选择不行动", "action": {}}'
NO_MEMORY_PLAN = '{"think": "本轮没有得到有效记忆查询计划，沿用保守召回", "context": "world", "queries": []}'


class ActionParser:
    """校验行动 LLM 输出，保证结果是 think/action JSON。"""

    def parse_action(self, text: str) -> str:
        action, error = self.parse_action_with_error(text)
        if error:
            logger.warning("ActionParser: %s，原始输出: %r", error, text)
            return ""
        return action

    def parse_action_with_error(self, text: str) -> tuple[str, str]:
        """返回合法 JSON 字符串和错误信息。"""

        content = text.strip()
        if not content:
            return json.dumps({"think": "LLM 输出为空，本轮不行动", "action": {}}, ensure_ascii=False), ""
        if re.search(r"</?Action>", content):
            return "", "输出必须是纯 JSON 对象字符串，不能包含 <Action> 标签"
        if re.search(r"</?Think>", content):
            return "", "输出必须是纯 JSON 对象字符串，不能包含 <Think> 标签"
        try:
            data = parse_json_object(content, context="action decision")
        except ValueError as exc:
            return "", f"输出不是合法 JSON：{exc}"
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
        return json.dumps({"think": data["think"].strip(), "action": action}, ensure_ascii=False), ""

    def parse_action_only_with_error(self, text: str) -> tuple[str, str]:
        """第二次决策拒绝再次输出记忆查询字段。"""

        try:
            data = parse_json_object(str(text or ""), context="action-only decision")
        except ValueError as exc:
            return "", f"输出不是合法 JSON：{exc}"
        unexpected = set(data) - {"think", "action"}
        if unexpected:
            return "", f"第二次决策只允许 think/action，禁止字段：{','.join(sorted(unexpected))}"
        return self.parse_action_with_error(text)


class MemoryQueryPlanParser:
    """校验记忆查询计划，禁止 planner 输出任意 SQL 或未知查询类型。"""

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
    MAX_SEMANTIC_QUERIES = 1
    MAX_SEMANTIC_QUERY_CHARS = 500
    MAX_LIST_ITEMS = 10
    MAX_FIELD_CHARS = 120
    LIST_FIELDS = {
        "target_agent_ids",
        "entity_ids",
        "post_ids",
        "author_ids",
        "kinds",
        "source_types",
        "memory_types",
    }
    SCALAR_FIELDS = {
        "target_agent_id",
        "entity_id",
        "post_id",
        "author_id",
        "entity_type",
        "task",
    }

    def parse_plan(self, text: str) -> str:
        plan, error = self.parse_plan_with_error(text)
        if error:
            logger.warning("MemoryQueryPlanParser: %s，原始输出: %r", error, text)
            return self.empty_plan()
        return plan

    def parse_plan_with_error(self, text: str) -> tuple[str, str]:
        """返回合法 JSON 查询计划和错误信息。"""

        content = str(text or "").strip()
        if not content:
            return self.empty_plan(), ""
        if content.startswith("```"):
            content = content.removeprefix("```json").removeprefix("```").strip()
            if content.endswith("```"):
                content = content[:-3].strip()
        try:
            data = parse_json_object(content, context="memory query plan")
        except ValueError as exc:
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
        semantic_query_count = 0
        for index, query in enumerate(queries[: self.MAX_QUERIES]):
            if not isinstance(query, dict):
                return "", f"queries[{index}] 必须是对象"
            query_type = str(query.get("type") or "")
            if query_type not in self.ALLOWED_QUERY_TYPES:
                return "", f"queries[{index}].type 不支持：{query_type}"
            normalized = dict(query)
            normalized["type"] = query_type
            # 限制结构化查询参数，避免超长数组扩张 SQLite IN 条件。
            for field in self.LIST_FIELDS:
                if field not in normalized:
                    continue
                raw_values = normalized.get(field)
                values = raw_values if isinstance(raw_values, list) else [raw_values]
                bounded_values = []
                seen_values = set()
                for value in values[: self.MAX_LIST_ITEMS]:
                    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
                        continue
                    text_value = str(value).strip()
                    if (
                        not text_value
                        or len(text_value) > self.MAX_FIELD_CHARS
                        or text_value in seen_values
                    ):
                        continue
                    seen_values.add(text_value)
                    bounded_values.append(text_value if isinstance(value, str) else value)
                normalized[field] = bounded_values
            for field in self.SCALAR_FIELDS:
                if field not in normalized:
                    continue
                value = normalized.get(field)
                if isinstance(value, bool) or not isinstance(value, (str, int, float)):
                    normalized.pop(field, None)
                    continue
                text_value = str(value).strip()
                if not text_value or len(text_value) > self.MAX_FIELD_CHARS:
                    normalized.pop(field, None)
                elif isinstance(value, str):
                    normalized[field] = text_value
            if query_type == "semantic":
                # 无效、超长或重复的语义查询只丢弃当前条目，不影响结构化查询。
                semantic_query = query.get("query")
                if not isinstance(semantic_query, str):
                    continue
                semantic_query = semantic_query.strip()
                if not semantic_query or len(semantic_query) > self.MAX_SEMANTIC_QUERY_CHARS:
                    continue
                if semantic_query_count >= self.MAX_SEMANTIC_QUERIES:
                    continue
                normalized["query"] = semantic_query
                semantic_query_count += 1
            normalized["intent"] = str(normalized.get("intent") or "").strip()[: self.MAX_FIELD_CHARS]
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


class Policy(ABC):
    @abstractmethod
    def decide(self, agent: "Agent", observation: str, mem_info) -> str:
        raise NotImplementedError


class LLMPolicy(Policy):
    """普通行动决策 policy，要求 LLM 输出含 think/action 的 JSON 字符串。"""

    def __init__(self, llm_client: "LLMClient", prompt_builder, parser):
        self.llm = llm_client
        self.prompt_builder = prompt_builder
        self.parser = parser
        self.memory_parser = MemoryQueryPlanParser()

    def decide_or_plan(self, agent: "Agent", observation: str, *, context: str) -> tuple[str, str]:
        """第一次决策返回 action 或非空记忆查询计划。"""

        system, user = self.prompt_builder.build_initial_decision(agent, observation, context)
        raw = self.llm.generate(system, user, response_format=JSON_OBJECT_RESPONSE_FORMAT)
        result_type, payload, error = self._parse_initial_decision(raw, context)
        for attempt in range(MAX_RETRIES):
            if not error:
                break
            retry_user = self._retry_prompt(user, raw, error, target="initial_decision")
            raw = self.llm.generate(system, retry_user, response_format=JSON_OBJECT_RESPONSE_FORMAT)
            result_type, payload, error = self._parse_initial_decision(raw, context)
        if error:
            logger.warning("[%s] 第一次决策重试%d次后仍无效，本轮不行动", agent.id, MAX_RETRIES)
            return "action", NO_ACTION_DECISION
        return result_type, payload

    async def adecide_or_plan(self, agent: "Agent", observation: str, *, context: str) -> tuple[str, str]:
        """异步执行第一次行动或记忆查询决策。"""

        system, user = self.prompt_builder.build_initial_decision(agent, observation, context)
        raw = await self.llm.agenerate(system, user, response_format=JSON_OBJECT_RESPONSE_FORMAT)
        result_type, payload, error = self._parse_initial_decision(raw, context)
        for attempt in range(MAX_RETRIES):
            if not error:
                break
            retry_user = self._retry_prompt(user, raw, error, target="initial_decision")
            raw = await self.llm.agenerate(system, retry_user, response_format=JSON_OBJECT_RESPONSE_FORMAT)
            result_type, payload, error = self._parse_initial_decision(raw, context)
        if error:
            logger.warning("[%s] 异步第一次决策重试%d次后仍无效，本轮不行动", agent.id, MAX_RETRIES)
            return "action", NO_ACTION_DECISION
        return result_type, payload

    def decide_after_memory(self, agent: "Agent", observation: str, mem_info) -> str:
        """使用独立提示词执行查询后的行动决策。"""

        system, user = self.prompt_builder.build_after_memory_decision(agent, observation, mem_info)
        raw = self.llm.generate(system, user, response_format=JSON_OBJECT_RESPONSE_FORMAT)
        action, error = self.parser.parse_action_only_with_error(raw)
        for attempt in range(MAX_RETRIES):
            if not error:
                break
            retry_user = self._retry_prompt(user, raw, error, target="action_after_memory")
            raw = self.llm.generate(system, retry_user, response_format=JSON_OBJECT_RESPONSE_FORMAT)
            action, error = self.parser.parse_action_only_with_error(raw)
        return NO_ACTION_DECISION if error else action

    async def adecide_after_memory(self, agent: "Agent", observation: str, mem_info) -> str:
        """异步执行查询后的行动决策。"""

        system, user = self.prompt_builder.build_after_memory_decision(agent, observation, mem_info)
        raw = await self.llm.agenerate(system, user, response_format=JSON_OBJECT_RESPONSE_FORMAT)
        action, error = self.parser.parse_action_only_with_error(raw)
        for attempt in range(MAX_RETRIES):
            if not error:
                break
            retry_user = self._retry_prompt(user, raw, error, target="action_after_memory")
            raw = await self.llm.agenerate(system, retry_user, response_format=JSON_OBJECT_RESPONSE_FORMAT)
            action, error = self.parser.parse_action_only_with_error(raw)
        return NO_ACTION_DECISION if error else action

    def _parse_initial_decision(self, raw: str, context: str) -> tuple[str, str, str]:
        """严格区分行动 JSON 与记忆查询 JSON。"""

        if not str(raw or "").strip():
            return "", "", "第一次决策输出为空"
        try:
            data = parse_json_object(raw, context="initial action-or-memory decision")
        except ValueError as exc:
            return "", "", f"输出不是合法 JSON：{exc}"
        has_action = "action" in data
        has_plan = "queries" in data or "context" in data
        if has_action == has_plan:
            return "", "", "必须且只能输出 action 或 queries/context 其中一种结构"
        if has_action:
            action, error = self.parser.parse_action_with_error(raw)
            return "action", action, error
        plan, error = self.memory_parser.parse_plan_with_error(raw)
        if error:
            return "memory", "", error
        plan_data = json.loads(plan)
        if plan_data.get("context") != context:
            return "memory", "", f"context 必须为 {context}"
        if not plan_data.get("queries"):
            return "memory", "", "queries 必须是非空数组"
        return "memory", plan, ""

    def decide(self, agent: "Agent", observation: str, mem_info) -> str:
        system, user = self.prompt_builder.build(agent, observation, mem_info)
        raw = self.llm.generate(system, user, response_format=JSON_OBJECT_RESPONSE_FORMAT)
        if not raw.strip():
            logger.warning("[%s] LLM returned an empty response; skip action this tick", agent.id)
            return NO_ACTION_DECISION
        logger.debug("[%s] LLM 输出: %s", agent.id, raw)
        action, error = self.parser.parse_action_with_error(raw)

        for attempt in range(MAX_RETRIES):
            if not error:
                break
            logger.warning("[%s] 解析失败（第%d次），错误：%s，尝试重试", agent.id, attempt + 1, error)
            retry_user = self._retry_prompt(user, raw, error, target="action")
            raw = self.llm.generate(system, retry_user, response_format=JSON_OBJECT_RESPONSE_FORMAT)
            logger.debug("[%s] 重试%d LLM 输出: %s", agent.id, attempt + 1, raw)
            action, error = self.parser.parse_action_with_error(raw)

        if error:
            logger.warning("[%s] 重试%d次后仍解析失败，本轮跳过行动", agent.id, MAX_RETRIES)
            return NO_ACTION_DECISION
        return action

    async def adecide(self, agent: "Agent", observation: str, mem_info) -> str:
        system, user = self.prompt_builder.build(agent, observation, mem_info)
        raw = await self.llm.agenerate(system, user, response_format=JSON_OBJECT_RESPONSE_FORMAT)
        if not raw.strip():
            logger.warning("[%s] LLM returned an empty response; skip action this tick", agent.id)
            return NO_ACTION_DECISION
        logger.debug("[%s] LLM 输出: %s", agent.id, raw)
        action, error = self.parser.parse_action_with_error(raw)

        for attempt in range(MAX_RETRIES):
            if not error:
                break
            logger.warning("[%s] 解析失败（第%d次），错误：%s，尝试重试", agent.id, attempt + 1, error)
            retry_user = self._retry_prompt(user, raw, error, target="action")
            raw = await self.llm.agenerate(system, retry_user, response_format=JSON_OBJECT_RESPONSE_FORMAT)
            logger.debug("[%s] 重试%d LLM 输出: %s", agent.id, attempt + 1, raw)
            action, error = self.parser.parse_action_with_error(raw)

        if error:
            logger.warning("[%s] 重试%d次后仍解析失败，本轮跳过行动", agent.id, MAX_RETRIES)
            return NO_ACTION_DECISION
        return action

    def _retry_prompt(self, user: str, raw: str, error: str, *, target: str) -> str:
        if target == "initial_decision":
            instruction = (
                "请重新输出合法 JSON，并且只能二选一：直接行动时输出 think/action；"
                "查询记忆时输出 think/context/非空 queries。禁止混合两种结构。"
            )
        elif target == "memory_plan":
            instruction = (
                "请重新输出合法 JSON 对象字符串，顶层必须包含 think、context、queries；"
                "queries 只能使用允许的 type，不能输出 SQL 或行动工具。"
            )
        elif target == "action_after_memory":
            instruction = (
                "这是记忆查询后的第二次决策。只能输出 think/action，"
                "禁止输出 context、queries 或再次请求记忆。"
            )
        else:
            instruction = (
                "请检查上述错误，重新输出符合格式要求的内容。"
                "必须只输出一个合法 JSON 对象字符串，顶层包含 think 和 action 字段；"
                "think 必须说明本次选择动作的思考过程和理由；"
                "不要输出 Markdown、额外解释、<Think> 或 <Action> 标签。"
            )
        return (
            f"{user}\n\n"
            f"[上一次输出]\n{raw}\n\n"
            f"[错误信息]\n{error}\n\n"
            f"{instruction}"
        )


class LLMMemoryPlannerPolicy:
    """行动前的 LLM 记忆查询规划器，输出受控 JSON 查询计划。"""

    def __init__(self, llm_client: "LLMClient", prompt_builder, parser):
        self.llm = llm_client
        self.prompt_builder = prompt_builder
        self.parser = parser

    def plan(
        self,
        agent: "Agent",
        observation: str,
        context: str = "world",
        recalled_memories: list[str] | None = None,
    ) -> str:
        """根据当前输入和基础召回，只规划仍缺失的信息。"""

        system, user = self.prompt_builder.build(
            agent,
            observation,
            context,
            recalled_memories=recalled_memories or [],
        )
        raw = self.llm.generate(system, user, response_format=JSON_OBJECT_RESPONSE_FORMAT)
        if not raw.strip():
            logger.warning("[%s] memory planner returned an empty response", agent.id)
            return self._fallback_plan(context)
        logger.debug("[%s] memory planner 输出: %s", agent.id, raw)
        plan, error = self.parser.parse_plan_with_error(raw)

        for attempt in range(MAX_RETRIES):
            if not error:
                break
            logger.warning("[%s] memory planner 解析失败（第%d次）：%s", agent.id, attempt + 1, error)
            retry_user = self._retry_prompt(user, raw, error)
            raw = self.llm.generate(system, retry_user, response_format=JSON_OBJECT_RESPONSE_FORMAT)
            logger.debug("[%s] memory planner 重试%d 输出: %s", agent.id, attempt + 1, raw)
            plan, error = self.parser.parse_plan_with_error(raw)

        if error:
            logger.warning("[%s] memory planner 重试%d次后仍解析失败，使用保守计划", agent.id, MAX_RETRIES)
            return self._fallback_plan(context)
        return plan

    async def aplan(
        self,
        agent: "Agent",
        observation: str,
        context: str = "world",
        recalled_memories: list[str] | None = None,
    ) -> str:
        """异步规划基础召回之后的一次补充查询。"""

        system, user = self.prompt_builder.build(
            agent,
            observation,
            context,
            recalled_memories=recalled_memories or [],
        )
        raw = await self.llm.agenerate(system, user, response_format=JSON_OBJECT_RESPONSE_FORMAT)
        if not raw.strip():
            logger.warning("[%s] memory planner returned an empty response", agent.id)
            return self._fallback_plan(context)
        logger.debug("[%s] memory planner 输出: %s", agent.id, raw)
        plan, error = self.parser.parse_plan_with_error(raw)

        for attempt in range(MAX_RETRIES):
            if not error:
                break
            logger.warning("[%s] memory planner 解析失败（第%d次）：%s", agent.id, attempt + 1, error)
            retry_user = self._retry_prompt(user, raw, error)
            raw = await self.llm.agenerate(system, retry_user, response_format=JSON_OBJECT_RESPONSE_FORMAT)
            logger.debug("[%s] memory planner 重试%d 输出: %s", agent.id, attempt + 1, raw)
            plan, error = self.parser.parse_plan_with_error(raw)

        if error:
            logger.warning("[%s] memory planner 重试%d次后仍解析失败，使用保守计划", agent.id, MAX_RETRIES)
            return self._fallback_plan(context)
        return plan

    def _retry_prompt(self, user: str, raw: str, error: str) -> str:
        return (
            f"{user}\n\n"
            f"[上一次输出]\n{raw}\n\n"
            f"[错误信息]\n{error}\n\n"
            "请重新输出合法 JSON 对象字符串，顶层必须包含 think、context、queries；"
            "queries 只能使用允许的 type，不能输出 SQL 或行动工具。"
        )

    def _fallback_plan(self, context: str) -> str:
        return NO_MEMORY_PLAN.replace('"context": "world"', f'"context": "{context}"')
