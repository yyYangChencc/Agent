from __future__ import annotations

import copy
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator

from persona.llm.interface import LLMClient


@dataclass(frozen=True)
class LLMTraceContext:
    """描述一次逻辑 LLM 调用所属的智能体、时间步和阶段。"""

    store: "LLMTraceStore"
    call_id: str
    agent_id: str
    tick: int
    stage: str
    metadata: dict[str, Any] = field(default_factory=dict)


_CURRENT_TRACE: ContextVar[LLMTraceContext | None] = ContextVar(
    "current_llm_debug_trace",
    default=None,
)


class LLMTraceStore:
    """按 tick 和智能体保存有界的完整 LLM 调试记录。"""

    def __init__(self, retention_ticks: int = 3) -> None:
        self.retention_ticks = max(2, int(retention_ticks))
        self._records: dict[int, dict[str, list[dict[str, Any]]]] = {}
        self._next_call_id = 1
        self._next_sequence = 1
        self._lock = threading.RLock()

    def allocate_call_id(self) -> str:
        with self._lock:
            call_id = f"llm-call-{self._next_call_id:08d}"
            self._next_call_id += 1
            return call_id

    def record_attempt(
        self,
        context: LLMTraceContext,
        *,
        system_prompt: str,
        user_prompt: str,
        response: str,
        response_format: dict | None,
        duration_ms: float,
        status: str,
        error_type: str = "",
        error: str = "",
    ) -> dict[str, Any]:
        """追加一次底层生成尝试，并返回保存后的记录副本。"""

        with self._lock:
            agent_records = self._records.setdefault(context.tick, {}).setdefault(context.agent_id, [])
            attempt = 1 + sum(item["call_id"] == context.call_id for item in agent_records)
            record = {
                "schema_version": 1,
                "sequence": self._next_sequence,
                "call_id": context.call_id,
                "agent_id": context.agent_id,
                "tick": context.tick,
                "stage": context.stage,
                "attempt": attempt,
                "system_prompt": str(system_prompt or ""),
                "user_prompt": str(user_prompt or ""),
                "response": str(response or ""),
                "response_format": copy.deepcopy(response_format),
                "status": str(status),
                "error_type": str(error_type or ""),
                "error": str(error or ""),
                "validation_error": "",
                "duration_ms": round(max(0.0, float(duration_ms)), 3),
                "metadata": copy.deepcopy(context.metadata),
            }
            self._next_sequence += 1
            agent_records.append(record)
            return copy.deepcopy(record)

    def annotate_current_call(self, context: LLMTraceContext, error: str) -> None:
        """把解析或校验错误附加到当前逻辑调用的最后一次响应。"""

        text = str(error or "")
        if not text:
            return
        with self._lock:
            records = self._records.get(context.tick, {}).get(context.agent_id, [])
            for record in reversed(records):
                if record["call_id"] != context.call_id:
                    continue
                record["validation_error"] = text
                if record["status"] == "completed":
                    record["status"] = "invalid_response"
                return

    def records_for(self, agent_id: str, tick: int) -> list[dict[str, Any]]:
        with self._lock:
            records = self._records.get(int(tick), {}).get(str(agent_id), [])
            return copy.deepcopy(sorted(records, key=lambda item: item["sequence"]))

    def count_for(self, agent_id: str, tick: int) -> int:
        with self._lock:
            return len(self._records.get(int(tick), {}).get(str(agent_id), []))

    def prune(self, current_tick: int) -> None:
        """保留当前 tick 及其之前有限数量的记录。"""

        minimum_tick = int(current_tick) - self.retention_ticks + 1
        with self._lock:
            for tick in list(self._records):
                if tick < minimum_tick:
                    self._records.pop(tick, None)

    def clear(self) -> None:
        with self._lock:
            self._records.clear()
            self._next_call_id = 1
            self._next_sequence = 1


@contextmanager
def trace_llm_call(
    llm,
    *,
    agent_id: str,
    tick: int,
    stage: str,
    metadata: dict[str, Any] | None = None,
) -> Iterator[None]:
    """为一个逻辑调用绑定追踪上下文；未启用包装器时保持空操作。"""

    store = getattr(llm, "trace_store", None)
    if not isinstance(store, LLMTraceStore):
        yield
        return
    context = LLMTraceContext(
        store=store,
        call_id=store.allocate_call_id(),
        agent_id=str(agent_id),
        tick=int(tick),
        stage=str(stage),
        metadata=copy.deepcopy(metadata or {}),
    )
    token = _CURRENT_TRACE.set(context)
    try:
        yield
    except Exception as exc:
        store.annotate_current_call(context, f"{type(exc).__name__}: {exc}")
        raise
    finally:
        _CURRENT_TRACE.reset(token)


def annotate_current_llm_trace(error: str) -> None:
    """记录调用方已经识别出的响应格式错误。"""

    context = _CURRENT_TRACE.get()
    if context is not None:
        context.store.annotate_current_call(context, error)


def record_manual_llm_attempt(
    *,
    system_prompt: str,
    user_prompt: str,
    response: str,
    duration_ms: float,
    status: str = "completed",
    error_type: str = "",
    error: str = "",
) -> None:
    """记录未经过 LLMClient 接口的本地生成模型调用。"""

    context = _CURRENT_TRACE.get()
    if context is None:
        return
    context.store.record_attempt(
        context,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response=response,
        response_format=None,
        duration_ms=duration_ms,
        status=status,
        error_type=error_type,
        error=error,
    )


class TracingLLMClient(LLMClient):
    """在不改变现有 LLM 行为的前提下记录完整生成请求和响应。"""

    def __init__(self, inner: LLMClient, trace_store: LLMTraceStore) -> None:
        self.inner = inner
        self.trace_store = trace_store

    def generate(self, system: str, user: str, *, response_format: dict | None = None) -> str:
        context = _CURRENT_TRACE.get()
        if context is None or context.store is not self.trace_store:
            return self.inner.generate(system, user, response_format=response_format)
        started_at = time.perf_counter()
        try:
            response = self.inner.generate(system, user, response_format=response_format)
        except Exception as exc:
            self.trace_store.record_attempt(
                context,
                system_prompt=system,
                user_prompt=user,
                response="",
                response_format=response_format,
                duration_ms=(time.perf_counter() - started_at) * 1000,
                status="failed",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        self.trace_store.record_attempt(
            context,
            system_prompt=system,
            user_prompt=user,
            response=response,
            response_format=response_format,
            duration_ms=(time.perf_counter() - started_at) * 1000,
            status="completed" if str(response or "") else "empty_response",
        )
        return response

    async def agenerate(self, system: str, user: str, *, response_format: dict | None = None) -> str:
        context = _CURRENT_TRACE.get()
        if context is None or context.store is not self.trace_store:
            return await self.inner.agenerate(system, user, response_format=response_format)
        started_at = time.perf_counter()
        try:
            response = await self.inner.agenerate(system, user, response_format=response_format)
        except Exception as exc:
            self.trace_store.record_attempt(
                context,
                system_prompt=system,
                user_prompt=user,
                response="",
                response_format=response_format,
                duration_ms=(time.perf_counter() - started_at) * 1000,
                status="failed",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        self.trace_store.record_attempt(
            context,
            system_prompt=system,
            user_prompt=user,
            response=response,
            response_format=response_format,
            duration_ms=(time.perf_counter() - started_at) * 1000,
            status="completed" if str(response or "") else "empty_response",
        )
        return response

    def get_embeddings(self, text: str) -> list[float]:
        return self.inner.get_embeddings(text)

    def get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        return self.inner.get_embeddings_batch(texts)

    async def aget_embeddings(self, text: str) -> list[float]:
        return await self.inner.aget_embeddings(text)

    async def aget_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        return await self.inner.aget_embeddings_batch(texts)

    def __getattr__(self, name: str):
        return getattr(self.inner, name)
