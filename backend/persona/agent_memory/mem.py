from __future__ import annotations
import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any
from persona.agent_memory.controller import MemoryController
from persona.agent_memory.query_builder import (
    DEFAULT_SEMANTIC_QUERY_MAX_BYTES,
    bound_utf8_text,
    build_semantic_observation_text,
)
from persona.agent_memory.structured_store import StructuredMemoryStore
from persona.logger import get_logger

logger = get_logger(__name__)

try:
    import chromadb
    from chromadb.config import Settings
except ModuleNotFoundError:
    chromadb = None

    def Settings(**kwargs):
        """缺少 chromadb 时的占位配置，供内存后备客户端使用。"""

        return kwargs

DEFAULT_MEMORY_TYPE = "episodic"
MEMORY_TYPE_ALIASES = {
    "trajectory": "episodic",
    "micro_reflection": "reflective",
    "system": "semantic",
}
CONTEXT_MEMORY_TYPES = {
    "world": ["semantic", "procedural", "episodic", "reflective"],
    "social": ["social", "semantic", "episodic", "reflective"],
    "conversation": ["social", "episodic", "reflective", "semantic"],
}


class _InMemoryCollection:
    """缺少 chromadb 时使用的最小内存 collection。"""

    def __init__(self, name: str, metadata: dict | None = None):
        self.name = name
        self.metadata = metadata or {}
        self._rows: dict[str, dict] = {}

    def count(self) -> int:
        return len(self._rows)

    def upsert(self, *, embeddings=None, documents=None, metadatas=None, ids=None) -> None:
        for index, memory_id in enumerate(ids or []):
            self._rows[str(memory_id)] = {
                "id": str(memory_id),
                "embedding": (embeddings or [[]])[index],
                "document": (documents or [""])[index],
                "metadata": (metadatas or [{}])[index],
            }

    def query(self, *, query_embeddings=None, query_texts=None, n_results=3, where=None, **kwargs) -> dict:
        rows = [row for row in self._rows.values() if self._matches_where(row.get("metadata") or {}, where)]
        rows.sort(key=lambda row: float((row.get("metadata") or {}).get("saved_at", 0) or 0), reverse=True)
        selected = rows[:max(0, int(n_results or 0))]
        return {
            "ids": [[row["id"] for row in selected]],
            "documents": [[row["document"] for row in selected]],
            "metadatas": [[row["metadata"] for row in selected]],
            "distances": [[0.0 for _ in selected]],
        }

    def get(self, *, limit=None, include=None, where=None, **kwargs) -> dict:
        rows = [row for row in self._rows.values() if self._matches_where(row.get("metadata") or {}, where)]
        rows.sort(key=lambda row: float((row.get("metadata") or {}).get("saved_at", 0) or 0), reverse=True)
        if limit is not None:
            rows = rows[:max(0, int(limit))]
        return {
            "ids": [row["id"] for row in rows],
            "documents": [row["document"] for row in rows],
            "metadatas": [row["metadata"] for row in rows],
        }

    def delete(self, *, ids=None, where=None, **kwargs) -> None:
        """按 ID 或 metadata 条件删除内存后备向量。"""

        if ids is not None:
            for memory_id in ids:
                self._rows.pop(str(memory_id), None)
            return
        if where:
            stale_ids = [
                memory_id
                for memory_id, row in self._rows.items()
                if self._matches_where(row.get("metadata") or {}, where)
            ]
            for memory_id in stale_ids:
                self._rows.pop(memory_id, None)

    def _matches_where(self, metadata: dict, where: dict | None) -> bool:
        if not where:
            return True
        for key, expected in where.items():
            if key == "$and" and isinstance(expected, list):
                return all(self._matches_where(metadata, item) for item in expected)
            if key == "$or" and isinstance(expected, list):
                return any(self._matches_where(metadata, item) for item in expected)
            actual = metadata.get(key)
            if isinstance(expected, dict):
                if "$in" in expected and actual not in expected["$in"]:
                    return False
                if "$eq" in expected and actual != expected["$eq"]:
                    return False
            elif actual != expected:
                return False
        return True


class _InMemoryChromaClient:
    """缺少 chromadb 时的进程内后备客户端，只用于本地 mock/测试。"""

    def __init__(self, *args, **kwargs):
        self._collections: dict[str, _InMemoryCollection] = {}

    def get_or_create_collection(self, *, name: str, metadata: dict | None = None, **kwargs):
        if name not in self._collections:
            self._collections[name] = _InMemoryCollection(name, metadata)
        return self._collections[name]

    def reset(self) -> None:
        self._collections.clear()


def _build_chroma_client(path: str, settings):
    """构建 Chroma 客户端；缺包时自动降级为内存实现。"""

    if chromadb is None:
        logger.warning("未安装 chromadb，使用内存记忆后备；该模式不持久化向量记忆。")
        return _InMemoryChromaClient(path=path, settings=settings)
    return chromadb.PersistentClient(path=path, settings=settings)


class MultiAgentMemoryManager:
    """多智能体长期记忆管理器。

    每个智能体对应一个 ChromaDB collection；存储时写入 embedding 和元数据，
    检索时先做向量召回，再按任务、需求急迫度、重要性和时间等因素重排。
    """

    def __init__(
        self,
        llm_client,
        persist_directory: str = "./chroma_agents",
        structured_db_path: str | None = None,
        *,
        config: Any | None = None,
    ):
        self.client = _build_chroma_client(
            path=persist_directory,
            settings=Settings(allow_reset=True),
        )
        self.llm_client = llm_client
        self.config = config
        self.agent_collections: dict = {}
        self._collection_lock = threading.Lock()
        self._embedding_cache: dict[str, list[float]] = {}
        self._embedding_cache_lock = threading.Lock()
        self._retrieval_cache: dict[tuple, tuple[int, list[str]]] = {}
        self._retrieval_cache_lock = threading.Lock()
        self._last_maintenance_results: dict[str, dict[str, Any]] = {}
        if structured_db_path is None:
            structured_db_path = str(Path(persist_directory) / "structured_memory.sqlite3")
        # SQLite 负责结构化状态和证据；MemoryController 负责把业务 payload 分流到 SQLite/Chroma。
        self.structured_store = StructuredMemoryStore(structured_db_path)
        self.controller = MemoryController(self.structured_store, self)

    def _get_collection_name(self, agent_id: str) -> str:
        return f"agent_{agent_id}_memory"

    def get_agent_collection(self, agent_id: str):
        if agent_id in self.agent_collections:          # 快路径，无锁
            return self.agent_collections[agent_id]["collection"]
        with self._collection_lock:
            if agent_id not in self.agent_collections:  # 双重检查
                collection_name = self._get_collection_name(agent_id)
                collection = self.client.get_or_create_collection(
                    name=collection_name,
                    metadata={
                        "agent_id": agent_id,
                        "created_at": time.time(),
                        "purpose": "agent_long_term_memory",
                    },
                )
                self.agent_collections[agent_id] = {
                    "collection": collection,
                    "name": collection_name,
                }
                logger.info("为智能体 %s 初始化记忆存储", agent_id)
        return self.agent_collections[agent_id]["collection"]

    def close(self) -> None:
        structured_store = getattr(self, "structured_store", None)
        if structured_store is not None:
            structured_store.close()

    def store_agent_memory(self, agent_id: str, memory_text: str, world_time: int = 0, **metadata) -> str:
        """写入长期语义记忆。

        文本进入 Chroma 参与语义召回，同时双写到 SQLite 的 derived_memories，便于
        后续 API 展示、访问统计和冲突调解。
        """

        structured_metadata = dict(metadata)
        full_metadata = self._normalize_metadata(agent_id, world_time, dict(metadata))
        raw_metadata = structured_metadata
        structured_metadata = {**raw_metadata, **full_metadata}
        for key, value in raw_metadata.items():
            if isinstance(value, (list, tuple, dict)):
                structured_metadata[key] = value
        memory_id = self._derived_memory_id(agent_id, memory_text, world_time, full_metadata)
        # SQLite 是长期记忆的事实底座；embedding 失败不能导致整条经历丢失。
        self._record_derived_memory(agent_id, memory_id, memory_text, world_time, structured_metadata)
        collection = self.get_agent_collection(agent_id)
        embedding = self._get_embedding(memory_text)
        if not embedding:
            logger.warning("[%s] 获取 embedding 失败，仅保留 SQLite 长期记忆: %r", agent_id, memory_text[:60])
            self._invalidate_retrieval_cache(agent_id)
            return memory_id
        try:
            collection.upsert(
                embeddings=[embedding],
                documents=[memory_text],
                metadatas=[full_metadata],
                ids=[memory_id],
            )
        except Exception as exc:
            # 向量索引失败时保留已写入的 SQLite 长期记忆，避免中断智能体流程。
            logger.warning("[%s] 向量记忆写入失败，仅保留 SQLite 长期记忆: %s", agent_id, exc, exc_info=True)
        self._invalidate_retrieval_cache(agent_id)
        return memory_id

    async def astore_agent_memory(self, agent_id: str, memory_text: str, world_time: int = 0, **metadata) -> str:
        """异步写入长期语义记忆，并保持与同步路径相同的 SQLite 双写行为。"""

        structured_metadata = dict(metadata)
        full_metadata = self._normalize_metadata(agent_id, world_time, dict(metadata))
        raw_metadata = structured_metadata
        structured_metadata = {**raw_metadata, **full_metadata}
        for key, value in raw_metadata.items():
            if isinstance(value, (list, tuple, dict)):
                structured_metadata[key] = value
        memory_id = self._derived_memory_id(agent_id, memory_text, world_time, full_metadata)
        # 异步路径与同步路径使用同一份 SQLite 优先写入契约。
        self._record_derived_memory(agent_id, memory_id, memory_text, world_time, structured_metadata)
        collection = self.get_agent_collection(agent_id)
        embedding = await self._aget_embedding(memory_text)
        if not embedding:
            logger.warning("[%s] 获取 embedding 失败，仅保留 SQLite 长期记忆: %r", agent_id, memory_text[:60])
            self._invalidate_retrieval_cache(agent_id)
            return memory_id
        try:
            collection.upsert(
                embeddings=[embedding],
                documents=[memory_text],
                metadatas=[full_metadata],
                ids=[memory_id],
            )
        except Exception as exc:
            # 异步路径使用相同的 SQLite 降级契约。
            logger.warning("[%s] 异步向量记忆写入失败，仅保留 SQLite 长期记忆: %s", agent_id, exc, exc_info=True)
        self._invalidate_retrieval_cache(agent_id)
        return memory_id

    def _derived_memory_id(
        self,
        agent_id: str,
        memory_text: str,
        world_time: int,
        metadata: dict[str, Any],
    ) -> str:
        """同一经历保持幂等，不同经历中的相同摘要不再互相覆盖。"""

        episode_id = str(metadata.get("episode_id") or "")
        if episode_id:
            identity = f"{episode_id}|{metadata.get('source_type', '')}|{memory_text}"
        else:
            identity = memory_text
        digest = hashlib.md5(identity.encode()).hexdigest()[:10]
        return f"{agent_id}_{digest}"

    def store_observation(self, agent_id: str, observation: Any) -> None:
        """兼容旧调用的 observe 写入入口，实际分类逻辑在 MemoryController 中。"""

        controller = getattr(self, "controller", None)
        if controller is not None:
            controller.store_observation(agent_id, observation)

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
        """兼容旧调用的动作结果写入入口，记录已执行动作而不是未执行计划。"""

        controller = getattr(self, "controller", None)
        if controller is not None:
            controller.store_action_result(
                agent_id,
                decision=decision,
                feedback=feedback,
                reward=reward,
                world_time=world_time,
                episode_id=episode_id,
                state_before=state_before,
                state_after=state_after,
                outcome=outcome,
                need_events=need_events,
            )
            self._invalidate_retrieval_cache(agent_id)

    def store_need_event(self, agent_id: str, event: Any, *, episode_id: str = "") -> None:
        """把需求变化从运行日志同步为可检索的自身经历。"""

        controller = getattr(self, "controller", None)
        if controller is not None:
            controller.store_need_event(agent_id, event, episode_id=episode_id)
            self._invalidate_retrieval_cache(agent_id)

    def store_psychological_assessment(
        self,
        agent_id: str,
        assessment: Any,
        *,
        episode_id: str = "",
    ) -> None:
        """保存心理评测及其来源窗口，供后续经历巩固和实验审计。"""

        controller = getattr(self, "controller", None)
        if controller is not None:
            controller.store_psychological_assessment(agent_id, assessment, episode_id=episode_id)
            self._invalidate_retrieval_cache(agent_id)

    def store_social_browse(self, agent_id: str, browse_payload: Any) -> None:
        """写入社交平台浏览结果，保留为 manager 方法以减少业务模块依赖。"""

        controller = getattr(self, "controller", None)
        if controller is not None:
            controller.store_social_browse(agent_id, browse_payload)
            self._invalidate_retrieval_cache(agent_id)

    def store_social_feedback(self, agent_id: str, feedback_payload: Any, *, world_time: int | None = None) -> None:
        """写入社交操作反馈，实际帖子快照和事件拆分由控制层完成。"""

        controller = getattr(self, "controller", None)
        if controller is not None:
            controller.store_social_feedback(agent_id, feedback_payload, world_time=world_time)
            self._invalidate_retrieval_cache(agent_id)

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
        """写入对话消息和回复，保持 Agent 侧只提交结构化 payload。"""

        controller = getattr(self, "controller", None)
        if controller is not None:
            controller.store_conversation(
                agent_id,
                messages=messages,
                reply=reply,
                observation=observation,
                world_time=world_time,
                episode_id=episode_id,
            )
            self._invalidate_retrieval_cache(agent_id)

    def store_opinion_assessment(self, agent_id: str, assessment: Any) -> None:
        """写入观念评测结果，作为 reflective 记忆证据。"""

        controller = getattr(self, "controller", None)
        if controller is not None:
            controller.store_opinion_assessment(agent_id, assessment)
            self._invalidate_retrieval_cache(agent_id)

    def retrieve_context(
        self,
        agent_id: str,
        observation: Any,
        task: str,
        urgency: dict,
        satisfaction_threshold: dict,
        n_results: int = 3,
        context: str = "world",
        timeout_seconds: float | None = None,
    ) -> list[str]:
        """统一检索入口。

        新路径先经过 MemoryController 做结构化召回；若测试替身或旧对象没有
        controller，则退回原来的 smart_retrieve，保证既有调用兼容。
        """

        cache_key, cache_time = self._retrieval_cache_key(
            "retrieve_context",
            agent_id,
            observation,
            task,
            urgency,
            satisfaction_threshold,
            n_results,
            context,
            None,
        )
        cached = self._get_retrieval_cache(cache_key, cache_time)
        if cached is not None:
            return list(cached)

        controller = getattr(self, "controller", None)
        if controller is None:
            semantic_observation = build_semantic_observation_text(observation, context)
            result = self.smart_retrieve(
                agent_id,
                semantic_observation,
                task,
                urgency,
                satisfaction_threshold,
                n_results=n_results,
                context=context,
            )
            self._set_retrieval_cache(cache_key, cache_time, result)
            return result
        result = controller.retrieve_context(
            agent_id,
            observation,
            task,
            urgency,
            satisfaction_threshold,
            n_results=n_results,
            context=context,
            timeout_seconds=timeout_seconds,
        )
        self._set_retrieval_cache(cache_key, cache_time, result)
        return result

    async def aretrieve_context(
        self,
        agent_id: str,
        observation: Any,
        task: str,
        urgency: dict,
        satisfaction_threshold: dict,
        n_results: int = 3,
        context: str = "world",
        timeout_seconds: float | None = None,
    ) -> list[str]:
        """异步统一检索入口，兼容没有结构化控制层的旧测试对象。"""

        cache_key, cache_time = self._retrieval_cache_key(
            "retrieve_context",
            agent_id,
            observation,
            task,
            urgency,
            satisfaction_threshold,
            n_results,
            context,
            None,
        )
        cached = self._get_retrieval_cache(cache_key, cache_time)
        if cached is not None:
            return list(cached)

        controller = getattr(self, "controller", None)
        if controller is None:
            semantic_observation = build_semantic_observation_text(observation, context)
            result = await self.asmart_retrieve(
                agent_id,
                semantic_observation,
                task,
                urgency,
                satisfaction_threshold,
                n_results=n_results,
                context=context,
            )
            self._set_retrieval_cache(cache_key, cache_time, result)
            return result
        result = await controller.aretrieve_context(
            agent_id,
            observation,
            task,
            urgency,
            satisfaction_threshold,
            n_results=n_results,
            context=context,
            timeout_seconds=timeout_seconds,
        )
        self._set_retrieval_cache(cache_key, cache_time, result)
        return result

    def execute_query_plan(
        self,
        agent_id: str,
        query_plan: Any,
        observation: Any,
        task: str,
        urgency: dict,
        satisfaction_threshold: dict,
        n_results: int = 3,
        context: str = "world",
        timeout_seconds: float | None = None,
    ) -> list[str]:
        """执行 LLM planner 输出的受控记忆查询计划；旧对象退回自动召回。"""

        cache_key, cache_time = self._retrieval_cache_key(
            "execute_query_plan",
            agent_id,
            observation,
            task,
            urgency,
            satisfaction_threshold,
            n_results,
            context,
            query_plan,
        )
        cached = self._get_retrieval_cache(cache_key, cache_time)
        if cached is not None:
            return list(cached)

        controller = getattr(self, "controller", None)
        if controller is None:
            semantic_observation = build_semantic_observation_text(observation, context)
            result = self.smart_retrieve(
                agent_id,
                semantic_observation,
                task,
                urgency,
                satisfaction_threshold,
                n_results=n_results,
                context=context,
            )
            self._set_retrieval_cache(cache_key, cache_time, result)
            return result
        result = controller.execute_query_plan(
            agent_id,
            query_plan,
            observation,
            task,
            urgency,
            satisfaction_threshold,
            n_results=n_results,
            context=context,
            timeout_seconds=timeout_seconds,
        )
        self._set_retrieval_cache(cache_key, cache_time, result)
        return result

    async def aexecute_query_plan(
        self,
        agent_id: str,
        query_plan: Any,
        observation: Any,
        task: str,
        urgency: dict,
        satisfaction_threshold: dict,
        n_results: int = 3,
        context: str = "world",
        timeout_seconds: float | None = None,
    ) -> list[str]:
        """异步执行 LLM planner 输出的受控记忆查询计划。"""

        cache_key, cache_time = self._retrieval_cache_key(
            "execute_query_plan",
            agent_id,
            observation,
            task,
            urgency,
            satisfaction_threshold,
            n_results,
            context,
            query_plan,
        )
        cached = self._get_retrieval_cache(cache_key, cache_time)
        if cached is not None:
            return list(cached)

        controller = getattr(self, "controller", None)
        if controller is None:
            semantic_observation = build_semantic_observation_text(observation, context)
            result = await self.asmart_retrieve(
                agent_id,
                semantic_observation,
                task,
                urgency,
                satisfaction_threshold,
                n_results=n_results,
                context=context,
            )
            self._set_retrieval_cache(cache_key, cache_time, result)
            return result
        result = await controller.aexecute_query_plan(
            agent_id,
            query_plan,
            observation,
            task,
            urgency,
            satisfaction_threshold,
            n_results=n_results,
            context=context,
            timeout_seconds=timeout_seconds,
        )
        self._set_retrieval_cache(cache_key, cache_time, result)
        return result

    def smart_retrieve(
        self,
        agent_id: str,
        observation: str,
        task: str,
        urgency: dict,
        satisfaction_threshold: dict,
        n_results: int = 3,
        context: str = "world",
        memory_types: list[str] | None = None,
        current_time: int | None = None,
    ) -> list[str]:
        """构造确定性检索查询，再召回与当前任务相关的记忆。"""
        query = self._build_retrieval_query(observation, task, urgency, satisfaction_threshold)
        where = self._memory_type_where(memory_types)
        result = self.retrieve_agent_memories(
            agent_id,
            query,
            n_results=n_results,
            context=context,
            task=task,
            urgency=urgency,
            where=where,
            current_time=current_time,
        )
        logger.debug("[%s] 记忆检索结果: %s", agent_id, result)
        return result

    def retrieve_agent_memories(
        self,
        agent_id: str,
        query: str,
        n_results: int = 5,
        *,
        context: str = "world",
        task: str = "",
        urgency: dict | None = None,
        current_time: int | None = None,
        **kwargs,
    ) -> list[str]:
        """同步向量检索入口。

        先扩大召回数量，再通过 _memory_score 重新排序，避免单纯向量距离忽略任务和需求。
        """

        started_at = time.perf_counter()
        query = self._bounded_semantic_query(agent_id, query, context)
        prepared = self._prepare_vector_retrieval(agent_id, n_results, context, kwargs)
        if prepared is None:
            return []
        collection, fetch_n, allowed_memory_types = prepared

        query_embedding = self._get_embedding(query)
        if not query_embedding:
            logger.warning("[%s] 获取查询 embedding 失败，返回空记忆", agent_id)
            return []

        results = self._query_vector_collection(agent_id, collection, query_embedding, fetch_n, kwargs)
        if results is None:
            return []
        out = self._format_retrieval_output(
            results, n_results, query, task, urgency, allowed_memory_types, context, current_time
        )
        logger.debug(
            "[%s] 记忆检索完成 context=%s returned=%d elapsed=%.3fs",
            agent_id,
            context,
            len(out),
            time.perf_counter() - started_at,
        )
        return out

    async def asmart_retrieve(
        self,
        agent_id: str,
        observation: str,
        task: str,
        urgency: dict,
        satisfaction_threshold: dict,
        n_results: int = 3,
        context: str = "world",
        memory_types: list[str] | None = None,
        current_time: int | None = None,
    ) -> list[str]:
        query = self._build_retrieval_query(observation, task, urgency, satisfaction_threshold)
        where = self._memory_type_where(memory_types)
        return await self.aretrieve_agent_memories(
            agent_id,
            query,
            n_results=n_results,
            context=context,
            task=task,
            urgency=urgency,
            where=where,
            current_time=current_time,
        )

    def _build_retrieval_query(self, observation, task, urgency, satisfaction_threshold) -> str:
        """把观察、当前任务和高急迫需求合并成检索文本。"""

        parts = [observation]
        if task and task != "none":
            parts.append(f"当前任务: {task}")
        urgent = [(k, v) for k, v in urgency.items()
                  if satisfaction_threshold.get(k) and urgency.get(k, 0) > 0.5]
        if urgent:
            urgent_str = ", ".join(f"{k}急切度{v:.2f}" for k, v in urgent[:2])
            parts.append(f"紧迫需求: {urgent_str}")
        return " | ".join(parts)

    def _bounded_semantic_query(self, agent_id: str, query: str, context: str) -> str:
        """在查询 embedding 前执行统一字节限制，不影响长期记忆写入。"""

        text = str(query or "")
        max_bytes = self._config_int(
            "memory_semantic_query_max_bytes",
            DEFAULT_SEMANTIC_QUERY_MAX_BYTES,
            minimum=1,
        )
        bounded = bound_utf8_text(text, max_bytes)
        original_bytes = len(text.encode("utf-8"))
        logger.debug(
            "[%s] semantic query prepared context=%s query_chars=%d query_bytes=%d",
            agent_id,
            context,
            len(bounded),
            len(bounded.encode("utf-8")),
        )
        if bounded != text:
            logger.warning(
                "[%s] semantic query truncated context=%s original_bytes=%d max_bytes=%d",
                agent_id,
                context,
                original_bytes,
                max_bytes,
            )
        return bounded

    async def aretrieve_agent_memories(
        self,
        agent_id: str,
        query: str,
        n_results: int = 5,
        *,
        context: str = "world",
        task: str = "",
        urgency: dict | None = None,
        current_time: int | None = None,
        **kwargs,
    ) -> list[str]:
        """异步向量检索入口，与同步路径保持同样的召回和重排逻辑。"""

        started_at = time.perf_counter()
        query = self._bounded_semantic_query(agent_id, query, context)
        prepared = self._prepare_vector_retrieval(agent_id, n_results, context, kwargs)
        if prepared is None:
            return []
        collection, fetch_n, allowed_memory_types = prepared

        query_embedding = await self._aget_embedding(query)
        if not query_embedding:
            logger.warning("[%s] 获取查询 embedding 失败，返回空记忆", agent_id)
            return []

        results = self._query_vector_collection(agent_id, collection, query_embedding, fetch_n, kwargs)
        if results is None:
            return []
        out = self._format_retrieval_output(
            results, n_results, query, task, urgency, allowed_memory_types, context, current_time
        )
        logger.debug(
            "[%s] 异步记忆检索完成 context=%s returned=%d elapsed=%.3fs",
            agent_id,
            context,
            len(out),
            time.perf_counter() - started_at,
        )
        return out

    def _prepare_vector_retrieval(self, agent_id: str, n_results: int, context: str, kwargs: dict):
        """准备 Chroma 查询参数，并扩大召回数量供本地重排。"""

        collection = self.get_agent_collection(agent_id)
        count = collection.count()
        if count == 0:
            return None
        # 先多取一批，再在本地用任务/需求/重要性重排。
        fetch_n = min(max(n_results * 4, n_results), count)
        kwargs["where"] = self._build_where(agent_id, kwargs.get("where"))
        return collection, fetch_n, self._memory_types_for_context(context)

    def _query_vector_collection(self, agent_id: str, collection, query_embedding: list[float], fetch_n: int, kwargs: dict):
        """执行 Chroma 查询，并集中处理 HNSW 索引损坏。"""

        try:
            return collection.query(
                query_embeddings=[query_embedding],
                n_results=fetch_n,
                include=["documents", "metadatas", "distances"],
                **kwargs,
            )
        except Exception as e:
            if "Nothing found on disk" in str(e):
                logger.warning("[%s] HNSW 索引损坏，重建 collection", agent_id)
                self.client.delete_collection(self._get_collection_name(agent_id))
                self.agent_collections.pop(agent_id, None)
            else:
                logger.error("[%s] 记忆查询失败: %s", agent_id, e, exc_info=True)
            return None

    def _format_retrieval_output(
        self,
        results: dict,
        n_results: int,
        query: str,
        task: str,
        urgency: dict | None,
        allowed_memory_types: set[str] | None,
        context: str,
        current_time: int | None,
    ) -> list[str]:
        """统一执行本地重排和格式化。"""

        return self._format_ranked_results(
            results,
            n_results,
            query,
            task,
            urgency or {},
            allowed_memory_types,
            context=context,
            current_time=current_time,
        )

    def list_agent_memories(self, agent_id: str) -> list[dict[str, Any]]:
        """列出某个智能体的全部原始记忆，供前端记忆窗口展示。"""

        collection = self.get_agent_collection(agent_id)
        if collection.count() == 0:
            return []

        try:
            result = collection.get(
                where={"agent_id": agent_id},
                include=["documents", "metadatas"],
            )
        except Exception as e:
            logger.error("[%s] 列出记忆失败: %s", agent_id, e, exc_info=True)
            return []

        ids = result.get("ids", [])
        documents = result.get("documents", [])
        metadatas = result.get("metadatas", [])
        row_count = max(len(ids), len(documents), len(metadatas))
        rows: list[dict[str, Any]] = []
        for index in range(row_count):
            metadata = metadatas[index] if index < len(metadatas) else {}
            if not isinstance(metadata, dict):
                metadata = {}
            rows.append({
                "id": ids[index] if index < len(ids) else "",
                "document": documents[index] if index < len(documents) else "",
                "metadata": metadata,
            })

        def sort_key(row: dict[str, Any]) -> tuple[float, str]:
            metadata = row.get("metadata") or {}
            try:
                saved_at = float(metadata.get("saved_at", -1))
            except (TypeError, ValueError):
                saved_at = -1.0
            return (-saved_at, str(row.get("id", "")))

        rows.sort(key=sort_key)
        return rows

    def list_agent_structured_memories(self, agent_id: str) -> dict[str, list[dict[str, Any]]]:
        """列出 SQLite 结构化记忆；controller 不存在时返回空结构保持 API 兼容。"""

        controller = getattr(self, "controller", None)
        if controller is None:
            return {}
        return controller.list_structured_memory(agent_id)

    def maintain_agent_memory(
        self,
        agent_id: str,
        *,
        current_time: int,
        raw_event_retention: int = 100,
    ) -> dict[str, Any]:
        """按配置间隔执行原始事件和语义向量轻量维护。"""

        if not self._config_bool("memory_forgetting_enabled", True):
            result = self._empty_maintenance_result(agent_id, skipped=True)
            self._last_maintenance_results[agent_id] = dict(result)
            return result
        interval = self._config_int("memory_maintenance_interval_ticks", 10, minimum=1)
        if int(current_time) % interval != 0:
            result = self._empty_maintenance_result(agent_id, skipped=True)
            self._last_maintenance_results[agent_id] = dict(result)
            return result

        controller = getattr(self, "controller", None)
        if controller is None:
            result = self._empty_maintenance_result(agent_id)
            self._last_maintenance_results[agent_id] = dict(result)
            return result
        result = controller.maintain_agent_memory(
            agent_id,
            current_time=current_time,
            raw_event_retention=self._config_int(
                "memory_event_retention_ticks", raw_event_retention, minimum=1
            ),
            max_prunable_importance=self._config_float(
                "memory_event_max_prunable_importance", 0.5
            ),
            active_event_limit=self._config_int(
                "memory_event_active_limit_per_agent", 2000, minimum=1
            ),
            protected_importance=self._config_float("memory_protected_importance", 0.7),
        )
        result.update(self._maintain_vector_memories(agent_id, current_time=int(current_time)))
        result["skipped"] = False
        self._last_maintenance_results[agent_id] = dict(result)
        return result

    def _maintain_vector_memories(self, agent_id: str, *, current_time: int) -> dict[str, int]:
        """清理 SQLite 已失效向量，并把活跃向量控制在配置上限内。"""

        collection = self.get_agent_collection(agent_id)
        counts = self.structured_store.get_active_memory_counts(agent_id)
        if collection.count() == 0:
            return {
                **counts,
                "active_memory_vectors": 0,
                "pruned_vector_memories": 0,
                "deleted_memory_vectors": 0,
                "vector_delete_failures": 0,
            }
        try:
            result = collection.get(where={"agent_id": agent_id}, include=["metadatas"])
        except Exception as exc:
            logger.warning("[%s] 读取向量维护数据失败: %s", agent_id, exc)
            return {
                **counts,
                "active_memory_vectors": collection.count(),
                "pruned_vector_memories": 0,
                "deleted_memory_vectors": 0,
                "vector_delete_failures": 1,
            }

        ids = [str(value) for value in result.get("ids", [])]
        metadatas = result.get("metadatas", [])
        valid_ids = self.structured_store.get_valid_derived_memory_ids(ids)
        stale_ids = [memory_id for memory_id in ids if memory_id not in valid_ids]
        active_rows = []
        for index, memory_id in enumerate(ids):
            if memory_id not in valid_ids:
                continue
            metadata = metadatas[index] if index < len(metadatas) and isinstance(metadatas[index], dict) else {}
            active_rows.append((memory_id, metadata))

        active_limit = self._config_int("memory_vector_active_limit_per_agent", 500, minimum=1)
        protected_importance = self._config_float("memory_protected_importance", 0.7)
        excess = max(0, len(active_rows) - active_limit)
        removable = []
        for memory_id, metadata in active_rows:
            importance = float(metadata.get("importance", 0.5) or 0.5)
            if importance >= protected_importance:
                continue
            saved_at = float(metadata.get("saved_at", 0) or 0)
            retention_score = 0.6 * importance + 0.4 * self._recency_score(saved_at, current_time)
            removable.append((retention_score, saved_at, memory_id))
        removable.sort(key=lambda item: (item[0], item[1], item[2]))
        pruned_ids = [memory_id for _, _, memory_id in removable[:excess]]
        self.structured_store.invalidate_derived_memories(pruned_ids)

        delete_ids = list(dict.fromkeys([*stale_ids, *pruned_ids]))
        deleted_count = 0
        delete_failures = 0
        if delete_ids:
            try:
                collection.delete(ids=delete_ids)
                deleted_count = len(delete_ids)
                self._invalidate_retrieval_cache(agent_id)
            except Exception as exc:
                delete_failures = len(delete_ids)
                logger.warning("[%s] 删除失效 Chroma 向量失败: %s", agent_id, exc)

        counts = self.structured_store.get_active_memory_counts(agent_id)
        return {
            **counts,
            "active_memory_vectors": collection.count(),
            "pruned_vector_memories": len(pruned_ids),
            "deleted_memory_vectors": deleted_count,
            "vector_delete_failures": delete_failures,
        }

    def get_active_memory_counts(self, agent_id: str) -> dict[str, int]:
        """返回运行记录需要的活跃事件、长期记忆和向量数量。"""

        counts = self.structured_store.get_active_memory_counts(agent_id)
        counts["active_memory_vectors"] = int(self.get_agent_collection(agent_id).count())
        return counts

    def get_last_memory_maintenance(self, agent_id: str) -> dict[str, Any]:
        """返回当前 tick 最近一次维护统计，供历史记录器使用。"""

        return dict(getattr(self, "_last_maintenance_results", {}).get(agent_id, {}))

    def _empty_maintenance_result(self, agent_id: str, *, skipped: bool = False) -> dict[str, Any]:
        counts = {"active_memory_events": 0, "active_derived_memories": 0, "active_memory_vectors": 0}
        if getattr(self, "structured_store", None) is not None:
            counts = self.get_active_memory_counts(agent_id)
        return {
            "agent_id": agent_id,
            **counts,
            "pruned_low_value_events": 0,
            "pruned_events_to_limit": 0,
            "pruned_vector_memories": 0,
            "deleted_memory_vectors": 0,
            "vector_delete_failures": 0,
            "unresolved_conflicts": 0,
            "skipped": skipped,
        }

    def _config_value(self, name: str, default: Any) -> Any:
        # 运行时显式配置优先，保留旧客户端私有配置的兼容读取。
        config = getattr(self, "config", None) or getattr(self.llm_client, "_config", None)
        return getattr(config, name, default) if config is not None else default

    def _config_bool(self, name: str, default: bool) -> bool:
        return bool(self._config_value(name, default))

    def _config_int(self, name: str, default: int, *, minimum: int | None = None) -> int:
        try:
            value = int(self._config_value(name, default))
        except (TypeError, ValueError):
            value = int(default)
        return max(minimum, value) if minimum is not None else value

    def _config_float(self, name: str, default: float) -> float:
        try:
            return float(self._config_value(name, default))
        except (TypeError, ValueError):
            return float(default)

    def update_person_profiles_from_reflection(self, agent_id: str, *, current_time: int, llm=None) -> dict[str, Any]:
        """reflect 阶段的人物档案更新入口。"""

        controller = getattr(self, "controller", None)
        if controller is None:
            return {"agent_id": agent_id, "updated_person_profiles": 0}
        return controller.update_person_profiles_from_reflection(agent_id, current_time=current_time, llm=llm)

    async def aupdate_person_profiles_from_reflection(self, agent_id: str, *, current_time: int, llm=None) -> dict[str, Any]:
        """异步 reflect 阶段的人物档案更新入口。"""

        controller = getattr(self, "controller", None)
        if controller is None:
            return {"agent_id": agent_id, "updated_person_profiles": 0}
        return await controller.aupdate_person_profiles_from_reflection(agent_id, current_time=current_time, llm=llm)

    def get_all_agents_stats(self) -> dict:
        stats = {}
        for collection_info in self.client.list_collections():
            name = collection_info.name
            if name.startswith("agent_") and name.endswith("_memory"):
                agent_id = name[len("agent_"):-len("_memory")]
                collection = self.client.get_collection(name)
                stats[agent_id] = {
                    "memory_count": collection.count(),
                    "collection_name": name,
                    "metadata": collection.metadata,
                }
        return stats

    def _get_embedding(self, text: str) -> list[float]:
        """同步获取 embedding，并按原文缓存，减少重复 API 调用。"""

        with self._embedding_cache_lock:
            cached = self._embedding_cache.get(text)
        if cached is not None:
            return cached
        try:
            embedding = self.llm_client.get_embeddings(text)
        except Exception as e:
            logger.error("获取 embedding 失败: %s", e, exc_info=True)
            return []
        if embedding:
            with self._embedding_cache_lock:
                self._embedding_cache[text] = embedding
        return embedding

    async def _aget_embedding(self, text: str) -> list[float]:
        """异步获取 embedding，并复用同一份进程内缓存。"""

        with self._embedding_cache_lock:
            cached = self._embedding_cache.get(text)
        if cached is not None:
            return cached
        try:
            embedding = await self.llm_client.aget_embeddings(text)
        except Exception as e:
            logger.error("异步获取 embedding 失败: %s", e, exc_info=True)
            return []
        if embedding:
            with self._embedding_cache_lock:
                self._embedding_cache[text] = embedding
        return embedding

    def _normalize_metadata(self, agent_id: str, world_time: int, metadata: dict[str, Any]) -> dict[str, Any]:
        """统一记忆元数据字段，并把旧类型名映射到当前类型体系。"""

        raw_type = metadata.pop("memory_type", metadata.pop("type", DEFAULT_MEMORY_TYPE))
        memory_type = MEMORY_TYPE_ALIASES.get(str(raw_type), str(raw_type))
        full_metadata = {
            "agent_id": agent_id,
            "saved_at": world_time,
            "last_accessed_at": world_time,
            "access_count": 0,
            "importance": float(metadata.pop("importance", 0.5)),
            "confidence": float(metadata.pop("confidence", 0.5)),
            "memory_type": memory_type,
            **metadata,
        }
        return self._sanitize_metadata(full_metadata)

    def _sanitize_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """ChromaDB metadata 只接受标量；列表和复杂对象在这里转成字符串。"""

        sanitized: dict[str, str | int | float | bool] = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif isinstance(value, (list, tuple)):
                sanitized[key] = ",".join(str(item) for item in value)
            else:
                sanitized[key] = str(value)
        return sanitized

    def _memory_types_for_context(self, context: str) -> set[str] | None:
        memory_types = CONTEXT_MEMORY_TYPES.get(context)
        return set(memory_types) if memory_types else None

    def _build_where(self, agent_id: str, extra_where: dict | None) -> dict:
        filters = [{"agent_id": agent_id}]
        if extra_where:
            filters.append(extra_where)
        if len(filters) == 1:
            return filters[0]
        return {"$and": filters}

    def _memory_type_where(self, memory_types: list[str] | None) -> dict | None:
        """把 planner 指定的 Chroma 记忆类型转换为 metadata 过滤条件。"""

        values = []
        for value in memory_types or []:
            text = str(value or "").strip()
            if text:
                values.append(MEMORY_TYPE_ALIASES.get(text, text))
        values = sorted(set(values))
        if not values:
            return None
        if len(values) == 1:
            return {"memory_type": values[0]}
        return {"memory_type": {"$in": values}}

    def _format_ranked_results(
        self,
        results: dict,
        n_results: int,
        query: str,
        task: str,
        urgency: dict,
        allowed_memory_types: set[str] | None = None,
        context: str = "world",
        current_time: int | None = None,
    ) -> list[str]:
        """过滤不适合当前上下文的记忆类型，并返回重排后的文本。"""

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]
        valid_ids: set[str] | None = None
        if ids and getattr(self, "structured_store", None) is not None:
            valid_ids = self.structured_store.get_valid_derived_memory_ids(ids)
        rows = []
        for index, (doc, meta) in enumerate(zip(docs, metas)):
            memory_id = str(ids[index]) if index < len(ids) else ""
            if valid_ids is not None and memory_id not in valid_ids:
                continue
            if allowed_memory_types and meta.get("memory_type", DEFAULT_MEMORY_TYPE) not in allowed_memory_types:
                continue
            # 观念评测不能把上一轮评测结果当作本轮证据，避免分数自我强化。
            if context == "opinion_assessment" and meta.get("source_type") == "opinion_assessment":
                continue
            distance = distances[index] if index < len(distances) else 1.0
            rows.append((self._memory_score(doc, meta, distance, query, task, urgency, current_time), doc, meta))
        rows.sort(key=lambda item: item[0], reverse=True)
        return [self._format_memory(doc, meta) for _, doc, meta in rows[:n_results]]

    def _memory_score(
        self,
        doc: str,
        meta: dict,
        distance: float,
        query: str,
        task: str,
        urgency: dict,
        current_time: int | None = None,
    ) -> float:
        """记忆重排分数。

        分数综合向量相似度、重要性、任务匹配、时间、置信度和当前需求急迫度。
        repetition_penalty 用来降低和查询高度重复但信息量低的记忆。
        """

        similarity = 1.0 / (1.0 + max(0.0, float(distance)))
        importance = float(meta.get("importance", 0.5) or 0.5)
        confidence = float(meta.get("confidence", 0.5) or 0.5)
        task_match = 1.0 if task and str(meta.get("task", "")) == task else 0.0
        need_key = str(meta.get("need_key", ""))
        need_match = float(urgency.get(need_key, 0.0)) if need_key else 0.0
        saved_at = float(meta.get("saved_at", 0) or 0)
        recency = self._recency_score(saved_at, current_time)
        query_terms = {term for term in query.replace("|", " ").replace("，", " ").split() if term}
        repetition_penalty = 0.15 if query_terms and len(query_terms.intersection(set(doc.split()))) > 6 else 0.0
        return (
            0.40 * similarity
            + 0.20 * importance
            + 0.15 * task_match
            + 0.10 * recency
            + 0.10 * confidence
            + 0.05 * need_match
            - repetition_penalty
        )

    def _recency_score(self, saved_at: float, current_time: int | None) -> float:
        """按记忆年龄计算最近性；缺少当前时间时保持兼容分值。"""

        if current_time is None:
            return 1.0 if saved_at >= 0 else 0.0
        window = self._config_int("memory_recency_window_ticks", 100, minimum=1)
        age = max(0.0, float(current_time) - float(saved_at))
        return 1.0 / (1.0 + age / float(window))

    def _format_memory(self, doc: str, meta: dict) -> str:
        memory_type = meta.get("memory_type", DEFAULT_MEMORY_TYPE)
        tick = meta.get("saved_at")
        task = meta.get("task")
        need_key = meta.get("need_key")
        object_id = meta.get("object_id")
        parts = [f"[{memory_type}"]
        if tick is not None:
            parts.append(f"t={tick}")
        if task:
            parts.append(f"task={task}")
        if need_key:
            parts.append(f"need={need_key}")
        if object_id:
            parts.append(f"object={object_id}")
        return f"{' '.join(parts)}] {doc}"

    def _record_derived_memory(
        self,
        agent_id: str,
        memory_id: str,
        memory_text: str,
        world_time: int,
        metadata: dict[str, Any],
    ) -> None:
        """在向量写入前把长期记忆登记到 SQLite。"""

        controller = getattr(self, "controller", None)
        if controller is None:
            return
        try:
            controller.record_derived_memory(agent_id, memory_id, memory_text, world_time, dict(metadata))
        except Exception as exc:
            logger.debug("[%s] structured derived memory write failed: %s", agent_id, exc)

    def reset_all(self) -> None:
        """清空所有智能体的记忆，包括 Chroma 语义索引和 SQLite 结构化记忆。"""
        self.client.reset()
        structured_store = getattr(self, "structured_store", None)
        if structured_store is not None:
            structured_store.reset_all()
        self.agent_collections.clear()
        with self._embedding_cache_lock:
            self._embedding_cache.clear()
        with self._retrieval_cache_lock:
            self._retrieval_cache.clear()
        logger.info("已清空所有智能体记忆存储")

    def _retrieval_cache_key(
        self,
        method: str,
        agent_id: str,
        observation: Any,
        task: str,
        urgency: dict,
        satisfaction_threshold: dict,
        n_results: int,
        context: str,
        query_plan: Any,
    ) -> tuple | None:
        """生成短期检索缓存键；ttl<=0 时禁用缓存。"""

        ttl = self._retrieval_cache_ttl_ticks()
        if ttl <= 0:
            return None, 0
        world_time = self._extract_world_time(observation)
        cache_time = world_time // ttl if world_time >= 0 else int(time.time() // max(1, ttl))
        payload = {
            "method": method,
            "agent_id": agent_id,
            "query_text": self._cache_query_text(observation),
            "query_plan": query_plan,
            "task": task,
            "urgency": urgency,
            "satisfaction_threshold": satisfaction_threshold,
            "n_results": n_results,
            "context": context,
        }
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        return (method, agent_id, context, hashlib.md5(text.encode("utf-8")).hexdigest()), cache_time

    def _cache_query_text(self, observation: Any) -> str:
        """提取会影响召回或可见对象排除的当前输入。"""

        if isinstance(observation, str):
            return observation[:500]
        if isinstance(observation, dict):
            compact = {
                "type": observation.get("type"),
                "time": observation.get("time"),
                "task_like": observation.get("task"),
                "people": observation.get("people"),
                "objects": observation.get("objects"),
                "social": observation.get("social"),
                "posts": observation.get("posts"),
                "visible_post_ids": observation.get("visible_post_ids"),
                "account_ids": observation.get("account_ids"),
                "following_ids": observation.get("following_ids"),
                "receiver_id": observation.get("receiver_id"),
            }
            return json.dumps(compact, ensure_ascii=False, sort_keys=True, default=str)
        return str(observation)[:500]

    def _get_retrieval_cache(self, key: tuple | None, cache_time: int) -> list[str] | None:
        if key is None:
            return None
        with self._retrieval_cache_lock:
            cached = self._retrieval_cache.get(key)
        if cached is None:
            return None
        cached_time, value = cached
        if cached_time != cache_time:
            return None
        return list(value)

    def _set_retrieval_cache(self, key: tuple | None, cache_time: int, value: list[str]) -> None:
        if key is None:
            return
        with self._retrieval_cache_lock:
            self._retrieval_cache[key] = (cache_time, list(value))

    def _invalidate_retrieval_cache(self, agent_id: str | None = None) -> None:
        """写入新记忆后清空相关缓存，避免旧证据继续被复用。"""

        with self._retrieval_cache_lock:
            if not agent_id:
                self._retrieval_cache.clear()
                return
            stale_keys = [key for key in self._retrieval_cache if len(key) > 1 and key[1] == agent_id]
            for key in stale_keys:
                self._retrieval_cache.pop(key, None)

    def _retrieval_cache_ttl_ticks(self) -> int:
        return self._config_int("memory_retrieval_ttl_ticks", 0, minimum=0)

    def _extract_world_time(self, observation: Any) -> int:
        if isinstance(observation, dict):
            try:
                return int(observation.get("time", -1))
            except (TypeError, ValueError):
                return -1
        if isinstance(observation, str):
            try:
                parsed = json.loads(observation)
            except json.JSONDecodeError:
                return -1
            if isinstance(parsed, dict):
                try:
                    return int(parsed.get("time", -1))
                except (TypeError, ValueError):
                    return -1
        return -1

    def backup_agent_memory(self, agent_id: str, backup_path: str | None = None) -> str:
        if backup_path is None:
            backup_path = f"./backups/agent_{agent_id}_{time.time()}"
        self.get_agent_collection(agent_id)
        logger.info("智能体 %s 的记忆已标记备份，实际备份需处理持久化文件", agent_id)
        return backup_path
