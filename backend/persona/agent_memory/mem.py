from __future__ import annotations
import chromadb
import hashlib
import threading
import time
from typing import Any
from chromadb.config import Settings
from persona.logger import get_logger

logger = get_logger(__name__)

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


class MultiAgentMemoryManager:
    def __init__(self, llm_client, persist_directory: str = "./chroma_agents"):
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(allow_reset=True),
        )
        self.llm_client = llm_client
        self.agent_collections: dict = {}
        self._collection_lock = threading.Lock()
        self._embedding_cache: dict[str, list[float]] = {}
        self._embedding_cache_lock = threading.Lock()

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

    def store_agent_memory(self, agent_id: str, memory_text: str, world_time: int = 0, **metadata) -> str:
        collection = self.get_agent_collection(agent_id)
        full_metadata = self._normalize_metadata(agent_id, world_time, metadata)
        memory_id = f"{agent_id}_{hashlib.md5(memory_text.encode()).hexdigest()[:10]}"
        embedding = self._get_embedding(memory_text)
        if not embedding:
            logger.warning("[%s] 获取 embedding 失败，跳过记忆存储: %r", agent_id, memory_text[:60])
            return memory_id
        collection.upsert(
            embeddings=[embedding],
            documents=[memory_text],
            metadatas=[full_metadata],
            ids=[memory_id],
        )
        return memory_id

    async def astore_agent_memory(self, agent_id: str, memory_text: str, world_time: int = 0, **metadata) -> str:
        collection = self.get_agent_collection(agent_id)
        full_metadata = self._normalize_metadata(agent_id, world_time, metadata)
        memory_id = f"{agent_id}_{hashlib.md5(memory_text.encode()).hexdigest()[:10]}"
        embedding = await self._aget_embedding(memory_text)
        if not embedding:
            logger.warning("[%s] 获取 embedding 失败，跳过记忆存储: %r", agent_id, memory_text[:60])
            return memory_id
        collection.upsert(
            embeddings=[embedding],
            documents=[memory_text],
            metadatas=[full_metadata],
            ids=[memory_id],
        )
        return memory_id

    def smart_retrieve(
        self,
        agent_id: str,
        observation: str,
        task: str,
        urgency: dict,
        satisfaction_threshold: dict,
        n_results: int = 3,
        context: str = "world",
    ) -> list[str]:
        """Build a deterministic retrieval query, then retrieve task-relevant memories."""
        query = self._build_retrieval_query(observation, task, urgency, satisfaction_threshold)
        result = self.retrieve_agent_memories(
            agent_id,
            query,
            n_results=n_results,
            context=context,
            task=task,
            urgency=urgency,
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
        **kwargs,
    ) -> list[str]:
        started_at = time.perf_counter()
        collection = self.get_agent_collection(agent_id)

        count = collection.count()
        if count == 0:
            return []
        fetch_n = min(max(n_results * 4, n_results), count)

        allowed_memory_types = self._memory_types_for_context(context)
        kwargs["where"] = self._build_where(agent_id, kwargs.get("where"))

        query_embedding = self._get_embedding(query)
        if not query_embedding:
            logger.warning("[%s] 获取查询 embedding 失败，返回空记忆", agent_id)
            return []

        try:
            results = collection.query(
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
            return []
        out = self._format_ranked_results(results, n_results, query, task, urgency or {}, allowed_memory_types)
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
    ) -> list[str]:
        query = self._build_retrieval_query(observation, task, urgency, satisfaction_threshold)
        return await self.aretrieve_agent_memories(
            agent_id,
            query,
            n_results=n_results,
            context=context,
            task=task,
            urgency=urgency,
        )

    def _build_retrieval_query(self, observation, task, urgency, satisfaction_threshold) -> str:
        parts = [observation]
        if task and task != "none":
            parts.append(f"当前任务: {task}")
        urgent = [(k, v) for k, v in urgency.items()
                  if satisfaction_threshold.get(k) and urgency.get(k, 0) > 0.5]
        if urgent:
            urgent_str = ", ".join(f"{k}急切度{v:.2f}" for k, v in urgent[:2])
            parts.append(f"紧迫需求: {urgent_str}")
        return " | ".join(parts)

    async def aretrieve_agent_memories(
        self,
        agent_id: str,
        query: str,
        n_results: int = 5,
        *,
        context: str = "world",
        task: str = "",
        urgency: dict | None = None,
        **kwargs,
    ) -> list[str]:
        started_at = time.perf_counter()
        collection = self.get_agent_collection(agent_id)

        count = collection.count()
        if count == 0:
            return []
        fetch_n = min(max(n_results * 4, n_results), count)

        allowed_memory_types = self._memory_types_for_context(context)
        kwargs["where"] = self._build_where(agent_id, kwargs.get("where"))

        query_embedding = await self._aget_embedding(query)
        if not query_embedding:
            logger.warning("[%s] 获取查询 embedding 失败，返回空记忆", agent_id)
            return []

        try:
            results = collection.query(
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
            return []
        out = self._format_ranked_results(results, n_results, query, task, urgency or {}, allowed_memory_types)
        logger.debug(
            "[%s] 异步记忆检索完成 context=%s returned=%d elapsed=%.3fs",
            agent_id,
            context,
            len(out),
            time.perf_counter() - started_at,
        )
        return out

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

    def _format_ranked_results(
        self,
        results: dict,
        n_results: int,
        query: str,
        task: str,
        urgency: dict,
        allowed_memory_types: set[str] | None = None,
    ) -> list[str]:
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        rows = []
        for index, (doc, meta) in enumerate(zip(docs, metas)):
            if allowed_memory_types and meta.get("memory_type", DEFAULT_MEMORY_TYPE) not in allowed_memory_types:
                continue
            distance = distances[index] if index < len(distances) else 1.0
            rows.append((self._memory_score(doc, meta, distance, query, task, urgency), doc, meta))
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
    ) -> float:
        similarity = 1.0 / (1.0 + max(0.0, float(distance)))
        importance = float(meta.get("importance", 0.5) or 0.5)
        confidence = float(meta.get("confidence", 0.5) or 0.5)
        task_match = 1.0 if task and str(meta.get("task", "")) == task else 0.0
        need_key = str(meta.get("need_key", ""))
        need_match = float(urgency.get(need_key, 0.0)) if need_key else 0.0
        saved_at = float(meta.get("saved_at", 0) or 0)
        recency = min(1.0, saved_at / 100.0) if saved_at > 0 else 0.0
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

    def reset_all(self) -> None:
        """清空所有智能体的记忆（ChromaDB reset）。"""
        self.client.reset()
        self.agent_collections.clear()
        with self._embedding_cache_lock:
            self._embedding_cache.clear()
        logger.info("已清空所有智能体记忆存储")

    def backup_agent_memory(self, agent_id: str, backup_path: str | None = None) -> str:
        if backup_path is None:
            backup_path = f"./backups/agent_{agent_id}_{time.time()}"
        self.get_agent_collection(agent_id)
        logger.info("智能体 %s 的记忆已标记备份，实际备份需处理持久化文件", agent_id)
        return backup_path
