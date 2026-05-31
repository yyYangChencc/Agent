from __future__ import annotations
import chromadb
import hashlib
import threading
import time
from chromadb.config import Settings
from persona.logger import get_logger

logger = get_logger(__name__)


class MultiAgentMemoryManager:
    def __init__(self, llm_client, persist_directory: str = "./chroma_agents"):
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(allow_reset=True),
        )
        self.llm_client = llm_client
        self.agent_collections: dict = {}
        self._collection_lock = threading.Lock()

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
        full_metadata = {"agent_id": agent_id, "saved_at": world_time, **metadata}
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
        full_metadata = {"agent_id": agent_id, "saved_at": world_time, **metadata}
        memory_id = f"{agent_id}_{hashlib.md5(memory_text.encode()).hexdigest()[:10]}"
        embedding = await self.llm_client.aget_embeddings(memory_text)
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
    ) -> list[str]:
        """Ask the LLM to extract search keywords, then retrieve matching memories."""
        system = (
            "你是记忆检索辅助模块。根据智能体当前状态与观测，"
            "提炼出3-5个最能帮助检索相关记忆的关键词。\n"
            "只输出关键词，用逗号分隔，不加任何解释。"
        )
        user = (
            f"任务：{task}\n"
            f"需求：satiety={urgency.get('satiety', 0):.2f}  relax={urgency.get('relax', 0):.2f}\n"
            f"阈值：satiety<{satisfaction_threshold.get('satiety', 0):.2f}  relax<{satisfaction_threshold.get('relax', 0):.2f}\n"
            f"观测：{observation}"
        )
        try:
            keywords = self.llm_client.generate(system, user).strip()
        except Exception as e:
            logger.error("[%s] 关键词提取失败，回退到原始观测: %s", agent_id, e)
            keywords = observation
        logger.debug("[%s] 记忆检索关键词: %s", agent_id, keywords)
        result = self.retrieve_agent_memories(agent_id, keywords, n_results=n_results)
        logger.debug("[%s] 记忆检索结果: %s", agent_id, result)
        return result

    def retrieve_agent_memories(
        self, agent_id: str, query: str, n_results: int = 5, **kwargs
    ) -> list[str]:
        collection = self.get_agent_collection(agent_id)

        count = collection.count()
        if count == 0:
            return []
        n_results = min(n_results, count)

        if "where" in kwargs:
            kwargs["where"] = {"$and": [{"agent_id": agent_id}, kwargs["where"]]}
        else:
            kwargs["where"] = {"agent_id": agent_id}

        query_embedding = self._get_embedding(query)
        if not query_embedding:
            logger.warning("[%s] 获取查询 embedding 失败，返回空记忆", agent_id)
            return []

        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas"],
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
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        out = []
        for doc, meta in zip(docs, metas):
            tick = meta.get("saved_at")
            prefix = f"[t={tick}] " if tick is not None else ""
            out.append(f"{prefix}{doc}")
        return out

    async def asmart_retrieve(
        self,
        agent_id: str,
        observation: str,
        task: str,
        urgency: dict,
        satisfaction_threshold: dict,
        n_results: int = 3,
    ) -> list[str]:
        query = self._build_retrieval_query(observation, task, urgency, satisfaction_threshold)
        return await self.aretrieve_agent_memories(agent_id, query, n_results=n_results)

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
        self, agent_id: str, query: str, n_results: int = 5, **kwargs
    ) -> list[str]:
        collection = self.get_agent_collection(agent_id)

        count = collection.count()
        if count == 0:
            return []
        n_results = min(n_results, count)

        if "where" in kwargs:
            kwargs["where"] = {"$and": [{"agent_id": agent_id}, kwargs["where"]]}
        else:
            kwargs["where"] = {"agent_id": agent_id}

        query_embedding = await self.llm_client.aget_embeddings(query)
        if not query_embedding:
            logger.warning("[%s] 获取查询 embedding 失败，返回空记忆", agent_id)
            return []

        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas"],
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
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        out = []
        for doc, meta in zip(docs, metas):
            tick = meta.get("saved_at")
            prefix = f"[t={tick}] " if tick is not None else ""
            out.append(f"{prefix}{doc}")
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
        try:
            return self.llm_client.get_embeddings(text)
        except Exception as e:
            logger.error("获取 embedding 失败: %s", e, exc_info=True)
            return []

    def reset_all(self) -> None:
        """清空所有智能体的记忆（ChromaDB reset）。"""
        self.client.reset()
        self.agent_collections.clear()
        logger.info("已清空所有智能体记忆存储")

    def backup_agent_memory(self, agent_id: str, backup_path: str | None = None) -> str:
        if backup_path is None:
            backup_path = f"./backups/agent_{agent_id}_{time.time()}"
        self.get_agent_collection(agent_id)
        logger.info("智能体 %s 的记忆已标记备份，实际备份需处理持久化文件", agent_id)
        return backup_path
