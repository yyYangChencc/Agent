from __future__ import annotations

import os
import sys
import threading
import types
import unittest


BACKEND_DIR = os.path.dirname(__file__)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


if "chromadb" not in sys.modules:
    chromadb_stub = types.ModuleType("chromadb")

    class _PersistentClient:
        def __init__(self, *args, **kwargs):
            self.collections = {}

        def get_or_create_collection(self, name, metadata=None):
            collection = self.collections.get(name)
            if collection is None:
                collection = _Collection()
                collection.metadata = metadata or {}
                self.collections[name] = collection
            return collection

        def list_collections(self):
            return []

        def reset(self):
            self.collections.clear()

    class _Collection:
        def __init__(self):
            self.ids: list[str] = []
            self.documents: list[str] = []
            self.metadatas: list[dict] = []
            self.metadata = {}

        def count(self):
            return len(self.ids)

        def upsert(self, embeddings=None, documents=None, metadatas=None, ids=None):
            for memory_id, document, metadata in zip(ids or [], documents or [], metadatas or []):
                if memory_id in self.ids:
                    index = self.ids.index(memory_id)
                    self.documents[index] = document
                    self.metadatas[index] = metadata
                else:
                    self.ids.append(memory_id)
                    self.documents.append(document)
                    self.metadatas.append(metadata)

        def get(self, where=None, include=None):
            indexes = range(len(self.ids))
            if where and "agent_id" in where:
                indexes = [
                    index
                    for index in indexes
                    if self.metadatas[index].get("agent_id") == where["agent_id"]
                ]
            return {
                "ids": [self.ids[index] for index in indexes],
                "documents": [self.documents[index] for index in indexes],
                "metadatas": [self.metadatas[index] for index in indexes],
            }

    chromadb_stub.PersistentClient = _PersistentClient
    sys.modules["chromadb"] = chromadb_stub

if "chromadb.config" not in sys.modules:
    chromadb_config_stub = types.ModuleType("chromadb.config")

    class Settings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    chromadb_config_stub.Settings = Settings
    sys.modules["chromadb.config"] = chromadb_config_stub


from persona.agent_memory.mem import MultiAgentMemoryManager
from persona.agents.prompt import BasePromptBuilder


class _LLM:
    def __init__(self):
        self.embedding_calls = 0
        self.generate_calls = 0
        self.embedded_texts: list[str] = []

    def get_embeddings(self, text: str) -> list[float]:
        self.embedding_calls += 1
        self.embedded_texts.append(text)
        return [float(len(text)), 1.0]

    def generate(self, system: str, user: str) -> str:
        self.generate_calls += 1
        raise AssertionError("smart_retrieve must not call generate")


class _QueryCollection:
    def __init__(self, results: dict):
        self.results = results
        self.query_calls: list[dict] = []

    def count(self):
        return len(self.results.get("documents", [[]])[0])

    def query(self, **kwargs):
        self.query_calls.append(kwargs)
        return self.results


class _ListCollection:
    def __init__(self):
        self.result = {
            "ids": ["old", "new", "missing_time"],
            "documents": ["old memory", "new memory", "memory without saved_at"],
            "metadatas": [
                {"agent_id": "agent_1", "saved_at": 1, "memory_type": "episodic"},
                {"agent_id": "agent_1", "saved_at": 5, "memory_type": "semantic"},
                {"agent_id": "agent_1", "memory_type": "reflective"},
            ],
        }
        self.get_calls: list[dict] = []

    def count(self):
        return len(self.result["ids"])

    def get(self, **kwargs):
        self.get_calls.append(kwargs)
        return self.result


def _manager(llm: _LLM | None = None) -> MultiAgentMemoryManager:
    manager = MultiAgentMemoryManager.__new__(MultiAgentMemoryManager)
    manager.llm_client = llm or _LLM()
    manager.agent_collections = {}
    manager._embedding_cache = {}
    manager._embedding_cache_lock = threading.Lock()
    return manager


class MemorySystemTest(unittest.TestCase):
    def test_embedding_cache_reuses_identical_text(self):
        llm = _LLM()
        manager = _manager(llm)

        self.assertEqual(manager._get_embedding("same text"), [9.0, 1.0])
        self.assertEqual(manager._get_embedding("same text"), [9.0, 1.0])

        self.assertEqual(llm.embedding_calls, 1)

    def test_normalize_metadata_adds_structured_fields_and_sanitizes_values(self):
        manager = _manager()

        metadata = manager._normalize_metadata(
            "agent_1",
            12,
            {
                "type": "trajectory",
                "importance": 0.8,
                "confidence": 0.9,
                "tags": ["food", "route"],
                "optional": None,
            },
        )

        self.assertEqual(metadata["agent_id"], "agent_1")
        self.assertEqual(metadata["saved_at"], 12)
        self.assertEqual(metadata["last_accessed_at"], 12)
        self.assertEqual(metadata["access_count"], 0)
        self.assertEqual(metadata["memory_type"], "episodic")
        self.assertEqual(metadata["importance"], 0.8)
        self.assertEqual(metadata["confidence"], 0.9)
        self.assertEqual(metadata["tags"], "food,route")
        self.assertNotIn("optional", metadata)

    def test_where_keeps_chroma_filter_simple(self):
        manager = _manager()

        self.assertEqual(manager._build_where("agent_1", None), {"agent_id": "agent_1"})
        self.assertEqual(
            manager._build_where("agent_1", {"task": "map_navigation"}),
            {"$and": [{"agent_id": "agent_1"}, {"task": "map_navigation"}]},
        )

    def test_ranked_results_prioritize_task_need_and_filter_context(self):
        manager = _manager()
        results = {
            "documents": [[
                "near but weak memory",
                "task matched food route",
                "social-only post memory",
            ]],
            "metadatas": [[
                {
                    "memory_type": "episodic",
                    "saved_at": 1,
                    "importance": 0.1,
                    "confidence": 0.1,
                },
                {
                    "memory_type": "episodic",
                    "saved_at": 90,
                    "task": "find_food",
                    "need_key": "satiety",
                    "importance": 1.0,
                    "confidence": 1.0,
                },
                {
                    "memory_type": "social",
                    "saved_at": 99,
                    "importance": 1.0,
                    "confidence": 1.0,
                },
            ]],
            "distances": [[0.01, 0.4, 0.0]],
        }

        ranked = manager._format_ranked_results(
            results,
            n_results=2,
            query="food route",
            task="find_food",
            urgency={"satiety": 0.8},
            allowed_memory_types={"episodic"},
        )

        self.assertIn("task matched food route", ranked[0])
        self.assertEqual(len(ranked), 2)
        self.assertNotIn("social-only post memory", "\n".join(ranked))

    def test_smart_retrieve_uses_deterministic_query_without_llm_keyword_call(self):
        llm = _LLM()
        manager = _manager(llm)
        collection = _QueryCollection({
            "documents": [["map memory"]],
            "metadatas": [[{"memory_type": "semantic", "saved_at": 3, "importance": 0.9}]],
            "distances": [[0.2]],
        })
        manager.get_agent_collection = lambda agent_id: collection

        memories = manager.smart_retrieve(
            "agent_1",
            "observe food shop",
            "find_food",
            {"satiety": 0.9},
            {"satiety": 30.0},
            n_results=1,
            context="world",
        )

        self.assertEqual(llm.generate_calls, 0)
        self.assertIn("当前任务: find_food", llm.embedded_texts[-1])
        self.assertEqual(collection.query_calls[0]["where"], {"agent_id": "agent_1"})
        self.assertIn("map memory", memories[0])

    def test_list_agent_memories_returns_structured_rows_sorted_by_saved_at(self):
        manager = _manager()
        collection = _ListCollection()
        manager.get_agent_collection = lambda agent_id: collection

        rows = manager.list_agent_memories("agent_1")

        self.assertEqual(collection.get_calls[0]["where"], {"agent_id": "agent_1"})
        self.assertEqual(collection.get_calls[0]["include"], ["documents", "metadatas"])
        self.assertEqual([row["id"] for row in rows], ["new", "old", "missing_time"])
        self.assertEqual(rows[0]["document"], "new memory")
        self.assertEqual(rows[0]["metadata"]["saved_at"], 5)

    def test_memory_prompt_adds_action_hints(self):
        block = BasePromptBuilder()._memory_block([
            "[semantic t=1 object=default_map] 食物店 shop_1 位于 (2,18)",
            "[reflective t=2 task=find_food] 不要重复询问已知位置",
        ])

        self.assertIn("可用记忆", block)
        self.assertIn("[地图/规则]", block)
        self.assertIn("[反思]", block)


if __name__ == "__main__":
    unittest.main()
