from __future__ import annotations

import asyncio
import threading
import unittest
from types import SimpleNamespace

from persona.llm.debug_trace import (
    LLMTraceStore,
    TracingLLMClient,
    annotate_current_llm_trace,
    trace_llm_call,
)
from persona.llm.interface import JSON_OBJECT_RESPONSE_FORMAT, LLMClient
from world.serializer import snapshot


class _ScriptedLLMClient(LLMClient):
    """按预设结果返回内容，便于验证成功、重试和异常记录。"""

    def __init__(self, results: list[str | Exception]) -> None:
        self.results = list(results)
        self._lock = threading.Lock()

    def generate(self, system: str, user: str, *, response_format: dict | None = None) -> str:
        with self._lock:
            result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def get_embeddings(self, text: str) -> list[float]:
        return [0.25, 0.75]


class _AsyncEchoLLMClient(LLMClient):
    """通过不同延迟制造可重复的并发完成顺序。"""

    def generate(self, system: str, user: str, *, response_format: dict | None = None) -> str:
        return user

    async def agenerate(self, system: str, user: str, *, response_format: dict | None = None) -> str:
        await asyncio.sleep(0.02 if user == "slow" else 0)
        return user

    def get_embeddings(self, text: str) -> list[float]:
        return [1.0]


def _record_one_call(store: LLMTraceStore, *, agent_id: str, tick: int) -> None:
    client = TracingLLMClient(_ScriptedLLMClient([f"response-{tick}"]), store)
    with trace_llm_call(client, agent_id=agent_id, tick=tick, stage="world_decision"):
        client.generate(f"system-{tick}", f"user-{tick}")


def _fake_agent(agent_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=agent_id,
        position=[1, 2],
        emotion="neutral",
        task="none",
        current_focus="",
        salary=0.0,
        satisfaction={"satiety": 50.0},
        urgency={"satiety": 0.5},
        satisfaction_threshold={"satiety": 30.0},
        need_gap={"satiety": 0.0},
        pressure_memory={"satiety": 0.0},
        load_saturation={"satiety": 0.0},
        effective_pressure={"satiety": 0.0},
        last_psychological_assessment=None,
        opinion=0.0,
        opinion_scores={},
        last_opinion_assessment=None,
        last_opinion_voting=None,
        short_term_memory=None,
        sleeping=False,
        sleep_ticks_remaining=0,
        personal_bed_id=None,
        personal_bed_position=None,
        personal_bed_entrance=None,
        inside_building_id=None,
        config=SimpleNamespace(simulation_step_limit=10),
    )


class LLMTraceStoreTests(unittest.TestCase):
    def test_sync_call_records_complete_payload_and_skips_embeddings(self) -> None:
        store = LLMTraceStore(retention_ticks=3)
        client = TracingLLMClient(_ScriptedLLMClient(['{"ok": true}']), store)

        with trace_llm_call(
            client,
            agent_id="agent_1",
            tick=7,
            stage="world_decision",
            metadata={"context": "world"},
        ):
            response = client.generate("完整 system", "完整 user", response_format=JSON_OBJECT_RESPONSE_FORMAT)
        self.assertEqual(client.get_embeddings("不应追踪"), [0.25, 0.75])

        self.assertEqual(response, '{"ok": true}')
        records = store.records_for("agent_1", 7)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["system_prompt"], "完整 system")
        self.assertEqual(records[0]["user_prompt"], "完整 user")
        self.assertEqual(records[0]["response"], '{"ok": true}')
        self.assertEqual(records[0]["response_format"], JSON_OBJECT_RESPONSE_FORMAT)
        self.assertEqual(records[0]["metadata"], {"context": "world"})
        self.assertEqual(records[0]["status"], "completed")

    def test_retries_share_call_id_and_keep_each_validation_error(self) -> None:
        store = LLMTraceStore(retention_ticks=3)
        client = TracingLLMClient(_ScriptedLLMClient(["bad-json", '{"ok": true}']), store)

        with trace_llm_call(client, agent_id="agent_1", tick=3, stage="memory_planning"):
            client.generate("system", "first")
            annotate_current_llm_trace("输出不是合法 JSON")
            client.generate("system", "retry")

        records = store.records_for("agent_1", 3)
        self.assertEqual([item["attempt"] for item in records], [1, 2])
        self.assertEqual(len({item["call_id"] for item in records}), 1)
        self.assertEqual(records[0]["status"], "invalid_response")
        self.assertEqual(records[0]["validation_error"], "输出不是合法 JSON")
        self.assertEqual(records[1]["status"], "completed")
        self.assertEqual(records[1]["user_prompt"], "retry")

    def test_exception_is_recorded_before_it_is_raised(self) -> None:
        store = LLMTraceStore(retention_ticks=3)
        client = TracingLLMClient(_ScriptedLLMClient([TimeoutError("request timed out")]), store)

        with self.assertRaisesRegex(TimeoutError, "request timed out"):
            with trace_llm_call(client, agent_id="agent_2", tick=4, stage="social_decision"):
                client.generate("system", "user")

        record = store.records_for("agent_2", 4)[0]
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["error_type"], "TimeoutError")
        self.assertEqual(record["error"], "request timed out")
        self.assertIn("TimeoutError", record["validation_error"])

    def test_prune_keeps_configured_tick_window(self) -> None:
        store = LLMTraceStore(retention_ticks=3)
        for tick in range(1, 5):
            _record_one_call(store, agent_id="agent_1", tick=tick)

        store.prune(current_tick=4)

        self.assertEqual(store.records_for("agent_1", 1), [])
        self.assertEqual(store.count_for("agent_1", 2), 1)
        self.assertEqual(store.count_for("agent_1", 3), 1)
        self.assertEqual(store.count_for("agent_1", 4), 1)

    def test_snapshot_exposes_only_previous_tick_metadata(self) -> None:
        store = LLMTraceStore(retention_ticks=3)
        _record_one_call(store, agent_id="agent_1", tick=4)
        _record_one_call(store, agent_id="agent_1", tick=5)
        agent = _fake_agent("agent_1")
        world = SimpleNamespace(
            time=5,
            scenario_name="test",
            agents={agent.id: agent},
            objects={},
            movements=[],
            map=SimpleNamespace(width=8, height=6),
            map_design=None,
            llm_trace_store=store,
        )

        state = snapshot(world)

        self.assertEqual(state["time"], 5)
        self.assertEqual(state["agents"][0]["llm_debug"], {"display_tick": 4, "call_count": 1})
        self.assertNotIn("system_prompt", state["agents"][0]["llm_debug"])


class LLMTraceAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_call_records_payload(self) -> None:
        store = LLMTraceStore(retention_ticks=3)
        client = TracingLLMClient(_AsyncEchoLLMClient(), store)

        with trace_llm_call(client, agent_id="agent_1", tick=8, stage="psychological_assessment"):
            response = await client.agenerate("async-system", "async-user")

        self.assertEqual(response, "async-user")
        record = store.records_for("agent_1", 8)[0]
        self.assertEqual(record["system_prompt"], "async-system")
        self.assertEqual(record["user_prompt"], "async-user")
        self.assertEqual(record["response"], "async-user")

    async def test_concurrent_calls_have_stable_unique_sequence(self) -> None:
        store = LLMTraceStore(retention_ticks=3)
        client = TracingLLMClient(_AsyncEchoLLMClient(), store)

        async def invoke(user: str) -> None:
            with trace_llm_call(client, agent_id="agent_1", tick=9, stage="opinion_voting"):
                await client.agenerate("system", user)

        await asyncio.gather(invoke("slow"), invoke("fast"))

        records = store.records_for("agent_1", 9)
        self.assertEqual([item["sequence"] for item in records], sorted(item["sequence"] for item in records))
        self.assertEqual([item["response"] for item in records], ["fast", "slow"])
        self.assertEqual(len({item["sequence"] for item in records}), 2)
        self.assertEqual(len({item["call_id"] for item in records}), 2)


if __name__ == "__main__":
    unittest.main()
