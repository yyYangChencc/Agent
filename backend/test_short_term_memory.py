from __future__ import annotations

import unittest

from persona.agents.agent import Agent
from persona.agents.policy import ActionParser, LLMPolicy
from persona.agents.prompt import ReflectPromptBuilder, WorldPromptBuilder
from persona.agents.short_term_memory import ShortTermMemoryBuffer
from persona.config import AgentConfig
from persona.llm.mock_client import MockLLMClient
from persona.reflect import Reflect
from world.world import World


def _summary(start_tick: int, end_tick: int) -> dict:
    return {
        "start_tick": start_tick,
        "end_tick": end_tick,
        "chronology": "压缩后的经历",
        "task_progress": "继续当前任务",
        "successful_actions": [],
        "failed_actions": [],
        "unresolved_goals": [],
        "referenced_entity_ids": [],
    }


class _World:
    """为短期记忆测试提供最小世界接口。"""

    def __init__(self) -> None:
        self.time = 0
        self.agents = {}
        self.tools_prompt = "（无工具）"

    def add_agent(self, agent) -> None:
        self.agents[agent.id] = agent


class _FailingReflect:
    def summarize_short_term_memory(self, *args, **kwargs):
        raise ValueError("invalid summary")

    async def asummarize_short_term_memory(self, *args, **kwargs):
        raise ValueError("invalid summary")


class _RecordingMockLLMClient(MockLLMClient):
    """记录真实世界决策链生成的行动 Prompt。"""

    def __init__(self, config) -> None:
        super().__init__(config)
        self.world_prompts: list[str] = []

    def generate(self, system: str, user: str, *, response_format: dict | None = None) -> str:
        if "自主智能体" in str(system or "") and "## 短期记忆" in str(user or ""):
            self.world_prompts.append(str(user))
        return super().generate(system, user, response_format=response_format)


class _MemorySpy:
    """统计真实 tick 链路触发的长期记忆写入。"""

    def __init__(self) -> None:
        self.observation_calls = 0
        self.action_result_calls = 0
        self.agent_memory_calls = 0

    def store_observation(self, *args, **kwargs) -> None:
        self.observation_calls += 1

    def store_action_result(self, *args, **kwargs) -> None:
        self.action_result_calls += 1

    def store_agent_memory(self, *args, **kwargs) -> str:
        self.agent_memory_calls += 1
        return "memory-spy-id"


class _OpinionAssessmentOrderSpy:
    """记录观念评测执行时已经存在的短期总结数量。"""

    def __init__(self) -> None:
        self.summary_counts: list[tuple[int, int]] = []

    async def aassess_all(self, agents, tick: int) -> None:
        for agent in agents:
            self.summary_counts.append((tick, len(agent.short_term_memory.summaries())))


class ShortTermMemoryBufferTests(unittest.TestCase):
    def test_compaction_keeps_complete_hot_tick(self) -> None:
        buffer = ShortTermMemoryBuffer()
        for tick in range(1, 5):
            buffer.append(world_time=tick, record_type="observation", content={"time": tick})
            buffer.append(world_time=tick, record_type="action_result", content={"tool": "move"})

        snapshot = buffer.compaction_snapshot(max_ticks=4, hot_ticks=1)
        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual((snapshot.start_tick, snapshot.end_tick), (1, 3))
        self.assertEqual({entry.world_time for entry in snapshot.entries}, {1, 2, 3})

        self.assertTrue(buffer.commit_summary(snapshot, _summary(1, 3)))
        self.assertEqual(buffer.raw_ticks(), [4])
        self.assertEqual(len(buffer.summaries()), 1)

    def test_repeated_compaction_merges_previous_summary(self) -> None:
        buffer = ShortTermMemoryBuffer()
        for tick in range(1, 5):
            buffer.append(world_time=tick, record_type="action_result", content={"tick": tick})
        first = buffer.compaction_snapshot(max_ticks=4, hot_ticks=1)
        assert first is not None
        self.assertTrue(buffer.commit_summary(first, _summary(1, 3)))

        for tick in range(5, 8):
            buffer.append(world_time=tick, record_type="action_result", content={"tick": tick})
        second = buffer.compaction_snapshot(max_ticks=4, hot_ticks=1)
        assert second is not None
        self.assertEqual((second.start_tick, second.end_tick), (1, 6))
        self.assertTrue(any(entry.record_type == "short_term_summary" for entry in second.entries))
        self.assertTrue(buffer.commit_summary(second, _summary(1, 6)))
        self.assertEqual(buffer.raw_ticks(), [7])

    def test_stale_snapshot_never_deletes_entries(self) -> None:
        buffer = ShortTermMemoryBuffer()
        for tick in range(1, 5):
            buffer.append(world_time=tick, record_type="action", content=str(tick))
        snapshot = buffer.compaction_snapshot(max_ticks=4, hot_ticks=1)
        assert snapshot is not None
        buffer.append(world_time=5, record_type="observation", content={"time": 5})
        before = buffer.entries
        self.assertFalse(buffer.commit_summary(snapshot, _summary(1, 3)))
        self.assertEqual(buffer.entries, before)


class ShortTermMemoryAgentTests(unittest.TestCase):
    def _agent(self, *, max_ticks: int = 4, hot_ticks: int = 1) -> Agent:
        config = AgentConfig(
            short_term_memory_max_ticks=max_ticks,
            short_term_memory_hot_ticks=hot_ticks,
            short_term_memory_summary_max_chars=1200,
            short_term_memory_summary_chunk_max_chars=5000,
        )
        llm = MockLLMClient(config)
        reflect = Reflect(llm, ReflectPromptBuilder(), config)
        return Agent(
            agent_id="agent_1",
            position=[1, 1],
            world=_World(),
            policy=None,
            mem=object(),
            reflect=reflect,
            platform=None,
            social_policy=None,
            memory_planner=None,
            config=config,
        )

    def _fill_ticks(self, agent: Agent, end_tick: int) -> None:
        for tick in range(1, end_tick + 1):
            agent.world.time = tick
            agent._current_episode_id = f"episode_{tick}"
            agent.add_history("observation", {
                "time": tick,
                "position": [tick, tick],
                "people": [],
                "objects": [],
                "actions": [],
                "social": {"notification_events": []},
            })
            agent.add_history("action_result", {
                "tool": "move",
                "feedback": f"moved at {tick}",
                "execution_status": "tool_returned",
            })

    def test_agent_compacts_with_mock_llm_and_prompt_uses_summary(self) -> None:
        agent = self._agent()
        self._fill_ticks(agent, 4)

        self.assertTrue(agent.compact_short_term_memory_if_needed())
        self.assertEqual(agent.short_term_memory.raw_ticks(), [4])
        summary = agent.short_term_memory.summaries()[0]
        self.assertEqual(summary.metadata["start_tick"], 1)
        self.assertEqual(summary.metadata["end_tick"], 3)

        _, user = WorldPromptBuilder().build(agent, "{}", [])
        self.assertIn("## 短期记忆", user)
        self.assertIn("较早经历的滚动总结", user)
        self.assertIn("t=4", user)
        self.assertNotIn("t=1\n- observation", user)
        self.assertNotIn("t=4\n- observation", user)
        self.assertIn("t=4\n- action_result", user)

    def test_prompt_excludes_only_current_observation(self) -> None:
        agent = self._agent(max_ticks=10, hot_ticks=2)
        self._fill_ticks(agent, 4)

        _, user = WorldPromptBuilder().build(agent, "current observation", [])

        self.assertIn("t=3\n- observation", user)
        self.assertNotIn("t=4\n- observation", user)
        self.assertIn("t=4\n- action_result", user)

    def test_failed_summary_preserves_all_raw_entries(self) -> None:
        agent = self._agent()
        self._fill_ticks(agent, 4)
        before = agent.short_term_memory.entries
        agent.reflect = _FailingReflect()

        self.assertFalse(agent.compact_short_term_memory_if_needed())
        self.assertEqual(agent.short_term_memory.entries, before)
        self.assertEqual(agent.last_short_term_compaction["status"], "failed")

    def test_task_replacement_clears_previous_trajectory(self) -> None:
        agent = self._agent()
        agent.set_task("寻找食物", "satiety")
        agent.world.time = 1
        agent.append_trajectory(
            {"time": 1},
            '{"action":{"tool":"move","args":{"x":2,"y":2}}}',
            0.0,
            action_result={"tool": "move", "feedback": "moved", "execution_status": "tool_returned"},
        )
        self.assertEqual(agent.task_working_memory()["step_count"], 1)

        agent.set_task("赚钱打工", "money")
        self.assertEqual(agent.trajectory_buffer, [])
        self.assertEqual(agent.task_working_memory(), {})


class ShortTermMemoryAsyncTests(unittest.IsolatedAsyncioTestCase):
    def _agent(self) -> Agent:
        config = AgentConfig(
            short_term_memory_max_ticks=4,
            short_term_memory_hot_ticks=1,
            short_term_memory_summary_max_chars=1200,
            short_term_memory_summary_chunk_max_chars=5000,
        )
        llm = MockLLMClient(config)
        reflect = Reflect(llm, ReflectPromptBuilder(), config)
        return Agent(
            agent_id="agent_async",
            position=[1, 1],
            world=_World(),
            policy=None,
            mem=object(),
            reflect=reflect,
            platform=None,
            social_policy=None,
            memory_planner=None,
            config=config,
        )

    def _fill_ticks(self, agent: Agent) -> None:
        for tick in range(1, 5):
            agent.world.time = tick
            agent._current_episode_id = f"episode_{tick}"
            agent.add_history("action_result", {
                "tool": "move",
                "feedback": f"moved at {tick}",
                "execution_status": "tool_returned",
            })

    async def test_agent_async_compaction(self) -> None:
        agent = self._agent()
        self._fill_ticks(agent)

        self.assertTrue(await agent.acompact_short_term_memory_if_needed())
        self.assertEqual(agent.short_term_memory.raw_ticks(), [4])
        self.assertEqual(len(agent.short_term_memory.summaries()), 1)

    async def test_failed_async_summary_preserves_all_entries(self) -> None:
        agent = self._agent()
        self._fill_ticks(agent)
        before = agent.short_term_memory.entries
        agent.reflect = _FailingReflect()

        self.assertFalse(await agent.acompact_short_term_memory_if_needed())
        self.assertEqual(agent.short_term_memory.entries, before)
        self.assertEqual(agent.last_short_term_compaction["status"], "failed")

    async def test_world_compacts_multiple_agents(self) -> None:
        first = self._agent()
        second = self._agent()
        second.id = "agent_async_2"
        self._fill_ticks(first)
        self._fill_ticks(second)
        world = World.__new__(World)
        world.time = 4

        await world._compact_short_term_memories([first, second])

        self.assertEqual(first.short_term_memory.raw_ticks(), [4])
        self.assertEqual(second.short_term_memory.raw_ticks(), [4])
        self.assertEqual(len(first.short_term_memory.summaries()), 1)
        self.assertEqual(len(second.short_term_memory.summaries()), 1)

    async def test_mock_world_compacts_after_assessment_and_reuses_summary(self) -> None:
        config = AgentConfig(
            short_term_memory_max_ticks=3,
            short_term_memory_hot_ticks=1,
            short_term_memory_summary_max_chars=1200,
            short_term_memory_summary_chunk_max_chars=5000,
            memory_forced_recall_enabled=False,
            memory_forgetting_enabled=False,
            psychological_assessment_enabled=False,
            micro_reflect_interval=100,
        )
        llm = _RecordingMockLLMClient(config)
        memory = _MemorySpy()
        assessment = _OpinionAssessmentOrderSpy()
        world = World(opinion_assessor=assessment)
        reflect = Reflect(llm, ReflectPromptBuilder(), config)
        policy = LLMPolicy(llm, WorldPromptBuilder(), ActionParser())
        agent = Agent(
            agent_id="agent_world",
            position=[1, 1],
            world=world,
            policy=policy,
            mem=memory,
            reflect=reflect,
            platform=None,
            social_policy=None,
            memory_planner=None,
            config=config,
        )

        for _ in range(4):
            await world.astep()

        self.assertEqual(assessment.summary_counts, [(1, 0), (2, 0), (3, 0), (4, 1)])
        self.assertEqual(agent.short_term_memory.raw_ticks(), [3, 4])
        self.assertEqual(len(agent.short_term_memory.summaries()), 1)
        self.assertIn("已压缩 t=1 至 t=2 的短期经历", llm.world_prompts[3])
        self.assertEqual(len(agent.trajectory_buffer), 4)
        self.assertEqual(agent.last_action["execution_status"], "no_action")
        self.assertEqual(
            agent.trajectory_buffer[-1]["action_result"]["execution_status"],
            "no_action",
        )
        self.assertEqual(memory.observation_calls, 4)
        self.assertEqual(memory.action_result_calls, 4)
        self.assertEqual(memory.agent_memory_calls, 0)


if __name__ == "__main__":
    unittest.main()
