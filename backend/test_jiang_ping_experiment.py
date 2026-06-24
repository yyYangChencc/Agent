from __future__ import annotations

import os
import sys
import types
import unittest


BACKEND_DIR = os.path.dirname(__file__)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


if "chromadb" not in sys.modules:
    chromadb_stub = types.ModuleType("chromadb")

    class _PersistentClient:
        def __init__(self, *args, **kwargs):
            pass

    chromadb_stub.PersistentClient = _PersistentClient
    sys.modules["chromadb"] = chromadb_stub

if "chromadb.config" not in sys.modules:
    chromadb_config_stub = types.ModuleType("chromadb.config")

    class Settings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    chromadb_config_stub.Settings = Settings
    sys.modules["chromadb.config"] = chromadb_config_stub


from persona.agents.agent import Agent
from persona.config import AgentConfig
from persona.news_events import NEWS_SCHEDULE, OPINION_SCALE
from persona.opinion.scale import JIANG_PING_TOPIC, OPINION_MAX, OPINION_MIN, clamp_opinion


class _World:
    time = 0

    def __init__(self):
        self.agents = {}

    def add_agent(self, agent):
        self.agents[agent.id] = agent


class _Stub:
    def smart_retrieve(self, *args, **kwargs):
        return []


class JiangPingExperimentTest(unittest.TestCase):
    def test_default_experiment_step_limit_is_100(self):
        self.assertEqual(AgentConfig().simulation_step_limit, 100)

    def test_news_schedule_is_jiang_ping_timeline_within_100_ticks(self):
        self.assertEqual(min(NEWS_SCHEDULE), 5)
        self.assertEqual(max(NEWS_SCHEDULE), 95)
        self.assertGreaterEqual(len(NEWS_SCHEDULE), 8)

        for tick, news in NEWS_SCHEDULE.items():
            self.assertGreaterEqual(tick, 1)
            self.assertLessEqual(tick, 100)
            self.assertEqual(news["topic"], JIANG_PING_TOPIC)
            self.assertIn("姜萍", news["title"] + news["content"])
            self.assertGreaterEqual(news["opinion_index"], OPINION_MIN)
            self.assertLessEqual(news["opinion_index"], OPINION_MAX)

    def test_opinion_scale_covers_negative_neutral_positive_ranges(self):
        ranges = [item["range"] for item in OPINION_SCALE]
        self.assertEqual(ranges[0][0], OPINION_MIN)
        self.assertEqual(ranges[-1][1], OPINION_MAX)
        self.assertTrue(any(item["range"][0] <= 0.0 <= item["range"][1] for item in OPINION_SCALE))

    def test_agent_opinion_clamps_to_negative_positive_scale(self):
        config = AgentConfig(initial_opinion=0.0)
        agent = Agent(
            agent_id="agent_1",
            position=[0, 0],
            world=_World(),
            policy=_Stub(),
            mem=_Stub(),
            reflect=_Stub(),
            platform=None,
            social_policy=_Stub(),
            config=config,
        )

        agent.update_opinion(-5.0)
        self.assertEqual(agent.opinion, OPINION_MIN)

        agent.update_opinion(5.0)
        self.assertEqual(agent.opinion, OPINION_MAX)

        self.assertEqual(clamp_opinion(-1.2345), OPINION_MIN)
        self.assertEqual(clamp_opinion(1.2345), OPINION_MAX)


if __name__ == "__main__":
    unittest.main()
