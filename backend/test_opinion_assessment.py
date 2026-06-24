from __future__ import annotations

import os
import sys
import types
import unittest
from unittest.mock import patch


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
from persona.opinion import OpinionAssessmentCoordinator
from persona.opinion.scale import JIANG_PING_TOPIC


class _World:
    time = 0

    def __init__(self):
        self.agents = {}

    def add_agent(self, agent):
        self.agents[agent.id] = agent


class _Stub:
    def smart_retrieve(self, *args, **kwargs):
        return []


class _MemoryStub:
    def __init__(self, memories=None):
        self.memories = memories or []

    def smart_retrieve(self, *args, **kwargs):
        return list(self.memories)


class _LLMStub:
    def __init__(self, response):
        self.response = response
        self.last_system = ""
        self.last_user = ""

    def generate(self, system, user):
        self.last_system = system
        self.last_user = user
        return self.response

    async def agenerate(self, system, user):
        self.last_system = system
        self.last_user = user
        return self.response


class OpinionAssessmentTest(unittest.TestCase):
    def _agent(self, *, config=None, mem=None):
        if config is None:
            config = AgentConfig(
                initial_opinion=0.7,
                opinion_assessment_mode="rule",
                opinion_assessment_history_limit=2,
            )
        if mem is None:
            mem = _MemoryStub()
        agent = Agent(
            agent_id="agent_1",
            position=[0, 0],
            world=_World(),
            policy=_Stub(),
            mem=mem,
            reflect=_Stub(),
            platform=None,
            social_policy=_Stub(),
            config=config,
        )
        return agent, OpinionAssessmentCoordinator(config)

    def test_assesses_only_system_news_topic(self):
        config = AgentConfig(
            initial_opinion=0.7,
            opinion_assessment_mode="rule",
            opinion_assessment_history_limit=2,
        )
        mem = _MemoryStub(["我支持 UBI，因为它能改善基本保障。"])
        agent, assessor = self._agent(config=config, mem=mem)
        agent.current_focus = "UBI policy"
        agent.add_history("event", "今天看到 UBI 能带来公平和安全。")

        result = assessor.assess_agent(agent, 1)

        self.assertEqual(result["topic"], JIANG_PING_TOPIC)
        self.assertEqual(result["score"], 0.7)
        self.assertEqual(result["source"], "rule_context_assessment")
        self.assertIn("context", result)
        self.assertTrue(result["evidence"])
        self.assertEqual(agent.last_opinion_assessment, result)
        self.assertEqual(agent.opinion_scores[JIANG_PING_TOPIC], result["score"])
        self.assertEqual(agent.opinion, 0.7)

    def test_assessment_score_updates_agent_opinion(self):
        config = AgentConfig(initial_opinion=0.7, opinion_assessment_mode="rule")
        mem = _MemoryStub(["我支持姜萍，因为这是一个励志的正面叙事。"])
        agent, assessor = self._agent(config=config, mem=mem)
        agent.current_focus = "resource policy"
        agent.add_history("event", "今天看到姜萍事件中的鼓励和支持声音。")

        result = assessor.assess_agent(agent, 1)

        self.assertEqual(result["topic"], JIANG_PING_TOPIC)
        self.assertGreater(result["score"], 0.7)
        self.assertEqual(agent.opinion, result["score"])
        self.assertEqual(agent.opinion_scores[JIANG_PING_TOPIC], result["score"])

    def test_agent_opinion_is_authoritative_anchor(self):
        agent, assessor = self._agent()
        agent.current_focus = "resource policy"
        agent.opinion_scores[JIANG_PING_TOPIC] = 0.2

        result = assessor.assess_agent(agent, 1)

        self.assertEqual(result["topic"], JIANG_PING_TOPIC)
        self.assertEqual(agent.opinion, 0.7)
        self.assertEqual(result["score"], agent.opinion)

    def test_llm_mode_uses_json_assessment(self):
        config = AgentConfig(
            initial_opinion=0.7,
            opinion_assessment_mode="llm",
        )
        agent, _ = self._agent(config=config)
        agent.current_focus = "resource policy"
        assessor = OpinionAssessmentCoordinator(
            config,
            llm=_LLMStub('{"score": 0.33, "confidence": 0.9, "reason": "测试理由", "evidence": ["证据A"]}'),
        )

        result = assessor.assess_agent(agent, 1)

        self.assertEqual(result["source"], "llm_context_assessment")
        self.assertEqual(result["score"], 0.33)
        self.assertEqual(result["confidence"], 0.9)
        self.assertEqual(result["reason"], "测试理由")
        self.assertEqual(result["topic"], JIANG_PING_TOPIC)
        self.assertEqual(agent.opinion, 0.33)

    def test_llm_context_contains_full_dynamic_role_card_once(self):
        config = AgentConfig(
            initial_opinion=0.7,
            opinion_assessment_mode="llm",
        )
        agent, _ = self._agent(config=config)
        agent.last_psychological_assessment = {
            "tick": 10,
            "status": "llm_single_evaluator",
            "activated_needs": ["satiety"],
            "role_card_delta": {
                "summary": "饥饿压力上升",
                "emotion_tone": "急迫",
                "cognition": ["优先关注食物"],
                "behavior": ["寻找补给"],
                "social_expression": ["表达更直接"],
                "online_behavior": ["询问资源"],
                "decision_bias": ["提高补给优先级"],
                "constraints": ["不得编造地点"],
                "mediator_focus": [{"key": "resource_seeking", "value": 0.8}],
            },
        }
        llm = _LLMStub('{"score": 0.33, "confidence": 0.9, "reason": "测试理由", "evidence": ["证据A"]}')
        assessor = OpinionAssessmentCoordinator(config, llm=llm)

        result = assessor.assess_agent(agent, 1)
        user_payload = __import__("json").loads(llm.last_user)
        context = user_payload["context"]

        self.assertIn("dynamic_role_card", context)
        self.assertNotIn("psychological_context", context)
        self.assertNotIn("psychological_context", result)
        self.assertEqual(
            context["dynamic_role_card"]["role_card_delta"]["cognition"],
            ["优先关注食物"],
        )
        self.assertEqual(
            context["dynamic_role_card"]["role_card_delta"]["online_behavior"],
            ["询问资源"],
        )

    def test_llm_mode_accepts_negative_opinion_score(self):
        config = AgentConfig(
            initial_opinion=0.0,
            opinion_assessment_mode="llm",
        )
        agent, _ = self._agent(config=config)
        agent.current_focus = "姜萍事件"
        assessor = OpinionAssessmentCoordinator(
            config,
            llm=_LLMStub('{"score": -0.72, "confidence": 0.9, "reason": "明显质疑", "evidence": ["违规说明"]}'),
        )

        result = assessor.assess_agent(agent, 1)

        self.assertEqual(result["source"], "llm_context_assessment")
        self.assertEqual(result["score"], -0.72)
        self.assertEqual(agent.opinion, -0.72)
        self.assertEqual(agent.opinion_scores[JIANG_PING_TOPIC], -0.72)

    def test_async_llm_mode_uses_json_assessment(self):
        config = AgentConfig(
            initial_opinion=0.7,
            opinion_assessment_mode="llm",
        )
        agent, _ = self._agent(config=config)
        agent.current_focus = "resource policy"
        assessor = OpinionAssessmentCoordinator(
            config,
            llm=_LLMStub('{"score": 0.44, "confidence": 0.8, "reason": "异步理由", "evidence": ["证据B"]}'),
        )

        result = __import__("asyncio").run(assessor.aassess_agent(agent, 1))

        self.assertEqual(result["source"], "llm_context_assessment")
        self.assertEqual(result["score"], 0.44)
        self.assertEqual(result["confidence"], 0.8)
        self.assertEqual(result["reason"], "异步理由")
        self.assertEqual(result["topic"], JIANG_PING_TOPIC)
        self.assertEqual(agent.opinion, 0.44)

    def test_async_llm_context_contains_dynamic_role_card_once(self):
        config = AgentConfig(
            initial_opinion=0.7,
            opinion_assessment_mode="llm",
        )
        agent, _ = self._agent(config=config)
        agent.last_psychological_assessment = {
            "status": "single_evaluator",
            "activated_needs": ["money"],
            "role_card_delta": {
                "summary": "经济安全压力上升",
                "decision_bias": ["更关注资源保障"],
            },
        }
        llm = _LLMStub('{"score": 0.12, "confidence": 0.8, "reason": "异步", "evidence": ["证据"]}')
        assessor = OpinionAssessmentCoordinator(config, llm=llm)

        result = __import__("asyncio").run(assessor.aassess_agent(agent, 1))
        user_payload = __import__("json").loads(llm.last_user)
        context = user_payload["context"]

        self.assertEqual(result["score"], 0.12)
        self.assertIn("dynamic_role_card", context)
        self.assertNotIn("psychological_context", context)
        self.assertNotIn("psychological_context", result)
        self.assertEqual(
            context["dynamic_role_card"]["role_card_delta"]["summary"],
            "经济安全压力上升",
        )

    def test_repeated_rule_assessment_does_not_ratchet_topic_score(self):
        config = AgentConfig(initial_opinion=0.7, opinion_assessment_mode="rule")
        mem = _MemoryStub(["我支持姜萍，因为她的故事有鼓励意义。"])
        agent, assessor = self._agent(config=config, mem=mem)
        agent.current_focus = "UBI policy"
        agent.add_history("event", "今天看到姜萍事件中的励志和正面叙事。")

        first = assessor.assess_agent(agent, 1)
        second = assessor.assess_agent(agent, 2)

        self.assertEqual(second["score"], first["score"])
        self.assertEqual(agent.opinion, first["score"])

    def test_async_assess_all_isolates_single_agent_failure(self):
        config = AgentConfig(initial_opinion=0.7, opinion_assessment_mode="rule")
        good_agent, assessor = self._agent(config=config)
        bad_agent, _ = self._agent(config=config)
        bad_agent.id = "bad_agent"
        good_agent.current_focus = "good"
        bad_agent.current_focus = "bad"
        original = assessor.aassess_agent

        async def flaky_assess(agent, tick):
            if agent.id == "bad_agent":
                raise RuntimeError("assessment failed")
            return await original(agent, tick)

        assessor.aassess_agent = flaky_assess

        with patch("persona.opinion.assessment.logger.warning"):
            __import__("asyncio").run(assessor.aassess_all([good_agent, bad_agent], 1))

        self.assertIsNotNone(good_agent.last_opinion_assessment)

    def test_history_limit_is_applied(self):
        agent, assessor = self._agent()

        for tick in range(1, 4):
            agent.current_focus = f"topic_{tick}"
            assessor.assess_agent(agent, tick)

        self.assertEqual(len(agent.opinion_assessment_history), 2)
        self.assertEqual(agent.opinion_assessment_history[0]["topic"], JIANG_PING_TOPIC)
        self.assertEqual(agent.opinion_assessment_history[1]["topic"], JIANG_PING_TOPIC)
        self.assertEqual(agent.opinion_assessment_history[0]["tick"], 2)
        self.assertEqual(agent.opinion_assessment_history[1]["tick"], 3)

    def test_default_mode_uses_llm_when_available(self):
        config = AgentConfig(initial_opinion=0.7)
        agent, _ = self._agent(config=config)
        agent.current_focus = "resource policy"
        assessor = OpinionAssessmentCoordinator(
            config,
            llm=_LLMStub('{"score": 0.21, "confidence": 0.7, "reason": "默认 LLM", "evidence": ["证据"]}'),
        )

        result = assessor.assess_agent(agent, 1)

        self.assertEqual(result["source"], "llm_context_assessment")
        self.assertEqual(result["score"], 0.21)
        self.assertEqual(result["topic"], JIANG_PING_TOPIC)
        self.assertEqual(agent.opinion, 0.21)

    def test_default_llm_failure_falls_back_to_rule(self):
        config = AgentConfig(initial_opinion=0.7)
        mem = _MemoryStub(["我支持姜萍，因为这是一个励志的正面叙事。"])
        agent, _ = self._agent(config=config, mem=mem)
        agent.current_focus = "UBI policy"
        assessor = OpinionAssessmentCoordinator(config, llm=_LLMStub("not-json"))

        result = assessor.assess_agent(agent, 1)

        self.assertEqual(result["source"], "rule_context_assessment")
        self.assertGreater(result["score"], 0.7)
        self.assertEqual(agent.opinion, result["score"])


if __name__ == "__main__":
    unittest.main()
