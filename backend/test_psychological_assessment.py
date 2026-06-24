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
from persona.psychology import PsychologicalAssessmentCoordinator


class _World:
    time = 0

    def __init__(self):
        self.agents = {}

    def add_agent(self, agent):
        self.agents[agent.id] = agent


class _Stub:
    pass


class _LLMStub:
    def __init__(self, response):
        self.response = response
        self.generate_calls = 0
        self.agenerate_calls = 0

    def generate(self, system, user):
        self.generate_calls += 1
        return self.response

    async def agenerate(self, system, user):
        self.agenerate_calls += 1
        return self.response


class PsychologicalAssessmentTest(unittest.TestCase):
    def _agent_and_coordinator(self, *, config=None, llm=None):
        if config is None:
            config = AgentConfig(
                satiety_threshold=30.0,
                relax_threshold=30.0,
                money_threshold=1.0,
                psychological_assessment_interval=3,
                psychological_pressure_thresholds={
                    "satiety": 0.5,
                    "relax": 0.5,
                    "money": 0.5,
                },
                psychological_recovery_pressure_threshold=0.25,
                psychological_mediator_decay_rate=0.5,
                psychological_mediator_clear_threshold=0.05,
                psychological_assessment_mode="rule",
            )
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
        return agent, PsychologicalAssessmentCoordinator(config, llm)

    def _llm_config(self):
        config = AgentConfig(
            satiety_threshold=30.0,
            relax_threshold=30.0,
            money_threshold=1.0,
            psychological_assessment_interval=3,
            psychological_pressure_thresholds={
                "satiety": 0.5,
                "relax": 0.5,
                "money": 0.5,
            },
            psychological_recovery_pressure_threshold=0.25,
            psychological_mediator_decay_rate=0.5,
            psychological_mediator_clear_threshold=0.05,
        )
        return config

    def test_assessment_waits_for_interval_and_activates_threshold_crossing_need(self):
        agent, coordinator = self._agent_and_coordinator()
        agent.effective_pressure = {"satiety": 0.0, "relax": 0.0, "money": 0.0}
        agent.effective_pressure["satiety"] = 0.7
        agent.add_history("event", "hungry")

        coordinator.observe_agent_tick(agent, 1)
        self.assertIsNone(coordinator.maybe_assess(agent, 1))
        coordinator.observe_agent_tick(agent, 2)
        self.assertIsNone(coordinator.maybe_assess(agent, 2))
        coordinator.observe_agent_tick(agent, 3)
        result = coordinator.maybe_assess(agent, 3)

        self.assertIsNotNone(result)
        self.assertEqual(result["status"], "single_evaluator")
        self.assertEqual(result["activated_needs"], ["satiety"])
        self.assertIn("satiety", agent.last_need_assessments)
        self.assertEqual(agent.last_need_assessments["satiety"]["status"], "theory_card_evaluated")
        self.assertIn("resource_seeking", agent.last_need_assessments["satiety"]["mediators"])
        self.assertIn("role_card_delta", result)
        self.assertEqual(result["role_card_delta"]["source_need"], "satiety")

    def test_multiple_active_needs_use_theory_card_arbiter(self):
        agent, coordinator = self._agent_and_coordinator()
        agent.effective_pressure = {"satiety": 0.0, "relax": 0.0, "money": 0.0}
        agent.effective_pressure["satiety"] = 0.8
        agent.effective_pressure["relax"] = 0.9

        for tick in range(1, 4):
            coordinator.observe_agent_tick(agent, tick)
        result = coordinator.maybe_assess(agent, 3)

        self.assertEqual(result["status"], "integrated_theory_cards")
        self.assertEqual(set(result["activated_needs"]), {"satiety", "relax"})
        self.assertIn("evaluator_results", result)
        self.assertIn("role_card_delta", result)
        self.assertIn("dominant_need", result["role_card_delta"])
        self.assertIn("cognition", result["role_card_delta"])

    def test_no_active_need_skips_and_starts_new_window(self):
        agent, coordinator = self._agent_and_coordinator()
        agent.effective_pressure = {"satiety": 0.0, "relax": 0.0, "money": 0.0}
        agent.effective_pressure["satiety"] = 0.1

        for tick in range(1, 4):
            coordinator.observe_agent_tick(agent, tick)
        result = coordinator.maybe_assess(agent, 3)

        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["activated_needs"], [])
        self.assertEqual(agent.psychological_assessment_window.start_tick, 4)

    def test_inactive_pressure_recovers_previous_psychological_state(self):
        agent, coordinator = self._agent_and_coordinator()
        agent.effective_pressure = {"satiety": 0.0, "relax": 0.0, "money": 0.0}
        agent.effective_pressure["satiety"] = 0.8

        for tick in range(1, 4):
            coordinator.observe_agent_tick(agent, tick)
        activated = coordinator.maybe_assess(agent, 3)
        self.assertEqual(activated["status"], "single_evaluator")
        previous_value = agent.last_need_assessments["satiety"]["mediators"]["resource_seeking"]

        agent.effective_pressure = {"satiety": 0.0, "relax": 0.0, "money": 0.0}
        for tick in range(4, 7):
            coordinator.observe_agent_tick(agent, tick)
        recovering = coordinator.maybe_assess(agent, 6)

        self.assertEqual(recovering["status"], "recovering_single")
        self.assertEqual(recovering["recovering_needs"], ["satiety"])
        recovered_value = agent.last_need_assessments["satiety"]["mediators"]["resource_seeking"]
        self.assertLess(recovered_value, previous_value)
        self.assertEqual(recovering["role_card_delta"]["severity"], "recovery")

    def test_recovered_state_is_cleared_after_decay(self):
        agent, coordinator = self._agent_and_coordinator()
        agent.effective_pressure = {"satiety": 0.0, "relax": 0.0, "money": 0.0}
        agent.last_need_assessments["satiety"] = {
            "status": "recovering",
            "mediators": {"resource_seeking": 0.06},
        }

        for tick in range(1, 4):
            coordinator.observe_agent_tick(agent, tick)
        result = coordinator.maybe_assess(agent, 3)

        self.assertEqual(result["status"], "skipped")
        self.assertNotIn("satiety", agent.last_need_assessments)

    def test_default_llm_evaluator_runs_on_threshold_crossing_need(self):
        config = self._llm_config()
        llm = _LLMStub(
            '{"mediators": {"resource_seeking": 0.66, "irritability": 0.42}, '
            '"role_card_delta": {"summary": "饥饿压力上升", "emotion_tone": "急迫", '
            '"cognition": ["优先关注食物"], "behavior": ["寻找补给"], '
            '"social_expression": ["表达更直接"], "online_behavior": ["询问资源"], '
            '"decision_bias": ["提高补给优先级"], "constraints": ["不得编造地点"]}, '
            '"reason": "窗口内饱腹压力超过阈值", "evidence": ["hungry"], "confidence": 0.82}'
        )
        agent, coordinator = self._agent_and_coordinator(config=config, llm=llm)
        agent.effective_pressure = {"satiety": 0.7, "relax": 0.0, "money": 0.0}
        agent.add_history("event", "hungry")

        for tick in range(1, 4):
            coordinator.observe_agent_tick(agent, tick)
        result = coordinator.maybe_assess(agent, 3)

        self.assertEqual(result["status"], "llm_single_evaluator")
        self.assertEqual(agent.last_need_assessments["satiety"]["status"], "llm_theory_card_evaluated")
        self.assertEqual(agent.last_need_assessments["satiety"]["mediators"]["resource_seeking"], 0.66)
        self.assertEqual(result["role_card_delta"]["summary"], "饥饿压力上升")
        self.assertEqual(llm.generate_calls, 1)

    def test_async_llm_evaluator_runs_on_threshold_crossing_need(self):
        config = self._llm_config()
        llm = _LLMStub(
            '{"mediators": {"resource_seeking": 0.55}, '
            '"role_card_delta": {"summary": "异步饥饿压力", "emotion_tone": "急迫"}, '
            '"reason": "异步评测", "evidence": ["hungry"], "confidence": 0.75}'
        )
        agent, coordinator = self._agent_and_coordinator(config=config, llm=llm)
        agent.effective_pressure = {"satiety": 0.7, "relax": 0.0, "money": 0.0}

        for tick in range(1, 4):
            coordinator.observe_agent_tick(agent, tick)
        result = __import__("asyncio").run(coordinator.amaybe_assess(agent, 3))

        self.assertEqual(result["status"], "llm_single_evaluator")
        self.assertEqual(agent.last_need_assessments["satiety"]["status"], "llm_theory_card_evaluated")
        self.assertEqual(llm.agenerate_calls, 1)

    def test_llm_invalid_json_falls_back_to_theory_card_evaluator(self):
        config = self._llm_config()
        llm = _LLMStub("not-json")
        agent, coordinator = self._agent_and_coordinator(config=config, llm=llm)
        agent.effective_pressure = {"satiety": 0.7, "relax": 0.0, "money": 0.0}

        for tick in range(1, 4):
            coordinator.observe_agent_tick(agent, tick)
        result = coordinator.maybe_assess(agent, 3)

        self.assertEqual(result["status"], "single_evaluator")
        self.assertEqual(agent.last_need_assessments["satiety"]["status"], "theory_card_evaluated")
        self.assertIn("resource_seeking", agent.last_need_assessments["satiety"]["mediators"])

    def test_multiple_llm_active_needs_use_existing_arbiter(self):
        config = self._llm_config()
        llm = _LLMStub(
            '{"mediators": {"resource_seeking": 0.58, "fatigue": 0.61}, '
            '"role_card_delta": {"summary": "需求压力上升", "emotion_tone": "紧张", '
            '"cognition": ["关注缺口"], "behavior": ["寻找恢复路径"]}, '
            '"reason": "多需求压力", "evidence": ["pressure"], "confidence": 0.7}'
        )
        agent, coordinator = self._agent_and_coordinator(config=config, llm=llm)
        agent.effective_pressure = {"satiety": 0.8, "relax": 0.9, "money": 0.0}

        for tick in range(1, 4):
            coordinator.observe_agent_tick(agent, tick)
        result = coordinator.maybe_assess(agent, 3)

        self.assertEqual(result["status"], "integrated_theory_cards")
        self.assertEqual(set(result["activated_needs"]), {"satiety", "relax"})
        self.assertEqual(agent.last_need_assessments["satiety"]["status"], "llm_theory_card_evaluated")
        self.assertEqual(agent.last_need_assessments["relax"]["status"], "llm_theory_card_evaluated")


if __name__ == "__main__":
    unittest.main()
