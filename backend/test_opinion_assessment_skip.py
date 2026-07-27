from types import SimpleNamespace
import unittest

from persona.config import AgentConfig
from persona.opinion.assessment import OpinionAssessmentCoordinator


class FailingOpinionAssessmentCoordinator(OpinionAssessmentCoordinator):
    """固定制造评测失败，隔离外部模型和显卡依赖。"""

    @staticmethod
    def _test_context() -> dict:
        return {
            "self_authored_posts": [{"id": 1, "content": "test"}],
            "self_authored_comments": [],
            "observed_other_speech": [],
            "likes_and_dislikes": [],
        }

    def _build_context(self, agent, topic: str, tick: int) -> dict:
        return self._test_context()

    async def _abuild_context(self, agent, topic: str, tick: int) -> dict:
        return self._test_context()

    def _assess_with_configured_method(self, agent, topic: str, context: dict) -> dict:
        raise RuntimeError("upstream unavailable")

    async def _aassess_with_configured_method(self, agent, topic: str, context: dict) -> dict:
        raise RuntimeError("upstream unavailable")


class OpinionAssessmentSkipTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        config = AgentConfig()
        config.opinion_assessment_mode = "llm_as_judge"
        config.opinion_assessment_interval = 5
        self.coordinator = FailingOpinionAssessmentCoordinator(config, llm=object())
        self.agent = SimpleNamespace(
            id="agent_test",
            opinion=0.53,
            current_focus="",
            task="test",
            opinion_seen_posts_buffer=[{"id": 1}],
            opinion_assessment_history=[],
            last_opinion_assessment=None,
        )

    def _assert_skipped(self, result: dict) -> None:
        self.assertEqual(result["source"], "skipped_llm_failure")
        self.assertEqual(result["reason"], "llm_assessment_failed")
        self.assertEqual(result["score"], 0.53)
        self.assertEqual(self.agent.opinion, 0.53)
        self.assertEqual(self.agent.opinion_seen_posts_buffer, [{"id": 1}])
        self.assertEqual(self.agent.opinion_assessment_history, [])
        self.assertIsNone(self.agent.last_opinion_assessment)

    def test_sync_failure_skips_assessment(self):
        # 同步评测失败后不得调用规则评分或写回观念状态。
        self._assert_skipped(self.coordinator.assess_agent(self.agent, 5))

    async def test_async_failure_skips_assessment(self):
        # 服务端异步评测失败后使用相同的跳过语义。
        result = await self.coordinator.aassess_agent(self.agent, 5)
        self._assert_skipped(result)


if __name__ == "__main__":
    unittest.main()
