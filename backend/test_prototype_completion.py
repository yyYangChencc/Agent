from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


class DummyCollection:
    def count(self):
        return 0

    def upsert(self, *args, **kwargs):
        return None


class DummyChromaClient:
    def __init__(self, *args, **kwargs):
        pass

    def get_or_create_collection(self, *args, **kwargs):
        return DummyCollection()

    def reset(self):
        return None


# 测试环境可能没有安装 chromadb，先注入最小 stub 再导入依赖记忆模块的业务代码。
sys.modules.setdefault("chromadb", types.SimpleNamespace(PersistentClient=DummyChromaClient))
sys.modules.setdefault("chromadb.config", types.SimpleNamespace(Settings=lambda **kwargs: kwargs))

from persona.agents.agent import Agent
from persona.config import AgentConfig
from persona.conversation.session import ConversationIntent
from persona.history_recorder import HistoryRecorder
from persona.need_events import apply_passive_need_decay
from persona.opinion.assessment import OpinionAssessmentCoordinator
from persona.opinion.scorer import evaluate_opinion
from persona.psychology.assessment import (
    AssessmentWindow,
    PsychologicalAssessmentCoordinator,
    TheoryCardNeedEvaluator,
    TheoryCardRepository,
)
from persona.llm.mock_client import MockLLMClient
from social_sys.platform import SocialPlatform
from social_sys.post import Post
from tools.operator_tools import Operator
from world.objects import building
from world.world import World


class DummyWorld:
    def __init__(self):
        self.time = 0

    def add_agent(self, agent):
        self.agent = agent


class DummyMemory:
    def __init__(self):
        self.opinion_assessments = []

    def retrieve_context(self, *args, **kwargs):
        return []

    async def aretrieve_context(self, *args, **kwargs):
        return []

    def store_opinion_assessment(self, agent_id, assessment):
        self.opinion_assessments.append((agent_id, assessment))


class DummyPolicy:
    def decide(self, *args, **kwargs):
        return '{"think":"test","action":{}}'

    async def adecide(self, *args, **kwargs):
        return '{"think":"test","action":{}}'


class DummyReflect:
    def step(self, agent):
        return None

    async def astep(self, agent):
        return None


class DummyLLM:
    def __init__(self, payload):
        self.payload = payload
        self._config = AgentConfig()

    def generate(self, system, user, *, response_format=None):
        return json.dumps(self.payload, ensure_ascii=False)

    async def agenerate(self, system, user, *, response_format=None):
        return json.dumps(self.payload, ensure_ascii=False)

    def get_embeddings(self, text):
        return []

    async def aget_embeddings(self, text):
        return []


def make_agent(config: AgentConfig | None = None) -> Agent:
    cfg = config or AgentConfig()
    return Agent(
        agent_id="agent_test",
        position=[0, 0],
        world=DummyWorld(),
        policy=DummyPolicy(),
        mem=DummyMemory(),
        reflect=DummyReflect(),
        platform=None,
        social_policy=DummyPolicy(),
        memory_planner=None,
        config=cfg,
    )


class PrototypeCompletionTests(unittest.TestCase):
    def _make_conversation_agents(self):
        world = World()
        cfg = AgentConfig()
        agent_a = Agent(
            agent_id="agent_1",
            position=[1, 1],
            world=world,
            policy=DummyPolicy(),
            mem=DummyMemory(),
            reflect=DummyReflect(),
            platform=None,
            social_policy=DummyPolicy(),
            memory_planner=None,
            config=cfg,
        )
        agent_b = Agent(
            agent_id="agent_2",
            position=[1, 2],
            world=world,
            policy=DummyPolicy(),
            mem=DummyMemory(),
            reflect=DummyReflect(),
            platform=None,
            social_policy=DummyPolicy(),
            memory_planner=None,
            config=cfg,
        )
        return world, agent_a, agent_b

    def _make_social_agents(self):
        config = AgentConfig()
        world = World()
        platform = SocialPlatform(config=config)
        author = Agent(
            agent_id="agent_author",
            position=[1, 1],
            world=world,
            policy=DummyPolicy(),
            mem=DummyMemory(),
            reflect=DummyReflect(),
            platform=platform,
            social_policy=DummyPolicy(),
            memory_planner=None,
            config=config,
        )
        actor = Agent(
            agent_id="agent_actor",
            position=[1, 2],
            world=world,
            policy=DummyPolicy(),
            mem=DummyMemory(),
            reflect=DummyReflect(),
            platform=platform,
            social_policy=DummyPolicy(),
            memory_planner=None,
            config=config,
        )
        platform.add_agent(author)
        platform.add_agent(actor)
        post = Post(1, author.id, "原帖内容", topic=config.default_opinion_topic)
        platform.posts.append(post)
        actor._last_seen_posts = [post]
        return platform, author, actor, post

    def test_scenario_registry_loads_required_scenarios(self):
        from scenarios.registry import get_scenario, list_scenarios

        self.assertIn("default_town", list_scenarios())
        self.assertIn("iac_gay_marriage", list_scenarios())
        self.assertIn("iac_gay_marriage_10", list_scenarios())
        self.assertIn("jiang_ping_polarization", list_scenarios())
        self.assertEqual(get_scenario("default_town").SPEC["name"], "default_town")
        self.assertEqual(get_scenario("iac_gay_marriage").SPEC["name"], "iac_gay_marriage")
        self.assertEqual(get_scenario("iac_gay_marriage_10").SPEC["name"], "iac_gay_marriage_10")
        self.assertEqual(get_scenario("jiang_ping_polarization").SPEC["name"], "jiang_ping_polarization")

    def test_polarization_scenario_builds_agents_and_influencers(self):
        from scenarios.registry import get_scenario

        config = AgentConfig()
        scenario = get_scenario("jiang_ping_polarization")
        runtime = scenario.build_runtime(config=config, llm_client=MockLLMClient(config))
        try:
            self.assertEqual(len(runtime.world.agents), 10)
            self.assertEqual(len(runtime.platform.influencers), 6)
            self.assertIn("jp_support_1", runtime.platform.influencers)
            self.assertIn("jp_oppose_1", runtime.platform.influencers)
            self.assertNotIn("jp_support_1", runtime.world.agents)
            self.assertNotIn("jp_oppose_1", runtime.world.agents)
        finally:
            runtime.mem.close()

    def test_iac_gay_marriage_scenario_uses_dataset_seed_and_injection_posts(self):
        from scenarios.registry import get_scenario

        config = AgentConfig()
        scenario = get_scenario("iac_gay_marriage")
        runtime = scenario.build_runtime(config=config, llm_client=MockLLMClient(config))
        try:
            self.assertEqual(config.default_opinion_topic, "For or Against Gay Marriage")
            self.assertEqual(len(runtime.world.agents), 105)
            self.assertEqual(len(runtime.platform.influencers), 6)
            self.assertIn("iac_gm_support_1", runtime.platform.influencers)
            self.assertIn("iac_gm_oppose_1", runtime.platform.influencers)
            self.assertNotIn("iac_gm_support_1", runtime.world.agents)
            self.assertNotIn("iac_gm_oppose_1", runtime.world.agents)

            first_spec = scenario.SPEC["agents"][0]
            first_agent = runtime.world.agents[first_spec["id"]]
            # 数据集场景显式带有初始观念，因此不走普通场景的开局中立逻辑。
            self.assertEqual(first_agent.opinion, first_spec["initial_opinion"])
            self.assertEqual(
                first_agent.opinion_scores["For or Against Gay Marriage"],
                first_spec["initial_opinion"],
            )

            injected_count = sum(len(items) for items in runtime.world.influencer_schedule.values())
            self.assertEqual(injected_count, 52)
        finally:
            runtime.mem.close()

    def test_iac_gay_marriage_10_scenario_is_derived_subset(self):
        from scenarios.registry import get_scenario

        config = AgentConfig()
        scenario = get_scenario("iac_gay_marriage_10")
        runtime = scenario.build_runtime(config=config, llm_client=MockLLMClient(config))
        try:
            expected_ids = [
                "iac_author_323",
                "iac_author_399",
                "iac_author_1491",
                "iac_author_105",
                "iac_author_1438",
                "iac_author_1234",
                "iac_author_817",
                "iac_author_883",
                "iac_author_357",
                "iac_author_99",
            ]
            self.assertEqual(scenario.SPEC["parent_scenario"], "iac_gay_marriage")
            self.assertEqual(scenario.SPEC["agent_ids"], expected_ids)
            self.assertEqual(list(runtime.world.agents.keys()), expected_ids)
            self.assertEqual(len(runtime.world.agents), 10)
            self.assertEqual(len(scenario.SPEC["memories"]), 10)
            self.assertEqual(len(runtime.platform.influencers), 6)
            self.assertNotIn("iac_gm_support_1", runtime.world.agents)

            injected_count = sum(len(items) for items in runtime.world.influencer_schedule.values())
            self.assertEqual(injected_count, 52)
            self.assertEqual(config.default_opinion_topic, "For or Against Gay Marriage")
            self.assertEqual(runtime.world.agents["iac_author_323"].opinion, 0.6)
            self.assertEqual(runtime.world.agents["iac_author_1234"].opinion, -1.0)
            self.assertEqual(runtime.world.agents["iac_author_817"].opinion, 0.0)
        finally:
            runtime.mem.close()

    def test_scenarios_start_with_neutral_opinion_before_news_exposure(self):
        from scenarios.registry import get_scenario

        config = AgentConfig()
        for scenario_name in ["default_town", "jiang_ping_polarization"]:
            scenario = get_scenario(scenario_name)
            runtime = scenario.build_runtime(config=config, llm_client=MockLLMClient(config))
            try:
                for agent in runtime.world.agents.values():
                    # 接触系统新闻主题前，opinion 及其评测镜像都必须保持未知/中立。
                    self.assertEqual(agent.opinion, 0.0)
                    self.assertEqual(agent.opinion_scores, {})
                    self.assertIsNone(agent.last_opinion_assessment)
                    self.assertEqual(agent.opinion_assessment_history, [])
                    self.assertEqual(agent.opinion_seen_posts_buffer, [])
            finally:
                runtime.mem.close()

    def test_scenario_initializes_moderate_needs_and_facility_memory(self):
        from scenarios.registry import get_scenario

        config = AgentConfig()
        scenario = get_scenario("default_town")
        runtime = scenario.build_runtime(config=config, llm_client=MockLLMClient(config))
        try:
            agent = runtime.world.agents["agent_1"]
            self.assertEqual(agent.satisfaction["satiety"], 55.0)
            self.assertEqual(agent.satisfaction["relax"], 55.0)
            structured = runtime.mem.list_agent_structured_memories("agent_1")
            memories = structured.get("derived_memories", [])
            facility_rows = [
                row for row in memories
                if row.get("object_id") == "scenario_facilities"
            ]
            self.assertTrue(facility_rows)
            facility_text = facility_rows[0].get("summary", "")
            self.assertIn("shop_1", facility_text)
            self.assertIn("company_1", facility_text)
            self.assertIn("playground_1", facility_text)
        finally:
            runtime.mem.close()

    def test_polarization_follow_groups_are_separated(self):
        from scenarios.registry import get_scenario

        config = AgentConfig()
        scenario = get_scenario("jiang_ping_polarization")
        runtime = scenario.build_runtime(config=config, llm_client=MockLLMClient(config))
        try:
            self.assertIn("jp_support_1", runtime.world.agents["agent_1"].followers)
            self.assertNotIn("jp_oppose_1", runtime.world.agents["agent_1"].followers)
            self.assertIn("jp_oppose_1", runtime.world.agents["agent_5"].followers)
            self.assertNotIn("jp_support_1", runtime.world.agents["agent_5"].followers)
            self.assertIn("jp_support_1", runtime.world.agents["agent_9"].followers)
            self.assertIn("jp_oppose_1", runtime.world.agents["agent_9"].followers)
        finally:
            runtime.mem.close()

    def test_polarization_official_news_has_no_real_final_result(self):
        from scenarios.registry import get_scenario

        scenario = get_scenario("jiang_ping_polarization")
        text = json.dumps(scenario.SPEC["official_news_schedule"], ensure_ascii=False)
        self.assertNotIn("违规", text)
        self.assertNotIn("教师也被点名", text)
        self.assertNotIn("处理结果", text)

    def test_official_news_starts_at_first_tick(self):
        from scenarios.registry import get_scenario

        for scenario_name in ["default_town", "jiang_ping_polarization", "iac_gay_marriage", "iac_gay_marriage_10"]:
            scenario = get_scenario(scenario_name)
            schedule = scenario.SPEC["official_news_schedule"]
            # 官方新闻从第一个时间步开始投放，避免开局长时间没有主题输入。
            self.assertIn(1, schedule)

    def test_theory_cards_match_runtime_need_keys(self):
        config = AgentConfig()
        agent = make_agent(config)
        repository = TheoryCardRepository.load_default()
        runtime_need_keys = set(agent.satisfaction.keys())
        expected_need_keys = {
            "satiety",
            "relax",
            "money",
            "belonging",
            "esteem",
            "self_actualization",
        }
        # 理论卡只覆盖运行时真实存在的需求键；money 是安全需求代理，不再单列 safety。
        self.assertEqual(runtime_need_keys, expected_need_keys)
        self.assertEqual(set(repository.by_need_key.keys()), expected_need_keys)
        self.assertNotIn("safety", repository.by_need_key)
        self.assertEqual(len(repository.cards), len(expected_need_keys))

    def test_theory_card_schema_is_complete(self):
        repository = TheoryCardRepository.load_default()
        required_fields = {
            "id",
            "need_key",
            "display_name",
            "maslow_need",
            "summary",
            "source_anchors",
            "mediators",
            "role_card_templates",
            "validation_predictions",
        }
        required_mediator_fields = {
            "key",
            "base",
            "pressure_weight",
            "decline_weight",
            "previous_weight",
        }
        for card in repository.cards:
            # 每张卡都必须能直接供规则评测器和 LLM 评测器使用。
            self.assertTrue(required_fields.issubset(card.keys()))
            self.assertGreaterEqual(len(card["mediators"]), 3)
            self.assertEqual(set(card["role_card_templates"].keys()), {"low", "moderate", "high"})
            mediator_keys = [mediator["key"] for mediator in card["mediators"]]
            self.assertEqual(len(mediator_keys), len(set(mediator_keys)))
            for mediator in card["mediators"]:
                self.assertTrue(required_mediator_fields.issubset(mediator.keys()))
                self.assertIsInstance(mediator["key"], str)
                self.assertTrue(mediator["key"])
                for number_key in ["base", "pressure_weight", "decline_weight", "previous_weight"]:
                    self.assertIsInstance(mediator[number_key], (int, float))

    def test_money_card_contains_safety_proxy_mediators(self):
        repository = TheoryCardRepository.load_default()
        money_card = repository.get("money")
        self.assertIsNotNone(money_card)
        self.assertEqual(money_card["maslow_need"], "safety")
        mediator_keys = {mediator["key"] for mediator in money_card["mediators"]}
        self.assertIn("economic_insecurity", mediator_keys)
        self.assertIn("risk_avoidance", mediator_keys)
        self.assertIn("control_seeking", mediator_keys)
        self.assertTrue(any("风险" in anchor or "安全" in anchor for anchor in money_card["source_anchors"]))

    def test_influencer_schedule_injects_visible_posts_by_follow_group(self):
        from scenarios.registry import get_scenario

        config = AgentConfig()
        scenario = get_scenario("jiang_ping_polarization")
        runtime = scenario.build_runtime(config=config, llm_client=MockLLMClient(config))
        try:
            runtime.world.time = 8
            runtime.platform.time = 8
            runtime.world._inject_scheduled_news()
            support_visible = runtime.platform.give_post("agent_1")["posts"]
            oppose_visible = runtime.platform.give_post("agent_5")["posts"]
            mixed_visible = runtime.platform.give_post("agent_9")["posts"]
            self.assertTrue(any(post["author_id"] == "jp_support_1" for post in support_visible))
            self.assertFalse(any(post["author_id"] == "jp_oppose_1" for post in support_visible))
            self.assertTrue(any(post["author_id"] == "jp_oppose_1" for post in oppose_visible))
            self.assertFalse(any(post["author_id"] == "jp_support_1" for post in oppose_visible))
            self.assertTrue(any(post["author_id"] == "jp_support_1" for post in mixed_visible))
            self.assertTrue(any(post["author_id"] == "jp_oppose_1" for post in mixed_visible))
            self.assertTrue(all(post["source_type"] == "influencer" for post in mixed_visible))
            runtime.world.agents["agent_9"]._record_opinion_seen_posts(runtime.platform.give_post("agent_9"))
            seen_posts = runtime.world.agents["agent_9"].opinion_seen_posts_buffer
            self.assertTrue(any(post.get("source_type") == "influencer" for post in seen_posts))
            # 历史 JSONL 必须保留投放者来源，保证实验日志可追溯极化暴露。
            with tempfile.TemporaryDirectory() as tmp:
                recorder = HistoryRecorder(base_dir=tmp)
                recorder.record(8, [runtime.world.agents["agent_9"]], platform=runtime.platform)
                recorder.close()
                run_dir = next(Path(tmp).iterdir())
                jsonl_row = json.loads((run_dir / "agent_9.jsonl").read_text(encoding="utf-8"))
            self.assertTrue(any(post.get("source_type") == "influencer" for post in jsonl_row["seen_topic_posts"]))
            self.assertTrue(any(post.get("author_id") == "jp_support_1" for post in jsonl_row["seen_topic_posts"]))
            self.assertTrue(any(post.get("author_id") == "jp_oppose_1" for post in jsonl_row["seen_topic_posts"]))
            self.assertTrue(any(post.get("source_type") == "influencer" for post in jsonl_row["visible_topic_posts"]))
        finally:
            runtime.mem.close()

    def test_urgency_floor_and_layer_cap(self):
        config = AgentConfig()
        agent = make_agent(config)
        agent.satisfaction.update({"satiety": 50.0, "relax": 50.0, "money": 50.0})
        agent.update_urgency_from_satisfaction()
        self.assertEqual(agent.urgency["satiety"], config.urgency_floors["satiety"])
        self.assertEqual(agent.urgency["relax"], config.urgency_floors["relax"])
        self.assertEqual(agent.urgency["money"], config.urgency_floors["money"])

        agent.satisfaction.update({"satiety": 0.0, "relax": 50.0, "money": 0.0})
        agent.update_urgency_from_satisfaction()
        physiological_cap = max(agent.urgency["satiety"], agent.urgency["relax"])
        self.assertGreater(agent.urgency["satiety"], config.urgency_floors["satiety"])
        self.assertLessEqual(agent.urgency["money"], physiological_cap)

    def test_pressure_accumulates_recovers_and_sleep_update_does_not_accumulate(self):
        agent = make_agent()
        agent.satisfaction.update({"satiety": 0.0, "relax": 0.0, "money": 0.0})
        agent.update_need_pressure(accumulate=True)
        first_memory = agent.pressure_memory["satiety"]
        self.assertGreater(first_memory, 0.0)

        agent.satisfaction["satiety"] = 100.0
        agent.update_need_pressure(accumulate=True)
        self.assertLess(agent.pressure_memory["satiety"], first_memory)

        before_sleep_memory = agent.pressure_memory["relax"]
        agent.sleeping = True
        agent.tick_sleep_recovery()
        self.assertEqual(agent.pressure_memory["relax"], before_sleep_memory)

    def test_move_has_no_observation_radius_cap(self):
        config = AgentConfig()
        config.observation_radius = 1
        config.relax_moving_usage = 1.0
        world = World()
        agent = Agent(
            agent_id="agent_move",
            position=[0, 0],
            world=world,
            policy=DummyPolicy(),
            mem=DummyMemory(),
            reflect=DummyReflect(),
            platform=None,
            social_policy=DummyPolicy(),
            memory_planner=None,
            config=config,
        )
        agent.update_satisfaction("relax", 20.0)
        result = Operator(world).move("agent_move", 0, 6)
        self.assertIn("成功到达目标", result)
        self.assertEqual(agent.position, [0, 6])
        self.assertLess(agent.satisfaction["relax"], 20.0)
        self.assertTrue(agent.did_move_this_tick)

    def test_move_stops_when_relax_reaches_zero(self):
        config = AgentConfig()
        config.relax_moving_usage = 1.0
        world = World()
        agent = Agent(
            agent_id="agent_tired",
            position=[0, 0],
            world=world,
            policy=DummyPolicy(),
            mem=DummyMemory(),
            reflect=DummyReflect(),
            platform=None,
            social_policy=DummyPolicy(),
            memory_planner=None,
            config=config,
        )
        agent.update_satisfaction("relax", 2.0)
        result = Operator(world).move("agent_tired", 0, 6)
        self.assertIn("尚未到达", result)
        self.assertEqual(agent.position, [0, 2])
        self.assertEqual(agent.satisfaction["relax"], 0.0)

        result = Operator(world).move("agent_tired", 0, 6)
        self.assertIn("relax 为 0", result)
        self.assertEqual(agent.position, [0, 2])

    def test_tick_satisfaction_recovers_relax_when_idle_only(self):
        config = AgentConfig()
        config.satiety_decay_rate = 0.0
        config.relax_increase_rate = 1.5
        agent = make_agent(config)
        agent.satisfaction.update({"satiety": 50.0, "relax": 40.0, "money": 0.0})

        agent.tick_satisfaction()
        self.assertEqual(agent.satisfaction["relax"], 41.5)

        agent.did_move_this_tick = True
        agent.tick_satisfaction()
        self.assertEqual(agent.satisfaction["relax"], 41.5)

        agent.did_move_this_tick = False
        agent.did_work_this_tick = True
        agent.tick_satisfaction()
        self.assertEqual(agent.satisfaction["relax"], 41.5)

    def test_opinion_assessment_writes_back_score(self):
        config = AgentConfig()
        config.opinion_assessment_triggered_only = False
        llm = DummyLLM({"score": 0.6, "confidence": 0.9, "reason": "测试", "evidence": ["证据"]})
        agent = make_agent(config)
        assessor = OpinionAssessmentCoordinator(config, llm)
        context = {
            "seen_posts": [{"topic": config.default_opinion_topic, "content": "支持姜萍"}],
            "recent_social": [{"topic": config.default_opinion_topic, "content": "支持姜萍"}],
        }
        payload = assessor._assess_with_configured_method(agent, config.default_opinion_topic, context)
        result = assessor._store_assessment(agent, 1, config.default_opinion_topic, context, payload)
        self.assertEqual(agent.opinion, result["score"])
        self.assertEqual(agent.opinion_scores[config.default_opinion_topic], result["score"])
        self.assertEqual(result["source"], "llm_context_assessment")

    def test_post_opinion_scorer_rule_fallback_is_not_fixed_neutral(self):
        config = AgentConfig()
        config.post_opinion_scoring_mode = "rule"
        positive = evaluate_opinion("我支持姜萍，这个经历很励志", topic=config.default_opinion_topic, config=config)
        negative = evaluate_opinion("我质疑成绩真实性，反对媒体造神", topic=config.default_opinion_topic, config=config)
        self.assertGreater(positive, 0.0)
        self.assertLess(negative, 0.0)

    def test_send_post_requires_explicit_opinion_index(self):
        config = AgentConfig()
        platform = SocialPlatform(config=config)
        agent = make_agent(config)
        agent.platform = platform
        platform.add_agent(agent)
        action = json.dumps(
            {
                "think": "缺少立场字段，应被拒绝。",
                "action": {"tool": "send_post", "args": {"topic": "姜萍事件", "content": "测试发帖"}},
            },
            ensure_ascii=False,
        )
        feedback = platform.execute(agent.id, action)
        self.assertFalse(feedback["ok"])
        self.assertEqual(len(platform.posts), 0)

    def test_send_post_uses_llm_supplied_opinion_index(self):
        config = AgentConfig()
        platform = SocialPlatform(config=config)
        agent = make_agent(config)
        agent.platform = platform
        platform.add_agent(agent)
        action = json.dumps(
            {
                "think": "直接使用动作中的立场分数。",
                "action": {
                    "tool": "send_post",
                    "args": {"topic": "姜萍事件", "content": "我谨慎支持继续等待证据。", "opinion_index": 0.42},
                },
            },
            ensure_ascii=False,
        )
        feedback = platform.execute(agent.id, action)
        self.assertTrue(feedback["ok"])
        self.assertEqual(len(platform.posts), 1)
        self.assertEqual(platform.posts[0].opinion_index, 0.42)
        self.assertEqual(agent.last_social_action["opinion_index"], 0.42)

    def test_send_post_rejects_out_of_range_opinion_index(self):
        config = AgentConfig()
        platform = SocialPlatform(config=config)
        agent = make_agent(config)
        agent.platform = platform
        platform.add_agent(agent)
        action = json.dumps(
            {
                "think": "超出范围的立场分数应被拒绝。",
                "action": {
                    "tool": "send_post",
                    "args": {"topic": "姜萍事件", "content": "测试发帖", "opinion_index": 1.2},
                },
            },
            ensure_ascii=False,
        )
        feedback = platform.execute(agent.id, action)
        self.assertFalse(feedback["ok"])
        self.assertEqual(len(platform.posts), 0)

    def test_comment_post_requires_agreement_to_post(self):
        config = AgentConfig()
        platform = SocialPlatform(config=config)
        agent = make_agent(config)
        agent.platform = platform
        platform.add_agent(agent)
        post = Post(1, "agent_other", "原帖内容", topic="姜萍事件")
        platform.posts.append(post)
        agent._last_seen_posts = [post]
        action = json.dumps(
            {
                "think": "缺少认同值，应被拒绝。",
                "action": {"tool": "comment_post", "args": {"post_id": 1, "content": "我部分同意。"}},
            },
            ensure_ascii=False,
        )
        feedback = platform.execute(agent.id, action)
        self.assertFalse(feedback["ok"])
        self.assertEqual(post.comments, 0)

    def test_comment_post_records_agreement_to_post(self):
        config = AgentConfig()
        platform = SocialPlatform(config=config)
        agent = make_agent(config)
        agent.platform = platform
        platform.add_agent(agent)
        post = Post(1, "agent_other", "原帖内容", topic="姜萍事件")
        platform.posts.append(post)
        agent._last_seen_posts = [post]
        action = json.dumps(
            {
                "think": "评论时记录对原帖的认同值。",
                "action": {
                    "tool": "comment_post",
                    "args": {"post_id": 1, "content": "我同意需要证据链，但不同意人身攻击。", "agreement_to_post": 0.25},
                },
            },
            ensure_ascii=False,
        )
        feedback = platform.execute(agent.id, action)
        self.assertTrue(feedback["ok"])
        self.assertEqual(post.comments, 1)
        self.assertEqual(post.comments_list[0].agreement_to_post, 0.25)
        self.assertEqual(post.to_dict()["comments"][0]["agreement_to_post"], 0.25)
        self.assertEqual(agent.last_social_action["agreement_to_post"], 0.25)

    def test_social_like_and_dislike_update_author_needs(self):
        platform, author, actor, post = self._make_social_agents()
        before_belonging = author.satisfaction["belonging"]
        before_esteem = author.satisfaction["esteem"]
        like_action = json.dumps(
            {"think": "点赞真实智能体帖子。", "action": {"tool": "like_post", "args": {"post_id": post.id}}},
            ensure_ascii=False,
        )
        feedback = platform.execute(actor.id, like_action)
        self.assertTrue(feedback["ok"])
        self.assertAlmostEqual(author.satisfaction["belonging"], before_belonging + actor.config.social_like_belonging_delta)
        self.assertAlmostEqual(author.satisfaction["esteem"], before_esteem + actor.config.social_like_esteem_delta)
        self.assertTrue(any(event["source"] == "social_feedback" for event in author.need_event_log))

        before_belonging = author.satisfaction["belonging"]
        before_esteem = author.satisfaction["esteem"]
        dislike_action = json.dumps(
            {"think": "点踩真实智能体帖子。", "action": {"tool": "dislike_post", "args": {"post_id": post.id}}},
            ensure_ascii=False,
        )
        feedback = platform.execute(actor.id, dislike_action)
        self.assertTrue(feedback["ok"])
        self.assertAlmostEqual(author.satisfaction["belonging"], before_belonging + actor.config.social_dislike_belonging_delta)
        self.assertAlmostEqual(author.satisfaction["esteem"], before_esteem + actor.config.social_dislike_esteem_delta)

    def test_social_comment_agreement_updates_author_needs(self):
        platform, author, actor, post = self._make_social_agents()
        positive_action = json.dumps(
            {
                "think": "正向评论真实智能体帖子。",
                "action": {
                    "tool": "comment_post",
                    "args": {"post_id": post.id, "content": "我认同你的观点。", "agreement_to_post": 0.8},
                },
            },
            ensure_ascii=False,
        )
        before_belonging = author.satisfaction["belonging"]
        before_esteem = author.satisfaction["esteem"]
        feedback = platform.execute(actor.id, positive_action)
        self.assertTrue(feedback["ok"])
        self.assertAlmostEqual(author.satisfaction["belonging"], before_belonging + actor.config.social_comment_positive_belonging_delta)
        self.assertAlmostEqual(author.satisfaction["esteem"], before_esteem + actor.config.social_comment_positive_esteem_delta)

        negative_action = json.dumps(
            {
                "think": "负向评论真实智能体帖子。",
                "action": {
                    "tool": "comment_post",
                    "args": {"post_id": post.id, "content": "我不认同。", "agreement_to_post": -0.8},
                },
            },
            ensure_ascii=False,
        )
        before_belonging = author.satisfaction["belonging"]
        before_esteem = author.satisfaction["esteem"]
        feedback = platform.execute(actor.id, negative_action)
        self.assertTrue(feedback["ok"])
        self.assertAlmostEqual(author.satisfaction["belonging"], before_belonging + actor.config.social_comment_negative_belonging_delta)
        self.assertAlmostEqual(author.satisfaction["esteem"], before_esteem + actor.config.social_comment_negative_esteem_delta)

    def test_influencer_social_feedback_does_not_create_need_event(self):
        config = AgentConfig()
        platform = SocialPlatform(config=config)
        actor = make_agent(config)
        actor.platform = platform
        platform.add_agent(actor)
        platform.add_influencer({"id": "jp_support_1", "stance": "support"})
        post = Post(1, "jp_support_1", "投放者帖子", topic=config.default_opinion_topic, source_type="influencer")
        platform.posts.append(post)
        actor._last_seen_posts = [post]
        before_self_actualization = actor.satisfaction["self_actualization"]
        action = json.dumps(
            {"think": "点赞无实体投放者帖子。", "action": {"tool": "like_post", "args": {"post_id": post.id}}},
            ensure_ascii=False,
        )
        feedback = platform.execute(actor.id, action)
        self.assertTrue(feedback["ok"])
        self.assertEqual(actor.satisfaction["self_actualization"], before_self_actualization)
        self.assertFalse(any(event["source"] == "social_feedback" for event in actor.need_event_log))

    def test_first_building_and_kind_raise_self_actualization_once(self):
        config = AgentConfig()
        world = World()
        agent = Agent(
            agent_id="agent_explorer",
            position=[0, 0],
            world=world,
            policy=DummyPolicy(),
            mem=DummyMemory(),
            reflect=DummyReflect(),
            platform=None,
            social_policy=DummyPolicy(),
            memory_planner=None,
            config=config,
        )
        building("building_1", [0, 1], world)
        before = agent.satisfaction["self_actualization"]
        result = Operator(world).enter_building(agent.id, "building_1")
        self.assertIn("进入了", result)
        expected_gain = config.self_actualization_first_building_delta + config.self_actualization_first_building_kind_delta
        self.assertAlmostEqual(agent.satisfaction["self_actualization"], before + expected_gain)
        self.assertTrue(any(event["source"] == "exploration" for event in agent.need_event_log))

        Operator(world).exit_building(agent.id)
        before_repeat = agent.satisfaction["self_actualization"]
        Operator(world).enter_building(agent.id, "building_1")
        self.assertEqual(agent.satisfaction["self_actualization"], before_repeat)

    def test_passive_decay_updates_high_level_needs(self):
        agent = make_agent()
        before_belonging = agent.satisfaction["belonging"]
        before_esteem = agent.satisfaction["esteem"]
        apply_passive_need_decay(agent, 10)
        self.assertAlmostEqual(agent.satisfaction["belonging"], before_belonging + agent.config.belonging_passive_decay_delta)
        self.assertEqual(agent.satisfaction["esteem"], before_esteem)

        apply_passive_need_decay(agent, 20)
        self.assertAlmostEqual(agent.satisfaction["esteem"], before_esteem + agent.config.esteem_passive_decay_delta)
        self.assertTrue(any(event["source"] == "passive_decay" for event in agent.need_event_log))

    def test_psychological_recovery_decays_mediators(self):
        card = {
            "need_key": "satiety",
            "display_name": "饱腹需求",
            "maslow_need": "physiological",
            "mediators": [{"key": "scarcity", "base": 0.0, "pressure_weight": 1.0}],
            "role_card_templates": {
                "moderate": {"summary": "资源压力", "emotion_tone": "紧张"}
            },
        }
        agent = make_agent()
        evaluator = TheoryCardNeedEvaluator(card)
        window = AssessmentWindow(start_tick=1)
        window.update_from_agent(agent)
        previous = {"mediators": {"scarcity": 0.8}}
        recovered = evaluator.recover(
            agent=agent,
            window=window,
            previous_result=previous,
            decay_rate=0.5,
            clear_threshold=0.05,
        )
        self.assertIsNotNone(recovered)
        self.assertLess(recovered["mediators"]["scarcity"], 0.8)

    def test_conversation_intent_contains_extended_values(self):
        values = {item.value for item in ConversationIntent}
        for expected in [
            "ask_help",
            "offer_help",
            "self_disclosure",
            "emotional_support",
            "disagreement",
            "conflict",
            "thanks",
            "apology",
        ]:
            self.assertIn(expected, values)

    def test_speak_metadata_flows_to_conversation_message_and_needs(self):
        world, agent_a, agent_b = self._make_conversation_agents()
        before_belonging = agent_b.satisfaction["belonging"]
        decision = {
            "think": "test",
            "action": {
                "tool": "speak",
                "args": {
                    "ID": "agent_2",
                    "content": "我理解你现在压力很大，可以先休息一下。",
                    "response_to": "我有点撑不住了",
                    "intent": "emotional_support",
                    "social_valence": 0.8,
                    "topic": agent_a.config.default_opinion_topic,
                    "topic_stance": 0.35,
                },
            },
        }
        world.execute(agent_a, json.dumps(decision, ensure_ascii=False))
        self.assertEqual(agent_b.inbox[-1]["intent"], "emotional_support")
        self.assertEqual(agent_b.inbox[-1]["social_valence"], 0.8)

        sessions = world.conversation_manager._seed_sessions_from_inboxes([agent_a, agent_b], 2)
        self.assertEqual(len(sessions), 1)
        message = sessions[0].messages[0]
        self.assertEqual(message.intent, ConversationIntent.EMOTIONAL_SUPPORT)
        self.assertEqual(message.social_valence, 0.8)
        self.assertEqual(message.topic, agent_a.config.default_opinion_topic)
        self.assertEqual(message.topic_stance, 0.35)
        self.assertGreater(agent_b.satisfaction["belonging"], before_belonging)
        self.assertTrue(agent_b.need_event_log)
        self.assertEqual(sessions[0].history_entries()[0]["social_valence"], 0.8)

    def test_conversation_metadata_invalid_values_fallback(self):
        world, agent_a, agent_b = self._make_conversation_agents()
        session = world.conversation_manager._new_session(
            "agent_1",
            "agent_2",
            "谢谢你",
            None,
            2,
            "not_a_real_intent",
        )
        message = world.conversation_manager._new_message(
            session,
            0,
            "agent_1",
            "agent_2",
            "谢谢你",
            None,
            "not_a_real_intent",
            "bad",
            "其他话题",
            0.9,
        )
        world.conversation_manager._finalize_message(session, message, [agent_a, agent_b])
        self.assertEqual(message.intent, ConversationIntent.THANKS)
        self.assertEqual(message.social_valence, 0.0)
        self.assertEqual(message.topic, "")
        self.assertIsNone(message.topic_stance)
        self.assertGreater(agent_b.satisfaction["esteem"], 55.0)

    def test_conflict_conversation_reduces_receiver_needs(self):
        world, agent_a, agent_b = self._make_conversation_agents()
        session = world.conversation_manager._new_session("agent_1", "agent_2", "冲突", None, 2, "conflict")
        before_belonging = agent_b.satisfaction["belonging"]
        before_esteem = agent_b.satisfaction["esteem"]
        message = world.conversation_manager._new_message(
            session,
            0,
            "agent_1",
            "agent_2",
            "你这样说很蠢。",
            None,
            "conflict",
            -0.8,
            "",
            None,
        )
        world.conversation_manager._finalize_message(session, message, [agent_a, agent_b])
        self.assertLess(agent_b.satisfaction["belonging"], before_belonging)
        self.assertLess(agent_b.satisfaction["esteem"], before_esteem)

    def test_history_recorder_writes_extended_fields_and_jsonl(self):
        agent = make_agent()
        agent.last_action = {"tool": "social_step", "args": {}, "feedback": "ok", "reward": 0.0}
        agent.last_social_action = {
            "action": "send_post",
            "post_id": 1,
            "post_content": "支持姜萍",
            "comment_content": "",
            "opinion_index": 0.7,
            "agreement_to_post": "",
        }
        agent.last_psychological_assessment = {
            "status": "single_evaluator",
            "activated_needs": ["satiety"],
            "mediators": {"satiety.scarcity": 0.6},
            "role_card_delta": {"summary": "资源压力"},
        }
        agent.last_opinion_assessment = {
            "topic": agent.config.default_opinion_topic,
            "before_score": 0.1,
            "score": 0.2,
            "reason": "测试理由",
            "evidence": ["证据"],
        }
        agent.conversation_event_log.append({
            "time": 1,
            "message_id": "conv_1_m1",
            "session_id": "conv_1",
            "sender": "agent_other",
            "target": agent.id,
            "content": "我支持你",
            "intent": "emotional_support",
            "social_valence": 0.8,
            "topic": "",
            "topic_stance": None,
        })
        agent.need_event_log.append({
            "tick": 1,
            "session_id": "conv_1",
            "message_id": "conv_1_m1",
            "sender": "agent_other",
            "target": agent.id,
            "need_key": "belonging",
            "delta": 2.0,
            "source": "conversation",
            "reason": "收到情绪支持",
        })
        with tempfile.TemporaryDirectory() as tmp:
            recorder = HistoryRecorder(base_dir=tmp)
            recorder.record(1, [agent])
            recorder.close()
            run_dir = next(Path(tmp).iterdir())
            csv_text = (run_dir / "agent_test.csv").read_text(encoding="utf-8")
            jsonl_text = (run_dir / "agent_test.jsonl").read_text(encoding="utf-8")
            jsonl_row = json.loads(jsonl_text)
        self.assertIn("role_card_delta", csv_text)
        self.assertIn("opinion_before", csv_text)
        self.assertIn("post_opinion_index", csv_text)
        self.assertIn("agreement_to_post", csv_text)
        self.assertIn("conversation_messages", csv_text)
        self.assertIn("need_events", csv_text)
        self.assertIn("支持姜萍", jsonl_text)
        self.assertEqual(jsonl_row["post_opinion_index"], 0.7)
        self.assertEqual(jsonl_row["agreement_to_post"], "")
        self.assertIsInstance(jsonl_row["satisfaction"], dict)
        self.assertIn("belonging", jsonl_row["satisfaction"])
        self.assertIsInstance(jsonl_row["need_gap"], dict)
        self.assertIsInstance(jsonl_row["pressure_memory"], dict)
        self.assertIsInstance(jsonl_row["load_saturation"], dict)
        self.assertIsInstance(jsonl_row["effective_pressure"], dict)
        self.assertIsInstance(jsonl_row["mediators"], dict)
        self.assertIsInstance(jsonl_row["role_card_delta"], dict)
        self.assertEqual(jsonl_row["conversation_messages"][0]["intent"], "emotional_support")
        self.assertEqual(jsonl_row["need_events"][0]["need_key"], "belonging")

    def test_history_recorder_creates_directory_only_on_first_record(self):
        agent = make_agent()
        with tempfile.TemporaryDirectory() as tmp:
            recorder = HistoryRecorder(base_dir=tmp)
            output_dir = Path(recorder.output_dir)
            # 仅加载场景或构造记录器时，不应创建空历史目录。
            self.assertFalse(output_dir.exists())

            recorder.record(1, [agent])
            recorder.close()
            self.assertTrue(output_dir.exists())
            self.assertTrue((output_dir / "agent_test.csv").exists())
            self.assertTrue((output_dir / "agent_test.jsonl").exists())

    def test_history_analysis_reads_exact_agent_id_csv_names(self):
        from analyze_history import read_history_rows, resolve_history_run_dir

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            # 分析入口应按表头识别智能体历史，不依赖 agent_ 文件名前缀。
            (run_dir / "iac_author_37.csv").write_text(
                "tick,agent_id,opinion\n1,iac_author_37,0.5\n",
                encoding="utf-8",
            )
            (run_dir / "polarization_metrics.csv").write_text(
                "tick,n_agents,mean_opinion\n1,1,0.5\n",
                encoding="utf-8",
            )

            self.assertEqual(resolve_history_run_dir(run_dir), run_dir)
            rows_by_agent = read_history_rows(run_dir)
            self.assertEqual(list(rows_by_agent.keys()), ["iac_author_37"])
            self.assertEqual(rows_by_agent["iac_author_37"][0]["opinion"], "0.5")

    def test_server_finalize_current_run_generates_summary_and_charts(self):
        from scenarios.registry import get_scenario
        from persona.run_archive import archive_runtime_run

        config = AgentConfig()
        scenario = get_scenario("default_town")
        with tempfile.TemporaryDirectory() as tmp:
            recorder = HistoryRecorder(base_dir=tmp)
            runtime = scenario.build_runtime(
                history_recorder=recorder,
                config=config,
                llm_client=MockLLMClient(config),
            )
            try:
                runtime.world.time = 1
                recorder.record(1, list(runtime.world.agents.values()), platform=runtime.platform)
                archived = archive_runtime_run(
                    runtime,
                    recorder,
                    reason="test_reset",
                    scenario_name="default_town",
                )
                self.assertIsNotNone(archived)
                run_dir = Path(archived["output_dir"])
                self.assertTrue((run_dir / "config_snapshot.json").exists())
                self.assertTrue((run_dir / "experiment_summary.md").exists())
                self.assertTrue((run_dir / "opinion_trends.svg").exists())
                self.assertTrue((run_dir / "polarization_report.md").exists())
                self.assertTrue((run_dir / "polarization_metrics.csv").exists())
                self.assertEqual(archived["tick_count"], 1)
                self.assertTrue(archived["summary_url"].endswith("/experiment_summary.md"))
            finally:
                runtime.mem.close()

    def test_memory_retrieval_ttl_cache_reuses_result(self):
        from persona.agent_memory.mem import MultiAgentMemoryManager

        config = AgentConfig()
        config.memory_retrieval_ttl_ticks = 3
        llm = DummyLLM({})
        manager = object.__new__(MultiAgentMemoryManager)
        manager.llm_client = llm
        manager.agent_collections = {}
        manager._collection_lock = None
        manager._embedding_cache = {}
        manager._retrieval_cache = {}
        import threading
        manager._retrieval_cache_lock = threading.Lock()
        calls = {"count": 0}

        def fake_smart_retrieve(*args, **kwargs):
            calls["count"] += 1
            return [f"memory-{calls['count']}"]

        manager.smart_retrieve = fake_smart_retrieve
        manager.controller = None
        observation = {"time": 1, "type": "observe", "social": {}}
        first = manager.retrieve_context("agent_1", observation, "none", {}, {}, n_results=2, context="world")
        second = manager.retrieve_context("agent_1", observation, "none", {}, {}, n_results=2, context="world")
        self.assertEqual(first, second)
        self.assertEqual(calls["count"], 1)

    def test_mock_llm_returns_valid_action_json(self):
        llm = MockLLMClient(AgentConfig())
        raw = llm.generate("你是自主智能体 agent_test", "## 当前状态\n- 任务：寻找食物\nsatiety: satisfaction=0.0 urgency=1.00 threshold=30.0\n## 观测（半径5格）\n{\"objects\": []}")
        payload = json.loads(raw)
        self.assertIn("think", payload)
        self.assertIn("action", payload)
        self.assertIsInstance(payload["action"], dict)

    def test_mock_social_post_stance_is_scorable(self):
        llm = MockLLMClient(AgentConfig())
        posts_payload = {
            "posts": [
                {
                    "id": 1,
                    "is_news": True,
                    "topic": "姜萍事件",
                    "opinion_index": -0.7,
                }
            ]
        }
        raw = llm.generate("你是社交平台智能体 agent_test", "## 当前浏览的帖子\n" + json.dumps(posts_payload, ensure_ascii=False))
        payload = json.loads(raw)
        content = payload["action"]["args"]["content"]
        self.assertIn("质疑", content)
        self.assertEqual(payload["action"]["args"]["opinion_index"], -0.7)

    def test_mock_social_skips_repeated_topic_post(self):
        llm = MockLLMClient(AgentConfig())
        posts_payload = {
            "posts": [
                {
                    "id": 1,
                    "is_news": True,
                    "topic": "姜萍事件",
                    "opinion_index": -0.7,
                }
            ]
        }
        user = (
            "## 你的发帖历史\n"
            "帖子ID: 8\n"
            "主题: 姜萍事件\n"
            "内容: 我已经表达过对姜萍事件的看法。\n\n"
            "## 当前浏览的帖子\n"
            f"{json.dumps(posts_payload, ensure_ascii=False)}"
        )
        raw = llm.generate("你是社交平台智能体 agent_test", user)
        payload = json.loads(raw)
        self.assertEqual(payload["action"], {})


if __name__ == "__main__":
    unittest.main()
