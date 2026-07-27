from __future__ import annotations

import argparse
import asyncio
import json
import random
from dataclasses import asdict
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv() -> None:
        """测试环境未安装 python-dotenv 时跳过 .env 加载。"""

        return None

from persona.config import AgentConfig
from analyze_history import analyze_history_run
from persona.history_recorder import HistoryRecorder
from persona.logger import setup_logging
from scenarios.registry import get_scenario, list_scenarios


EXPERIMENT_VERSIONS = {"full", "no_psychology", "no_role_card"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行固定 100 tick 原型实验。")
    parser.add_argument("--ticks", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--version", choices=sorted(EXPERIMENT_VERSIONS), default="full")
    parser.add_argument("--scenario", choices=list_scenarios(), default="default_town")
    parser.add_argument(
        "--flan-model",
        default=None,
        help="覆盖场景中的 FLAN 模型名称或本地模型目录",
    )
    parser.add_argument("--output-base", default=str(Path(__file__).parent / "history"))
    parser.add_argument(
        "--opinion-mode",
        choices=["llm_as_judge", "llm_voting", "rule"],
        default="llm_as_judge",
    )
    parser.add_argument("--psychology-mode", choices=["llm", "rule", "off"], default="llm")
    parser.add_argument("--llm", choices=["real", "mock"], default="real")
    return parser.parse_args()


async def run_experiment(args: argparse.Namespace) -> Path:
    """构建默认场景并运行指定 tick 数。"""

    _validate_experiment_contract(args)
    random.seed(args.seed)
    config = _build_config(args)
    recorder = HistoryRecorder(base_dir=args.output_base, scenario_name=args.scenario)
    llm_client = _build_llm_client(args, config)
    scenario = get_scenario(args.scenario)
    runtime = scenario.build_runtime(history_recorder=recorder, reset_memory=True, config=config, llm_client=llm_client)
    # 实验入口的显式观念模式优先于场景默认值，避免场景静默改变处理组。
    runtime.config.opinion_assessment_mode = args.opinion_mode
    if args.flan_model:
        # 命令行路径优先于场景绑定，便于在不同计算节点复用同一实验配置。
        runtime.config.opinion_flan_model_name = args.flan_model
        runtime.world.opinion_assessor.flan_scorer.model_name = args.flan_model
    runtime.config.post_opinion_scoring_mode = "rule" if args.opinion_mode == "rule" else "llm"
    _apply_experiment_version(runtime, args.version)
    # 命令行实验一旦进入运行流程，就创建输出目录并保存配置快照。
    recorder.ensure_output_dir()
    _write_config_snapshot(runtime, recorder.output_dir, args)

    try:
        for _ in range(config.simulation_step_limit):
            await runtime.world.astep()
        # 投票不进入模拟循环，结束后按十个时间步切分全部个人发言。
        voting_results = await runtime.world.opinion_assessor.afinalize_voting(
            list(runtime.world.agents.values()),
            runtime.world.time,
        )
        _write_posthoc_voting(recorder.output_dir, voting_results)
    finally:
        _write_platform_exposure_events(recorder.output_dir, runtime.platform.exposure_events)
        _write_platform_events(recorder.output_dir, getattr(runtime.platform, "events", []) or [])
        recorder.close()
        runtime.mem.close()

    analyze_history_run(
        recorder.output_dir,
        config_snapshot=_config_snapshot(runtime, args),
    )
    return Path(recorder.output_dir)


def _write_posthoc_voting(output_dir: str | Path, voting_results: list[dict]) -> Path:
    """把结束后投票保存为独立结构化文件，避免污染实时状态行。"""

    path = Path(output_dir) / "opinion_voting_posthoc.json"
    path.write_text(
        json.dumps(voting_results, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return path


def _write_platform_exposure_events(output_dir: str | Path, events: list[dict]) -> Path:
    """逐行保存平台曝光阶段，供推荐和关注实验审计。"""

    path = Path(output_dir) / "platform_exposure_events.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
    return path


def _write_platform_events(output_dir: str | Path, events: list[dict]) -> Path:
    """逐行保存统一平台事件，供运行回放和完整性审计。"""

    path = Path(output_dir) / "platform_events.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
    return path


def _build_config(args: argparse.Namespace) -> AgentConfig:
    config = AgentConfig()
    config.simulation_step_limit = max(1, int(args.ticks))
    config.opinion_assessment_mode = args.opinion_mode
    # 发帖立场评分仍使用原有 llm/rule 契约，不复用观念评测方式名称。
    config.post_opinion_scoring_mode = "rule" if args.opinion_mode == "rule" else "llm"
    config.psychological_assessment_mode = args.psychology_mode
    if args.psychology_mode == "off":
        # 显式关闭评测；高阈值只用于兼容读取旧配置的代码路径。
        config.psychological_assessment_enabled = False
        config.psychological_pressure_default_threshold = 999.0
        config.psychological_pressure_thresholds = {
            "satiety": 999.0,
            "relax": 999.0,
            "money": 999.0,
        }
        config.dynamic_role_card_enabled = False
        config.dynamic_role_card_behavior_enabled = False
        config.dynamic_role_card_opinion_enabled = False
    return config


def _build_llm_client(args: argparse.Namespace, config: AgentConfig):
    if args.llm == "mock":
        # mock LLM 只用于无外部模型的集成验收，不改变默认真实 LLM 路径。
        from persona.llm.mock_client import MockLLMClient
        return MockLLMClient(config)
    return None


def _apply_experiment_version(runtime, version: str) -> None:
    if version == "full":
        runtime.config.psychological_assessment_enabled = True
        runtime.config.dynamic_role_card_enabled = True
        runtime.config.dynamic_role_card_behavior_enabled = True
        runtime.config.dynamic_role_card_opinion_enabled = True
    elif version == "no_psychology":
        runtime.config.psychological_assessment_enabled = False
        runtime.config.psychological_pressure_default_threshold = 999.0
        runtime.config.psychological_pressure_thresholds = {
            "satiety": 999.0,
            "relax": 999.0,
            "money": 999.0,
        }
        runtime.config.dynamic_role_card_enabled = False
        runtime.config.dynamic_role_card_behavior_enabled = False
        runtime.config.dynamic_role_card_opinion_enabled = False
    elif version == "no_role_card":
        runtime.config.psychological_assessment_enabled = True
        runtime.config.dynamic_role_card_enabled = False
        runtime.config.dynamic_role_card_behavior_enabled = False
        runtime.config.dynamic_role_card_opinion_enabled = False


def _validate_experiment_contract(args: argparse.Namespace) -> None:
    """拒绝会使实验版本含义坍缩的参数组合。"""

    if args.psychology_mode == "off" and args.version != "no_psychology":
        raise ValueError("--psychology-mode off 只能与 --version no_psychology 同时使用")


def _write_config_snapshot(runtime, output_dir: str, args: argparse.Namespace) -> None:
    path = Path(output_dir) / "config_snapshot.json"
    path.write_text(
        json.dumps(_config_snapshot(runtime, args), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _config_snapshot(runtime, args: argparse.Namespace) -> dict:
    agents = {}
    scenario_agent_ids = list(getattr(runtime, "scenario_agent_ids", [])) or list(runtime.world.agents.keys())
    scenario_spec = getattr(runtime, "scenario_spec", {}) or {}
    spec_agents = {
        str(item.get("id")): item
        for item in (scenario_spec.get("agents") or [])
        if isinstance(item, dict) and item.get("id") is not None
    }
    for agent_id in scenario_agent_ids:
        agent = runtime.world.agents.get(agent_id)
        if agent is None:
            continue
        spec_agent = spec_agents.get(agent_id, {})
        agents[agent_id] = {
            "position": list(agent.position),
            "opinion": agent.opinion,
            "satisfaction": dict(agent.satisfaction),
            "speaking_style": agent.speaking_style,
            "initial_role": spec_agent.get("initial_role"),
            "community_id": spec_agent.get("community_id"),
            "personal_bed_id": spec_agent.get("personal_bed_id"),
            "source_author_id": spec_agent.get("source_author_id"),
            "followers": list(agent.followers),
            "online_trust": dict(agent.online_trust),
            "offline_trust": dict(agent.offline_trust),
        }
    return {
        "seed": args.seed,
        "version": args.version,
        "scenario_name": getattr(runtime, "scenario_name", args.scenario),
        "scenario_agent_ids": scenario_agent_ids,
        "influencers": list(scenario_spec.get("influencers") or []),
        "official_news_schedule": scenario_spec.get("official_news_schedule") or {},
        "influencer_schedule": scenario_spec.get("influencer_schedule") or {},
        "scenario_controls": {
            "dataset": scenario_spec.get("dataset"),
            "discussion_id": scenario_spec.get("discussion_id"),
            "discussion_title": scenario_spec.get("discussion_title"),
            "quarter_time": scenario_spec.get("quarter_time"),
            "source_paths": scenario_spec.get("source_paths") or {},
            "network_mode": scenario_spec.get("network_mode"),
            "controlled_variable": scenario_spec.get("controlled_variable"),
            "topology_rule": scenario_spec.get("topology_rule"),
            "community_assignment_rule": scenario_spec.get("community_assignment_rule"),
            "influencer_follow_rule": scenario_spec.get("influencer_follow_rule"),
            "online_trust_rule": scenario_spec.get("online_trust_rule"),
            "entity_follow_count": scenario_spec.get("entity_follow_count"),
            "influencer_follow_count": scenario_spec.get("influencer_follow_count"),
            "influencer_follow_randomization": scenario_spec.get("influencer_follow_randomization"),
        },
        "llm": args.llm,
        "ticks": runtime.config.simulation_step_limit,
        "default_opinion_topic": runtime.config.default_opinion_topic,
        "opinion_assessment_mode": runtime.config.opinion_assessment_mode,
        "psychological_assessment_mode": runtime.config.psychological_assessment_mode,
        "psychological_assessment_enabled": runtime.config.psychological_assessment_enabled,
        "dynamic_role_card_enabled": runtime.config.dynamic_role_card_enabled,
        "dynamic_role_card_behavior_enabled": runtime.config.dynamic_role_card_behavior_enabled,
        "dynamic_role_card_opinion_enabled": runtime.config.dynamic_role_card_opinion_enabled,
        "requested_opinion_mode": args.opinion_mode,
        "requested_psychology_mode": args.psychology_mode,
        "agent_config": asdict(runtime.config),
        "agents": agents,
    }


if __name__ == "__main__":
    setup_logging()
    load_dotenv()
    output = asyncio.run(run_experiment(parse_args()))
    print(f"experiment_output_dir={output}")
