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
    parser.add_argument("--output-base", default=str(Path(__file__).parent / "history"))
    parser.add_argument("--opinion-mode", choices=["llm", "rule"], default="llm")
    parser.add_argument("--psychology-mode", choices=["llm", "rule", "off"], default="llm")
    parser.add_argument("--llm", choices=["real", "mock"], default="real")
    return parser.parse_args()


async def run_experiment(args: argparse.Namespace) -> Path:
    """构建默认场景并运行指定 tick 数。"""

    random.seed(args.seed)
    config = _build_config(args)
    recorder = HistoryRecorder(base_dir=args.output_base)
    llm_client = _build_llm_client(args, config)
    scenario = get_scenario(args.scenario)
    runtime = scenario.build_runtime(history_recorder=recorder, reset_memory=True, config=config, llm_client=llm_client)
    _apply_experiment_version(runtime, args.version)
    # 命令行实验一旦进入运行流程，就创建输出目录并保存配置快照。
    recorder.ensure_output_dir()
    _write_config_snapshot(runtime, recorder.output_dir, args)

    try:
        for _ in range(config.simulation_step_limit):
            await runtime.world.astep()
    finally:
        recorder.close()
        runtime.mem.close()

    analyze_history_run(
        recorder.output_dir,
        config_snapshot=_config_snapshot(runtime, args),
    )
    return Path(recorder.output_dir)


def _build_config(args: argparse.Namespace) -> AgentConfig:
    config = AgentConfig()
    config.simulation_step_limit = max(1, int(args.ticks))
    config.opinion_assessment_mode = args.opinion_mode
    config.post_opinion_scoring_mode = args.opinion_mode
    config.psychological_assessment_mode = "rule" if args.psychology_mode == "rule" else "llm"
    if args.psychology_mode == "off":
        # 关闭心理评测时把阈值提高到不可触发，保留世界和观念评测流程。
        config.psychological_pressure_default_threshold = 999.0
        config.psychological_pressure_thresholds = {
            "satiety": 999.0,
            "relax": 999.0,
            "money": 999.0,
        }
    return config


def _build_llm_client(args: argparse.Namespace, config: AgentConfig):
    if args.llm == "mock":
        # mock LLM 只用于无外部模型的集成验收，不改变默认真实 LLM 路径。
        from persona.llm.mock_client import MockLLMClient
        return MockLLMClient(config)
    return None


def _apply_experiment_version(runtime, version: str) -> None:
    if version == "no_psychology":
        runtime.config.psychological_pressure_default_threshold = 999.0
        runtime.config.psychological_pressure_thresholds = {
            "satiety": 999.0,
            "relax": 999.0,
            "money": 999.0,
        }
        runtime.config.dynamic_role_card_enabled = False
    elif version == "no_role_card":
        runtime.config.dynamic_role_card_enabled = False


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
    for agent_id in scenario_agent_ids:
        agent = runtime.world.agents.get(agent_id)
        if agent is None:
            continue
        agents[agent_id] = {
            "position": list(agent.position),
            "opinion": agent.opinion,
            "satisfaction": dict(agent.satisfaction),
            "speaking_style": agent.speaking_style,
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
        "llm": args.llm,
        "ticks": runtime.config.simulation_step_limit,
        "default_opinion_topic": runtime.config.default_opinion_topic,
        "opinion_assessment_mode": runtime.config.opinion_assessment_mode,
        "psychological_assessment_mode": runtime.config.psychological_assessment_mode,
        "dynamic_role_card_enabled": runtime.config.dynamic_role_card_enabled,
        "agent_config": asdict(runtime.config),
        "agents": agents,
    }


if __name__ == "__main__":
    setup_logging()
    load_dotenv()
    output = asyncio.run(run_experiment(parse_args()))
    print(f"experiment_output_dir={output}")
