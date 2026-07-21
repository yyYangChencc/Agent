from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from analyze_history import analyze_history_run
from persona.logger import get_logger

logger = get_logger(__name__)


def archive_runtime_run(
    runtime,
    recorder,
    *,
    reason: str,
    scenario_name: str = "",
    voting_results: list[dict] | None = None,
) -> dict | None:
    """关闭当前记录器，并为已有 tick 生成配置快照、摘要和图表。"""

    if runtime is None or recorder is None:
        return None

    tick_count = int(getattr(runtime.world, "time", 0) or 0)
    recorder.close()
    if tick_count <= 0:
        return None

    # 只有真实运行过 tick 才创建归档目录并补写配置快照。
    output_dir = Path(recorder.ensure_output_dir())
    # 未预先完成投票时保留同步入口，server 可传入异步并行结果。
    if voting_results is None:
        voting_results = runtime.world.opinion_assessor.finalize_voting(
            list(runtime.world.agents.values()),
            tick_count,
        )
    (output_dir / "opinion_voting_posthoc.json").write_text(
        json.dumps(voting_results, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    with (output_dir / "platform_exposure_events.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for event in getattr(runtime.platform, "exposure_events", []) or []:
            handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
    # 即使平台尚未产生事件，也保留空账本以明确归档契约。
    with (output_dir / "platform_events.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for event in getattr(runtime.platform, "events", []) or []:
            handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
    config_snapshot = build_runtime_config_snapshot(runtime, reason=reason, scenario_name=scenario_name)
    (output_dir / "config_snapshot.json").write_text(
        json.dumps(config_snapshot, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    summary_path = None
    summary_error = ""
    try:
        summary_path = analyze_history_run(output_dir, config_snapshot=config_snapshot).summary_path
    except Exception as exc:
        summary_error = str(exc)
        logger.warning("[Archive] summarize run failed output_dir=%s: %s", output_dir, exc, exc_info=True)

    run_name = output_dir.name
    output_names = [
        "opinion_trends.svg",
        "opinion_dashboard.html",
        "opinion_distribution.svg",
        "opinion_distribution.csv",
        "opinion_analysis.json",
        "opinion_voting_trends.svg",
        "opinion_voting_polarization_trends.svg",
        "effective_pressure_trends.svg",
        "mediator_peak_trends.svg",
        "memory_event_trends.svg",
        "memory_vector_trends.svg",
        "polarization_report.md",
        "polarization_metrics.csv",
        "polarization_agent_shift.csv",
        "opinion_voting_polarization_report.md",
        "opinion_voting_polarization_metrics.csv",
        "opinion_voting_agent_metrics.csv",
        "opinion_voting_skipped_records.csv",
        "opinion_voting_posthoc.json",
        "platform_exposure_events.jsonl",
        "platform_events.jsonl",
    ]
    archived = {
        "run_name": run_name,
        "output_dir": str(output_dir),
        "tick_count": tick_count,
        "reason": reason,
        "summary_path": str(summary_path) if summary_path is not None else "",
        "summary_url": f"/history/{run_name}/experiment_summary.md" if summary_path is not None else "",
        "config_url": f"/history/{run_name}/config_snapshot.json",
        # 只向前端返回本次分析实际生成的文件，避免模式分流后出现失效链接。
        "charts": [
            {"name": name, "url": f"/history/{run_name}/{name}"}
            for name in output_names
            if (output_dir / name).exists()
        ],
        "error": summary_error,
    }
    logger.info("[Archive] archived current run before %s: %s", reason, archived)
    return archived


def build_runtime_config_snapshot(runtime, *, reason: str, scenario_name: str = "") -> dict:
    """保存前端运行的配置和智能体当前状态，保证 reset 前结果可复现。"""

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
        "source": "frontend_reset_archive",
        "reason": reason,
        "scenario_name": getattr(runtime, "scenario_name", scenario_name),
        "scenario_agent_ids": scenario_agent_ids,
        "recorded_ticks": int(getattr(runtime.world, "time", 0) or 0),
        "influencers": list(scenario_spec.get("influencers") or []),
        "official_news_schedule": scenario_spec.get("official_news_schedule") or {},
        "influencer_schedule": scenario_spec.get("influencer_schedule") or {},
        "default_opinion_topic": runtime.config.default_opinion_topic,
        "opinion_assessment_mode": runtime.config.opinion_assessment_mode,
        "psychological_assessment_mode": runtime.config.psychological_assessment_mode,
        "psychological_assessment_enabled": runtime.config.psychological_assessment_enabled,
        "dynamic_role_card_enabled": runtime.config.dynamic_role_card_enabled,
        "dynamic_role_card_behavior_enabled": runtime.config.dynamic_role_card_behavior_enabled,
        "dynamic_role_card_opinion_enabled": runtime.config.dynamic_role_card_opinion_enabled,
        "agent_config": asdict(runtime.config),
        "agents": agents,
    }
