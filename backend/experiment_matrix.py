from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from persona.history_csv import configure_csv_field_size_limit


configure_csv_field_size_limit()


SCHEMA_VERSION = 1
RUN_STATUSES = {"pending", "running", "succeeded", "failed", "incomplete"}
EXPERIMENT_VERSIONS = {"full", "no_psychology", "no_role_card"}
OPINION_MODES = {"llm_as_judge", "llm_voting", "rule"}
PSYCHOLOGY_MODES = {"llm", "rule", "off"}
LLM_MODES = {"real", "mock"}
REQUIRED_RUN_FILES = {
    "config_snapshot.json",
    "experiment_summary.md",
    "opinion_voting_posthoc.json",
    "platform_exposure_events.jsonl",
    "platform_events.jsonl",
    "polarization_metrics.csv",
    "polarization_agent_shift.csv",
}
VERSION_SWITCHES = {
    "full": {
        "psychological_assessment_enabled": True,
        "dynamic_role_card_enabled": True,
        "dynamic_role_card_behavior_enabled": True,
        "dynamic_role_card_opinion_enabled": True,
    },
    "no_psychology": {
        "psychological_assessment_enabled": False,
        "dynamic_role_card_enabled": False,
        "dynamic_role_card_behavior_enabled": False,
        "dynamic_role_card_opinion_enabled": False,
    },
    "no_role_card": {
        "psychological_assessment_enabled": True,
        "dynamic_role_card_enabled": False,
        "dynamic_role_card_behavior_enabled": False,
        "dynamic_role_card_opinion_enabled": False,
    },
}
ProcessRunner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class MetricSpec:
    name: str
    source: str
    column: str


@dataclass(frozen=True)
class ComparisonSpec:
    left: str
    right: str


@dataclass(frozen=True)
class MatrixConfig:
    ticks: int
    scenario: str
    opinion_mode: str
    psychology_mode: str
    llm: str
    versions: tuple[str, ...]
    seeds: tuple[int, ...]
    repetitions: int
    timeout_seconds: float | None
    stop_on_failure: bool
    window_size: int
    bootstrap_samples: int
    bootstrap_seed: int
    metrics: tuple[MetricSpec, ...]
    comparisons: tuple[ComparisonSpec, ...]


def load_matrix_config(path: str | Path) -> tuple[MatrixConfig, dict]:
    """读取并严格校验批量实验配置。"""

    config_path = Path(path)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("矩阵配置根节点必须是 JSON 对象。")
    _require_exact_keys(raw, {"schema_version", "experiment", "versions", "seeds", "repetitions", "execution", "summary"}, "根节点")
    if raw["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version 必须为 {SCHEMA_VERSION}。")

    experiment = _require_dict(raw["experiment"], "experiment")
    _require_exact_keys(experiment, {"ticks", "scenario", "opinion_mode", "psychology_mode", "llm"}, "experiment")
    ticks = _require_positive_int(experiment["ticks"], "experiment.ticks")
    scenario = _require_nonempty_string(experiment["scenario"], "experiment.scenario")
    opinion_mode = _require_choice(experiment["opinion_mode"], OPINION_MODES, "experiment.opinion_mode")
    psychology_mode = _require_choice(experiment["psychology_mode"], PSYCHOLOGY_MODES, "experiment.psychology_mode")
    llm = _require_choice(experiment["llm"], LLM_MODES, "experiment.llm")

    versions = tuple(_require_string_list(raw["versions"], "versions"))
    if not versions or len(set(versions)) != len(versions):
        raise ValueError("versions 必须是非空且不重复的字符串数组。")
    invalid_versions = sorted(set(versions) - EXPERIMENT_VERSIONS)
    if invalid_versions:
        raise ValueError(f"versions 存在未支持值：{invalid_versions}")
    if psychology_mode == "off" and any(version != "no_psychology" for version in versions):
        raise ValueError("experiment.psychology_mode=off 只能用于 no_psychology 版本。")

    seeds_raw = raw["seeds"]
    if not isinstance(seeds_raw, list) or not seeds_raw:
        raise ValueError("seeds 必须是非空整数数组。")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in seeds_raw):
        raise ValueError("seeds 中每一项都必须是整数。")
    if len(set(seeds_raw)) != len(seeds_raw):
        raise ValueError("seeds 不得重复。")
    seeds = tuple(seeds_raw)
    repetitions = _require_positive_int(raw["repetitions"], "repetitions")

    execution = _require_dict(raw["execution"], "execution")
    _require_exact_keys(execution, {"timeout_seconds", "stop_on_failure"}, "execution")
    timeout_seconds = execution["timeout_seconds"]
    if timeout_seconds is not None:
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
            raise ValueError("execution.timeout_seconds 必须为 null 或正数。")
        timeout_seconds = float(timeout_seconds)
    stop_on_failure = execution["stop_on_failure"]
    if not isinstance(stop_on_failure, bool):
        raise ValueError("execution.stop_on_failure 必须是布尔值。")

    summary = _require_dict(raw["summary"], "summary")
    _require_exact_keys(summary, {"window_size", "bootstrap_samples", "bootstrap_seed", "metrics", "comparisons"}, "summary")
    window_size = _require_positive_int(summary["window_size"], "summary.window_size")
    if window_size * 2 > ticks:
        raise ValueError("summary.window_size 的首尾窗口不得重叠。")
    bootstrap_samples = _require_nonnegative_int(summary["bootstrap_samples"], "summary.bootstrap_samples")
    bootstrap_seed = _require_int(summary["bootstrap_seed"], "summary.bootstrap_seed")
    metrics = _parse_metrics(summary["metrics"])
    comparisons = _parse_comparisons(summary["comparisons"], versions)

    return MatrixConfig(
        ticks=ticks,
        scenario=scenario,
        opinion_mode=opinion_mode,
        psychology_mode=psychology_mode,
        llm=llm,
        versions=versions,
        seeds=seeds,
        repetitions=repetitions,
        timeout_seconds=timeout_seconds,
        stop_on_failure=stop_on_failure,
        window_size=window_size,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        metrics=metrics,
        comparisons=comparisons,
    ), raw


def _parse_metrics(value: object) -> tuple[MetricSpec, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("summary.metrics 必须是非空对象数组。")
    metrics = []
    for index, item in enumerate(value):
        path = f"summary.metrics[{index}]"
        obj = _require_dict(item, path)
        _require_exact_keys(obj, {"name", "source", "column"}, path)
        metrics.append(MetricSpec(
            name=_require_nonempty_string(obj["name"], f"{path}.name"),
            source=_require_plain_filename(obj["source"], f"{path}.source"),
            column=_require_nonempty_string(obj["column"], f"{path}.column"),
        ))
    names = [metric.name for metric in metrics]
    if len(set(names)) != len(names):
        raise ValueError("summary.metrics 的 name 不得重复。")
    return tuple(metrics)


def _parse_comparisons(value: object, versions: tuple[str, ...]) -> tuple[ComparisonSpec, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("summary.comparisons 必须是非空对象数组。")
    comparisons = []
    for index, item in enumerate(value):
        path = f"summary.comparisons[{index}]"
        obj = _require_dict(item, path)
        _require_exact_keys(obj, {"left", "right"}, path)
        left = _require_nonempty_string(obj["left"], f"{path}.left")
        right = _require_nonempty_string(obj["right"], f"{path}.right")
        if left not in versions or right not in versions:
            raise ValueError(f"{path} 只能引用 versions 中的精确值。")
        if left == right:
            raise ValueError(f"{path}.left 与 {path}.right 不得相同。")
        comparisons.append(ComparisonSpec(left=left, right=right))
    pairs = [(item.left, item.right) for item in comparisons]
    if len(set(pairs)) != len(pairs):
        raise ValueError("summary.comparisons 不得重复。")
    return tuple(comparisons)


def _require_exact_keys(value: dict, expected: set[str], path: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{path} 键集合不匹配；缺少={missing}，多出={extra}。")


def _require_dict(value: object, path: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{path} 必须是 JSON 对象。")
    return value


def _require_nonempty_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} 必须是非空字符串。")
    return value


def _require_plain_filename(value: object, path: str) -> str:
    text = _require_nonempty_string(value, path)
    if Path(text).name != text:
        raise ValueError(f"{path} 必须是运行目录直属文件名。")
    return text


def _require_string_list(value: object, path: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{path} 必须是非空字符串数组。")
    return value


def _require_choice(value: object, choices: set[str], path: str) -> str:
    text = _require_nonempty_string(value, path)
    if text not in choices:
        raise ValueError(f"{path} 必须是 {sorted(choices)} 中的精确值。")
    return text


def _require_int(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} 必须是整数。")
    return value


def _require_positive_int(value: object, path: str) -> int:
    number = _require_int(value, path)
    if number <= 0:
        raise ValueError(f"{path} 必须大于 0。")
    return number


def _require_nonnegative_int(value: object, path: str) -> int:
    number = _require_int(value, path)
    if number < 0:
        raise ValueError(f"{path} 不得小于 0。")
    return number


def build_run_records(config: MatrixConfig) -> list[dict]:
    """按稳定顺序展开版本、种子和重复编号。"""

    records = []
    index = 0
    for repetition in range(1, config.repetitions + 1):
        for seed in config.seeds:
            for version in config.versions:
                index += 1
                records.append({
                    "run_id": f"run-{index:06d}",
                    "version": version,
                    "seed": seed,
                    "repetition": repetition,
                    "status": "pending",
                    "attempts": [],
                    "output_dir": "",
                    "completeness_report": {},
                })
    return records


def run_matrix(
    matrix_path: str | Path,
    output_dir: str | Path,
    *,
    run_script: str | Path | None = None,
    python_executable: str | Path | None = None,
    process_runner: ProcessRunner = subprocess.run,
) -> dict:
    """执行矩阵；已通过完整性门禁的运行会被跳过。"""

    config, raw_config = load_matrix_config(matrix_path)
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    script = Path(run_script) if run_script is not None else Path(__file__).with_name("run_experiment.py")
    executable = str(python_executable or sys.executable)
    manifest_path = root / "batch_manifest.json"
    config_hash = _config_hash(raw_config)
    manifest = _load_or_create_manifest(manifest_path, config, raw_config, config_hash)

    for record in manifest["runs"]:
        prior_output = Path(record["output_dir"]) if record.get("output_dir") else None
        if record.get("status") == "succeeded" and prior_output is not None:
            report = validate_run_completeness(prior_output, config, record)
            record["completeness_report"] = report
            if report["complete"]:
                continue
            record["status"] = "incomplete"
            _write_manifest(manifest_path, manifest)

        attempt_number = len(record["attempts"]) + 1
        attempt_dir = root / "runs" / record["run_id"] / "attempts" / f"attempt-{attempt_number:04d}"
        output_base = attempt_dir / "history"
        output_base.mkdir(parents=True, exist_ok=False)
        command = build_run_command(executable, script, config, record, output_base)
        attempt = {
            "attempt": attempt_number,
            "started_at": _utc_now(),
            "finished_at": "",
            "command": command,
            "exit_code": None,
            "stdout_path": str(attempt_dir / "stdout.txt"),
            "stderr_path": str(attempt_dir / "stderr.txt"),
            "output_dir": "",
            "failure": "",
        }
        record["attempts"].append(attempt)
        record["status"] = "running"
        _write_manifest(manifest_path, manifest)

        try:
            completed = process_runner(
                command,
                cwd=str(script.parent),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=config.timeout_seconds,
                check=False,
            )
            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
            attempt["exit_code"] = completed.returncode
            (attempt_dir / "stdout.txt").write_text(stdout, encoding="utf-8")
            (attempt_dir / "stderr.txt").write_text(stderr, encoding="utf-8")
            if completed.returncode != 0:
                attempt["failure"] = f"run_experiment.py 退出码为 {completed.returncode}。"
                record["status"] = "failed"
            else:
                actual_output = parse_experiment_output_dir(stdout)
                attempt["output_dir"] = str(actual_output)
                record["output_dir"] = str(actual_output)
                report = validate_run_completeness(actual_output, config, record)
                record["completeness_report"] = report
                record["status"] = "succeeded" if report["complete"] else "incomplete"
                if not report["complete"]:
                    attempt["failure"] = "运行产物未通过完整性门禁。"
        except subprocess.TimeoutExpired as exc:
            stdout = _timeout_text(exc.stdout)
            stderr = _timeout_text(exc.stderr)
            (attempt_dir / "stdout.txt").write_text(stdout, encoding="utf-8")
            (attempt_dir / "stderr.txt").write_text(stderr, encoding="utf-8")
            attempt["failure"] = f"运行超过 timeout_seconds={config.timeout_seconds}。"
            record["status"] = "failed"
        except Exception as exc:
            stdout_path = attempt_dir / "stdout.txt"
            stderr_path = attempt_dir / "stderr.txt"
            if not stdout_path.exists():
                stdout_path.write_text("", encoding="utf-8")
            prior_stderr = stderr_path.read_text(encoding="utf-8") if stderr_path.exists() else ""
            separator = "\n" if prior_stderr and not prior_stderr.endswith("\n") else ""
            detail = f"批量器执行失败：{exc}"
            stderr_path.write_text(f"{prior_stderr}{separator}{detail}\n", encoding="utf-8")
            attempt["failure"] = f"批量器执行失败：{exc}"
            record["status"] = "failed"
        finally:
            attempt["finished_at"] = _utc_now()
            _write_manifest(manifest_path, manifest)

        if record["status"] != "succeeded" and config.stop_on_failure:
            break

    write_cross_run_summary(root, config, manifest)
    manifest["updated_at"] = _utc_now()
    _write_manifest(manifest_path, manifest)
    return manifest


def build_run_command(
    executable: str,
    run_script: Path,
    config: MatrixConfig,
    record: dict,
    output_base: Path,
) -> list[str]:
    """仅使用 run_experiment.py 已定义的精确命令行参数。"""

    return [
        executable,
        str(run_script.resolve()),
        "--ticks", str(config.ticks),
        "--seed", str(record["seed"]),
        "--version", str(record["version"]),
        "--scenario", config.scenario,
        "--output-base", str(output_base.resolve()),
        "--opinion-mode", config.opinion_mode,
        "--psychology-mode", config.psychology_mode,
        "--llm", config.llm,
    ]


def parse_experiment_output_dir(stdout: str) -> Path:
    """读取 run_experiment.py 输出的精确目录标记。"""

    prefix = "experiment_output_dir="
    values = [line[len(prefix):] for line in stdout.splitlines() if line.startswith(prefix)]
    if len(values) != 1 or not values[0]:
        raise ValueError("stdout 必须且只能包含一个 experiment_output_dir= 标记。")
    return Path(values[0]).resolve()


def validate_run_completeness(run_dir: str | Path, config: MatrixConfig, record: dict) -> dict:
    """检查配置、实体智能体轨迹和分析产物是否完整。"""

    configure_csv_field_size_limit()
    path = Path(run_dir)
    errors: list[str] = []
    if not path.is_dir():
        return {"complete": False, "errors": [f"运行目录不存在：{path}"], "agent_count": 0, "tick_count": 0}
    for name in sorted(REQUIRED_RUN_FILES):
        if not (path / name).is_file():
            errors.append(f"缺少运行产物：{name}")

    snapshot = _read_json_object(path / "config_snapshot.json", errors)
    expected_snapshot = {
        "seed": record["seed"],
        "version": record["version"],
        "scenario_name": config.scenario,
        "llm": config.llm,
        "ticks": config.ticks,
        "opinion_assessment_mode": config.opinion_mode,
        "psychological_assessment_mode": config.psychology_mode,
        "requested_opinion_mode": config.opinion_mode,
        "requested_psychology_mode": config.psychology_mode,
    }
    expected_snapshot.update(VERSION_SWITCHES[record["version"]])
    for key, expected in expected_snapshot.items():
        if snapshot.get(key) != expected:
            errors.append(f"config_snapshot.json 的 {key}={snapshot.get(key)!r}，预期为 {expected!r}。")

    agent_ids = snapshot.get("scenario_agent_ids")
    if not isinstance(agent_ids, list) or not agent_ids or any(not isinstance(item, str) or not item for item in agent_ids):
        errors.append("config_snapshot.json 的 scenario_agent_ids 必须是非空字符串数组。")
        agent_ids = []
    elif len(set(agent_ids)) != len(agent_ids):
        errors.append("config_snapshot.json 的 scenario_agent_ids 存在重复值。")
    agents = snapshot.get("agents")
    if not isinstance(agents, dict) or set(agents) != set(agent_ids):
        errors.append("config_snapshot.json 的 agents 键集合必须与 scenario_agent_ids 完全一致。")

    agent_config = snapshot.get("agent_config")
    voting_window_size = None
    if not isinstance(agent_config, dict):
        errors.append("config_snapshot.json 的 agent_config 必须是 JSON 对象。")
    else:
        value = agent_config.get("opinion_voting_window_size")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            errors.append("config_snapshot.json 的 agent_config.opinion_voting_window_size 必须为正整数。")
        else:
            voting_window_size = value

    expected_ticks = set(range(1, config.ticks + 1))
    for agent_id in agent_ids:
        _validate_agent_csv(path / f"{agent_id}.csv", agent_id, expected_ticks, errors)
        _validate_agent_jsonl(path / f"{agent_id}.jsonl", agent_id, expected_ticks, errors)
    _validate_metrics_ticks(path / "polarization_metrics.csv", expected_ticks, errors)
    _validate_agent_shift_ids(path / "polarization_agent_shift.csv", set(agent_ids), errors)
    if voting_window_size is not None:
        _validate_posthoc_voting(
            path / "opinion_voting_posthoc.json",
            set(agent_ids),
            config.ticks,
            voting_window_size,
            errors,
        )
    _validate_platform_exposure(path / "platform_exposure_events.jsonl", set(agent_ids), errors)
    _validate_platform_events(path / "platform_events.jsonl", errors)

    return {
        "complete": not errors,
        "errors": errors,
        "agent_count": len(agent_ids),
        "tick_count": config.ticks if agent_ids and not errors else 0,
        "checked_at": _utc_now(),
    }


def _read_json_object(path: Path, errors: list[str]) -> dict:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"无法读取 {path.name}：{exc}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"{path.name} 根节点必须是 JSON 对象。")
        return {}
    return value


def _validate_agent_csv(path: Path, agent_id: str, expected_ticks: set[int], errors: list[str]) -> None:
    if not path.is_file():
        errors.append(f"缺少智能体 CSV：{path.name}")
        return
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            missing = sorted({"tick", "agent_id", "opinion"} - set(fieldnames))
            if missing:
                errors.append(f"{path.name} 缺少列：{missing}")
                return
            rows = list(reader)
        row_ids = {row["agent_id"] for row in rows}
        if row_ids != {agent_id}:
            errors.append(f"{path.name} 的 agent_id 集合为 {sorted(row_ids)}。")
        ticks = _parse_tick_set(rows, path.name, errors)
        if ticks != expected_ticks or len(rows) != len(expected_ticks):
            errors.append(f"{path.name} 的 tick 必须恰好为 1..{len(expected_ticks)} 且不得重复。")
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        errors.append(f"无法读取 {path.name}：{exc}")


def _validate_agent_jsonl(path: Path, agent_id: str, expected_ticks: set[int], errors: list[str]) -> None:
    if not path.is_file():
        errors.append(f"缺少智能体 JSONL：{path.name}")
        return
    rows = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                errors.append(f"{path.name} 第 {line_number} 行必须是 JSON 对象。")
                continue
            rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        errors.append(f"无法读取 {path.name}：{exc}")
        return
    row_ids = {row.get("agent_id") for row in rows}
    if row_ids != {agent_id}:
        errors.append(f"{path.name} 的 agent_id 集合与文件名不一致。")
    ticks = _parse_tick_set(rows, path.name, errors)
    if ticks != expected_ticks or len(rows) != len(expected_ticks):
        errors.append(f"{path.name} 的 tick 必须恰好为 1..{len(expected_ticks)} 且不得重复。")


def _parse_tick_set(rows: list[dict], name: str, errors: list[str]) -> set[int]:
    ticks = []
    for row_number, row in enumerate(rows, start=2):
        value = row.get("tick")
        try:
            tick = int(str(value))
        except (TypeError, ValueError):
            errors.append(f"{name} 第 {row_number} 行 tick 不是整数：{value!r}")
            continue
        ticks.append(tick)
    if len(ticks) != len(set(ticks)):
        errors.append(f"{name} 存在重复 tick。")
    return set(ticks)


def _validate_metrics_ticks(path: Path, expected_ticks: set[int], errors: list[str]) -> None:
    if not path.is_file():
        return
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if "tick" not in (reader.fieldnames or []):
                errors.append(f"{path.name} 缺少 tick 列。")
                return
            rows = list(reader)
        ticks = _parse_tick_set(rows, path.name, errors)
        if ticks != expected_ticks or len(rows) != len(expected_ticks):
            errors.append(f"{path.name} 的 tick 必须恰好为 1..{len(expected_ticks)} 且不得重复。")
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        errors.append(f"无法读取 {path.name}：{exc}")


def _validate_agent_shift_ids(path: Path, expected_ids: set[str], errors: list[str]) -> None:
    if not path.is_file():
        return
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if "agent_id" not in (reader.fieldnames or []):
                errors.append(f"{path.name} 缺少 agent_id 列。")
                return
            actual_ids = {row["agent_id"] for row in reader}
        if actual_ids != expected_ids:
            errors.append(f"{path.name} 的 agent_id 集合与 scenario_agent_ids 不一致。")
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        errors.append(f"无法读取 {path.name}：{exc}")


def _validate_posthoc_voting(
    path: Path,
    expected_ids: set[str],
    final_tick: int,
    window_size: int,
    errors: list[str],
) -> None:
    """核对每个实体智能体是否具备全部互斥投票窗口。"""

    if not path.is_file():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        errors.append(f"无法读取 {path.name}：{exc}")
        return
    if not isinstance(payload, list):
        errors.append(f"{path.name} 根节点必须是 JSON 数组。")
        return
    expected_windows = {
        (start, min(start + window_size - 1, final_tick))
        for start in range(1, final_tick + 1, window_size)
    }
    expected = {
        (agent_id, start, end)
        for agent_id in expected_ids
        for start, end in expected_windows
    }
    actual: set[tuple[str, int, int]] = set()
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            errors.append(f"{path.name} 第 {index + 1} 项必须是 JSON 对象。")
            continue
        agent_id = item.get("agent_id")
        try:
            start = int(item.get("window_start_tick"))
            end = int(item.get("window_end_tick"))
            tick = int(item.get("tick"))
        except (TypeError, ValueError):
            errors.append(f"{path.name} 第 {index + 1} 项的投票窗口字段必须是整数。")
            continue
        key = (agent_id, start, end)
        if not isinstance(agent_id, str) or agent_id not in expected_ids:
            errors.append(f"{path.name} 第 {index + 1} 项的 agent_id 不属于 scenario_agent_ids。")
        if tick != end:
            errors.append(f"{path.name} 第 {index + 1} 项的 tick 必须等于 window_end_tick。")
        if key in actual:
            errors.append(f"{path.name} 存在重复的智能体投票窗口：{key}")
        actual.add(key)
    if actual != expected:
        errors.append(f"{path.name} 的智能体投票窗口集合不完整。")


def _validate_platform_exposure(path: Path, expected_ids: set[str], errors: list[str]) -> None:
    """校验项目曝光审计日志的精确结构。"""

    if not path.is_file():
        return
    required = {
        "tick",
        "receiver_id",
        "feed_mode",
        "limit",
        "eligible_post_ids",
        "ranked_post_ids",
        "displayed_post_ids",
    }
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        errors.append(f"无法读取 {path.name}：{exc}")
        return
    for line_number, line in enumerate(lines, start=1):
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{path.name} 第 {line_number} 行不是有效 JSON：{exc}")
            continue
        if not isinstance(item, dict) or not required.issubset(item):
            errors.append(f"{path.name} 第 {line_number} 行缺少曝光阶段字段。")
            continue
        if item["receiver_id"] not in expected_ids:
            errors.append(f"{path.name} 第 {line_number} 行的 receiver_id 不属于 scenario_agent_ids。")
        for field in ("eligible_post_ids", "ranked_post_ids", "displayed_post_ids"):
            if not isinstance(item[field], list):
                errors.append(f"{path.name} 第 {line_number} 行的 {field} 必须是 JSON 数组。")


def _validate_platform_events(path: Path, errors: list[str]) -> None:
    """校验统一平台事件账本的最低字段契约。"""

    if not path.is_file():
        return
    required = {"schema_version", "event_id", "event_type", "tick"}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        errors.append(f"无法读取 {path.name}：{exc}")
        return
    for line_number, line in enumerate(lines, start=1):
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{path.name} 第 {line_number} 行不是有效 JSON：{exc}")
            continue
        if not isinstance(item, dict) or not required.issubset(item):
            errors.append(f"{path.name} 第 {line_number} 行缺少统一事件必需字段。")


def write_cross_run_summary(root: Path, config: MatrixConfig, manifest: dict) -> dict:
    """仅汇总通过门禁的完整运行，并按种子与重复编号配对。"""

    summary_dir = root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    run_rows: list[dict] = []
    for record in manifest["runs"]:
        if record.get("status") != "succeeded" or not record.get("completeness_report", {}).get("complete"):
            continue
        run_dir = Path(record["output_dir"])
        for metric in config.metrics:
            baseline, final = read_run_metric(run_dir, metric, config.window_size)
            run_rows.append({
                "run_id": record["run_id"],
                "version": record["version"],
                "seed": record["seed"],
                "repetition": record["repetition"],
                "metric": metric.name,
                "source": metric.source,
                "column": metric.column,
                "baseline": baseline,
                "final": final,
                "delta": final - baseline,
                "output_dir": record["output_dir"],
            })
    _write_csv(summary_dir / "run_metrics.csv", run_rows, [
        "run_id", "version", "seed", "repetition", "metric", "source", "column",
        "baseline", "final", "delta", "output_dir",
    ])

    pair_rows = build_pair_rows(run_rows, config.comparisons)
    _write_csv(summary_dir / "paired_differences.csv", pair_rows, [
        "seed", "repetition", "metric", "left_version", "right_version",
        "left_run_id", "right_run_id", "left_delta", "right_delta", "difference",
    ])
    version_rows = summarize_versions(run_rows)
    _write_csv(summary_dir / "version_summary.csv", version_rows, [
        "version", "metric", "n_runs", "mean_baseline", "mean_final", "mean_delta", "sample_stddev_delta",
    ])
    pair_summary = summarize_pairs(
        pair_rows,
        samples=config.bootstrap_samples,
        seed=config.bootstrap_seed,
    )
    _write_csv(summary_dir / "pair_summary.csv", pair_summary, [
        "metric", "left_version", "right_version", "n_pairs", "mean_difference",
        "median_difference", "sample_stddev_difference", "min_difference", "max_difference",
        "sign_flip_p_two_sided", "bootstrap_ci_low", "bootstrap_ci_high",
    ])
    result = {
        "complete_runs": len({row["run_id"] for row in run_rows}),
        "paired_rows": len(pair_rows),
        "run_metrics_path": str(summary_dir / "run_metrics.csv"),
        "paired_differences_path": str(summary_dir / "paired_differences.csv"),
        "version_summary_path": str(summary_dir / "version_summary.csv"),
        "pair_summary_path": str(summary_dir / "pair_summary.csv"),
    }
    (summary_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def read_run_metric(run_dir: Path, metric: MetricSpec, window_size: int) -> tuple[float, float]:
    """读取单次实验指标，并计算互不重叠的首尾窗口均值。"""

    configure_csv_field_size_limit()
    path = run_dir / metric.source
    if not path.is_file():
        raise ValueError(f"完整运行缺少汇总指标文件：{path}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        missing = [name for name in ("tick", metric.column) if name not in fields]
        if missing:
            raise ValueError(f"{path} 缺少汇总列：{missing}")
        rows = list(reader)
    ordered = sorted(rows, key=lambda row: int(str(row["tick"])))
    if len(ordered) < window_size * 2:
        raise ValueError(f"{path} 的首尾统计窗口会重叠。")
    baseline_values = [_finite_float(row[metric.column], path, metric.column) for row in ordered[:window_size]]
    final_values = [_finite_float(row[metric.column], path, metric.column) for row in ordered[-window_size:]]
    return statistics.fmean(baseline_values), statistics.fmean(final_values)


def _finite_float(value: object, path: Path, column: str) -> float:
    try:
        number = float(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} 的 {column} 包含非数值：{value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{path} 的 {column} 包含非有限数值：{value!r}")
    return number


def build_pair_rows(run_rows: list[dict], comparisons: tuple[ComparisonSpec, ...]) -> list[dict]:
    by_key = {
        (row["version"], row["seed"], row["repetition"], row["metric"]): row
        for row in run_rows
    }
    rows = []
    metric_names = sorted({row["metric"] for row in run_rows})
    blocks = sorted({(row["seed"], row["repetition"]) for row in run_rows})
    for comparison in comparisons:
        for seed, repetition in blocks:
            for metric_name in metric_names:
                left = by_key.get((comparison.left, seed, repetition, metric_name))
                right = by_key.get((comparison.right, seed, repetition, metric_name))
                if left is None or right is None:
                    continue
                rows.append({
                    "seed": seed,
                    "repetition": repetition,
                    "metric": metric_name,
                    "left_version": comparison.left,
                    "right_version": comparison.right,
                    "left_run_id": left["run_id"],
                    "right_run_id": right["run_id"],
                    "left_delta": left["delta"],
                    "right_delta": right["delta"],
                    "difference": left["delta"] - right["delta"],
                })
    return rows


def summarize_versions(run_rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in run_rows:
        groups.setdefault((row["version"], row["metric"]), []).append(row)
    out = []
    for (version, metric), rows in sorted(groups.items()):
        deltas = [float(row["delta"]) for row in rows]
        out.append({
            "version": version,
            "metric": metric,
            "n_runs": len(rows),
            "mean_baseline": statistics.fmean(float(row["baseline"]) for row in rows),
            "mean_final": statistics.fmean(float(row["final"]) for row in rows),
            "mean_delta": statistics.fmean(deltas),
            "sample_stddev_delta": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
        })
    return out


def summarize_pairs(pair_rows: list[dict], *, samples: int, seed: int) -> list[dict]:
    groups: dict[tuple[str, str, str], list[float]] = {}
    for row in pair_rows:
        key = (row["metric"], row["left_version"], row["right_version"])
        groups.setdefault(key, []).append(float(row["difference"]))
    out = []
    for offset, ((metric, left, right), values) in enumerate(sorted(groups.items())):
        ci_low, ci_high = bootstrap_mean_ci(values, samples=samples, seed=seed + offset)
        out.append({
            "metric": metric,
            "left_version": left,
            "right_version": right,
            "n_pairs": len(values),
            "mean_difference": statistics.fmean(values),
            "median_difference": statistics.median(values),
            "sample_stddev_difference": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min_difference": min(values),
            "max_difference": max(values),
            "sign_flip_p_two_sided": exact_sign_flip_p_two_sided(values),
            "bootstrap_ci_low": ci_low,
            "bootstrap_ci_high": ci_high,
        })
    return out


def exact_sign_flip_p_two_sided(values: Sequence[float]) -> float:
    nonzero = [value for value in values if value != 0.0]
    if not nonzero:
        return 1.0
    positives = sum(value > 0.0 for value in nonzero)
    n = len(nonzero)
    tail = min(positives, n - positives)
    probability = sum(math.comb(n, k) for k in range(tail + 1)) / (2 ** n)
    return min(1.0, 2.0 * probability)


def bootstrap_mean_ci(values: Sequence[float], *, samples: int, seed: int) -> tuple[float | None, float | None]:
    if not values or samples == 0:
        return None, None
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        drawn = [values[rng.randrange(len(values))] for _ in values]
        means.append(statistics.fmean(drawn))
    means.sort()
    low_index = max(0, int(0.025 * (len(means) - 1)))
    high_index = min(len(means) - 1, int(0.975 * (len(means) - 1)))
    return means[low_index], means[high_index]


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_or_create_manifest(
    path: Path,
    config: MatrixConfig,
    raw_config: dict,
    config_hash: str,
) -> dict:
    if path.is_file():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("batch_manifest.json 的 schema_version 不匹配。")
        if manifest.get("config_hash") != config_hash:
            raise ValueError("已有 batch_manifest.json 与当前矩阵配置不一致；请使用新的输出目录。")
        expected_ids = [record["run_id"] for record in build_run_records(config)]
        actual_ids = [record.get("run_id") for record in manifest.get("runs", [])]
        if actual_ids != expected_ids:
            raise ValueError("batch_manifest.json 的运行单元与当前矩阵配置不一致。")
        if any(record.get("status") not in RUN_STATUSES for record in manifest["runs"]):
            raise ValueError("batch_manifest.json 包含未支持的运行状态。")
        return manifest
    now = _utc_now()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "config_hash": config_hash,
        "matrix_config": raw_config,
        "created_at": now,
        "updated_at": now,
        "runs": build_run_records(config),
    }
    _write_manifest(path, manifest)
    return manifest


def _write_manifest(path: Path, manifest: dict) -> None:
    manifest["updated_at"] = _utc_now()
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _config_hash(raw_config: dict) -> str:
    canonical = json.dumps(raw_config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timeout_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)
