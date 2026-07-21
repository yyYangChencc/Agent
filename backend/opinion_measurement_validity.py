from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from analyze_history import (
    read_history_rows,
    read_posthoc_voting_rows,
    read_voting_agent_metrics,
    resolve_history_run_dir,
)
from persona.history_csv import configure_csv_field_size_limit
from persona.opinion.scale import (
    VOTING_ROLE_OPPOSE,
    VOTING_ROLE_SUPPORT,
    VOTING_ROLE_UNKNOWN,
    VOTING_STANCE_INVALID,
)


configure_csv_field_size_limit()


FLAN_LABELS = ("-2", "-1", "0", "1", "2")
VOTING_LABELS = (
    VOTING_ROLE_SUPPORT,
    VOTING_ROLE_OPPOSE,
    VOTING_ROLE_UNKNOWN,
)
FLAN_TEMPLATE_NAME = "human_flan_annotations.csv"
VOTING_TEMPLATE_NAME = "human_voting_annotations.csv"


@dataclass(frozen=True)
class ValidityMetrics:
    method: str
    total_rows: int
    annotated_rows: int
    unannotated_rows: int
    valid_predictions: int
    invalid_predictions: int
    accuracy: float
    macro_f1: float
    labels: tuple[str, ...]
    prediction_labels: tuple[str, ...]
    confusion: dict[str, dict[str, int]]
    per_label: dict[str, dict[str, float | int]]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出人工标注模板并评估观念测量效度。")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="从单次运行目录导出人工标注模板。")
    export_parser.add_argument("history_dir", help="单次运行目录或 history 根目录。")
    export_parser.add_argument("output_dir", help="人工标注模板输出目录。")

    evaluate_parser = subparsers.add_parser("evaluate", help="评估已完成人工标注的模板。")
    evaluate_parser.add_argument("annotation_dir", help="包含两份人工标注 CSV 的目录。")
    evaluate_parser.add_argument("--output-dir", help="评估结果目录；默认写入标注目录。")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.command == "export":
        paths = export_annotation_templates(args.history_dir, args.output_dir)
    else:
        paths = evaluate_annotation_files(args.annotation_dir, output_dir=args.output_dir)
    for path in paths:
        print(f"已生成：{path}")
    return 0


def export_annotation_templates(
    history_dir: str | Path,
    output_dir: str | Path,
) -> list[Path]:
    """分别导出 FLAN 五级评分和 llm_voting 离散立场标注模板。"""

    run_dir = resolve_history_run_dir(history_dir)
    out_dir = Path(output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    flan_rows = build_flan_annotation_rows(read_history_rows(run_dir))
    voting_source = read_posthoc_voting_rows(run_dir) or read_history_rows(run_dir)
    voting_rows = build_voting_annotation_rows(voting_source)

    flan_path = out_dir / FLAN_TEMPLATE_NAME
    voting_path = out_dir / VOTING_TEMPLATE_NAME
    write_csv_rows(
        flan_path,
        [
            "sample_id",
            "agent_id",
            "history_tick",
            "opinion_assessment_topic",
            "current_honest_belief",
            "opinion_flan_rating",
            "opinion_flan_model",
            "human_flan_rating",
        ],
        flan_rows,
    )
    write_csv_rows(
        voting_path,
        [
            "sample_id",
            "agent_id",
            "opinion_voting_tick",
            "opinion_voting_window_start_tick",
            "opinion_voting_window_end_tick",
            "opinion_assessment_topic",
            "opinion_voting_speech_history",
            "opinion_voting_stance",
            "opinion_voting_stance_valid",
            "human_voting_stance",
        ],
        voting_rows,
    )
    return [flan_path, voting_path]


def build_flan_annotation_rows(rows_by_agent: dict[str, list[dict]]) -> list[dict]:
    """按实际评测文本去重，避免历史记录重复携带上一次评测结果。"""

    output: list[dict] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for agent_id, rows in sorted(rows_by_agent.items()):
        for row in rows:
            belief = str(row.get("current_honest_belief") or "").strip()
            rating = str(row.get("opinion_flan_rating") or "").strip()
            if not belief or not rating:
                continue
            topic = str(row.get("opinion_assessment_topic") or "")
            model = str(row.get("opinion_flan_model") or "")
            key = (agent_id, topic, belief, rating, model)
            if key in seen:
                continue
            seen.add(key)
            output.append({
                "sample_id": f"flan_{len(output) + 1:06d}",
                "agent_id": agent_id,
                "history_tick": row.get("tick", ""),
                "opinion_assessment_topic": topic,
                "current_honest_belief": belief,
                "opinion_flan_rating": rating,
                "opinion_flan_model": model,
                "human_flan_rating": "",
            })
    return output


def build_voting_annotation_rows(rows_by_agent: dict[str, list[dict]]) -> list[dict]:
    """把每个智能体的每个投票窗口导出为一条人工标注样本。"""

    metrics = {
        (metric.agent_id, metric.tick): metric
        for metric in read_voting_agent_metrics(rows_by_agent)
    }
    output: list[dict] = []
    for agent_id, rows in sorted(rows_by_agent.items()):
        voting_rows = [row for row in rows if row.get("opinion_voting_tick") not in (None, "")]
        for row in sorted(voting_rows, key=lambda item: int(float(item["opinion_voting_tick"]))):
            tick = int(float(row["opinion_voting_tick"]))
            metric = metrics[(agent_id, tick)]
            # 零请求窗口没有线上发言证据，不生成无法判定的人工标注任务。
            if metric.requested_votes == 0:
                continue
            output.append({
                "sample_id": f"voting_{len(output) + 1:06d}",
                "agent_id": agent_id,
                "opinion_voting_tick": tick,
                "opinion_voting_window_start_tick": row.get("opinion_voting_window_start_tick", ""),
                "opinion_voting_window_end_tick": row.get("opinion_voting_window_end_tick", ""),
                "opinion_assessment_topic": row.get("opinion_assessment_topic", ""),
                "opinion_voting_speech_history": row.get("opinion_voting_speech_history", ""),
                "opinion_voting_stance": metric.stance,
                "opinion_voting_stance_valid": bool(metric.stance_valid),
                "human_voting_stance": "",
            })
    return output


def evaluate_annotation_files(
    annotation_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> list[Path]:
    """读取已标注模板，生成结构化指标、混淆矩阵和 Markdown 报告。"""

    source_dir = Path(annotation_dir).expanduser()
    out_dir = Path(output_dir).expanduser() if output_dir is not None else source_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    flan_rows = read_csv_rows(source_dir / FLAN_TEMPLATE_NAME)
    voting_rows = read_csv_rows(source_dir / VOTING_TEMPLATE_NAME)

    flan_metrics = compute_validity_metrics(
        method="FLAN",
        rows=flan_rows,
        truth_field="human_flan_rating",
        prediction_field="opinion_flan_rating",
        labels=FLAN_LABELS,
        prediction_invalid_label=VOTING_STANCE_INVALID,
    )
    voting_metrics = compute_validity_metrics(
        method="llm_voting",
        rows=voting_rows,
        truth_field="human_voting_stance",
        prediction_field="opinion_voting_stance",
        labels=VOTING_LABELS,
        prediction_invalid_label=VOTING_STANCE_INVALID,
    )

    flan_matrix_path = out_dir / "flan_confusion_matrix.csv"
    voting_matrix_path = out_dir / "voting_confusion_matrix.csv"
    metrics_path = out_dir / "measurement_validity_metrics.json"
    report_path = out_dir / "measurement_validity_report.md"
    write_confusion_matrix(flan_matrix_path, flan_metrics)
    write_confusion_matrix(voting_matrix_path, voting_metrics)
    metrics_path.write_text(
        json.dumps(
            {
                "FLAN": metrics_to_dict(flan_metrics),
                "llm_voting": metrics_to_dict(voting_metrics),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    report_path.write_text(
        build_validity_report(flan_metrics, voting_metrics),
        encoding="utf-8",
    )
    return [flan_matrix_path, voting_matrix_path, metrics_path, report_path]


def compute_validity_metrics(
    *,
    method: str,
    rows: list[dict[str, str]],
    truth_field: str,
    prediction_field: str,
    labels: tuple[str, ...],
    prediction_invalid_label: str,
) -> ValidityMetrics:
    """仅使用精确类别值计算分类指标，不改写标签。"""

    prediction_labels = labels + (prediction_invalid_label,)
    confusion = {
        truth: {prediction: 0 for prediction in prediction_labels}
        for truth in labels
    }
    annotated_rows = 0
    valid_predictions = 0
    invalid_predictions = 0
    for row_number, row in enumerate(rows, start=2):
        truth = str(row.get(truth_field) or "").strip()
        if not truth:
            continue
        if truth not in labels:
            raise ValueError(
                f"{method} 标注文件第 {row_number} 行的 {truth_field} 必须是 {list(labels)} 之一：{truth!r}"
            )
        prediction = str(row.get(prediction_field) or "").strip()
        if prediction in labels:
            valid_predictions += 1
        elif prediction == prediction_invalid_label or prediction == "":
            prediction = prediction_invalid_label
            invalid_predictions += 1
        else:
            raise ValueError(
                f"{method} 标注文件第 {row_number} 行的 {prediction_field} 不是精确类别：{prediction!r}"
            )
        confusion[truth][prediction] += 1
        annotated_rows += 1

    per_label: dict[str, dict[str, float | int]] = {}
    correct = 0
    for label in labels:
        true_positive = confusion[label][label]
        false_positive = sum(confusion[truth][label] for truth in labels if truth != label)
        false_negative = sum(
            count for prediction, count in confusion[label].items() if prediction != label
        )
        support = sum(confusion[label].values())
        precision = safe_divide(true_positive, true_positive + false_positive)
        recall = safe_divide(true_positive, true_positive + false_negative)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        correct += true_positive
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    return ValidityMetrics(
        method=method,
        total_rows=len(rows),
        annotated_rows=annotated_rows,
        unannotated_rows=len(rows) - annotated_rows,
        valid_predictions=valid_predictions,
        invalid_predictions=invalid_predictions,
        accuracy=safe_divide(correct, annotated_rows),
        macro_f1=safe_divide(sum(float(item["f1"]) for item in per_label.values()), len(labels)),
        labels=labels,
        prediction_labels=prediction_labels,
        confusion=confusion,
        per_label=per_label,
    )


def metrics_to_dict(metrics: ValidityMetrics) -> dict:
    return {
        "method": metrics.method,
        "total_rows": metrics.total_rows,
        "annotated_rows": metrics.annotated_rows,
        "unannotated_rows": metrics.unannotated_rows,
        "annotation_completion_ratio": safe_divide(metrics.annotated_rows, metrics.total_rows),
        "valid_predictions": metrics.valid_predictions,
        "invalid_predictions": metrics.invalid_predictions,
        "valid_prediction_ratio": safe_divide(metrics.valid_predictions, metrics.annotated_rows),
        "invalid_prediction_ratio": safe_divide(metrics.invalid_predictions, metrics.annotated_rows),
        "accuracy": metrics.accuracy,
        "macro_f1": metrics.macro_f1,
        "labels": list(metrics.labels),
        "prediction_labels": list(metrics.prediction_labels),
        "confusion_matrix": metrics.confusion,
        "per_label": metrics.per_label,
    }


def build_validity_report(*metrics_items: ValidityMetrics) -> str:
    lines = ["# 观念测量效度报告", ""]
    for metrics in metrics_items:
        payload = metrics_to_dict(metrics)
        lines.extend([
            f"## {metrics.method}",
            "",
            f"- 总样本数：{metrics.total_rows}",
            f"- 已完成人工标注：{metrics.annotated_rows}",
            f"- 未完成人工标注：{metrics.unannotated_rows}",
            f"- 标注完成比例：{payload['annotation_completion_ratio']:.6f}",
            f"- 自动结果有效比例：{payload['valid_prediction_ratio']:.6f}",
            f"- 自动结果无效比例：{payload['invalid_prediction_ratio']:.6f}",
            f"- 准确率：{metrics.accuracy:.6f}",
            f"- 宏平均 F1：{metrics.macro_f1:.6f}",
            "",
            "| 类别 | Precision | Recall | F1 | Support |",
            "| --- | ---: | ---: | ---: | ---: |",
        ])
        for label in metrics.labels:
            item = metrics.per_label[label]
            lines.append(
                f"| {label} | {float(item['precision']):.6f} | {float(item['recall']):.6f} | "
                f"{float(item['f1']):.6f} | {int(item['support'])} |"
            )
        lines.extend([
            "",
            "### 混淆矩阵",
            "",
            "| 人工标签 \\ 自动结果 | " + " | ".join(metrics.prediction_labels) + " |",
            "| --- | " + " | ".join("---:" for _ in metrics.prediction_labels) + " |",
        ])
        for truth in metrics.labels:
            counts = [str(metrics.confusion[truth][prediction]) for prediction in metrics.prediction_labels]
            lines.append(f"| {truth} | " + " | ".join(counts) + " |")
        lines.append("")
    return "\n".join(lines)


def write_confusion_matrix(path: Path, metrics: ValidityMetrics) -> None:
    rows = []
    for truth in metrics.labels:
        row = {"human_label": truth}
        row.update(metrics.confusion[truth])
        rows.append(row)
    write_csv_rows(path, ["human_label", *metrics.prediction_labels], rows)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"人工标注文件不存在：{path}")
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def write_csv_rows(path: Path, fieldnames: list[str], rows: Iterable[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(1)
