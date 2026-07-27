from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IAC_DIR = PROJECT_ROOT / "data" / "processed" / "iac_v2"
DEFAULT_METADATA_PATH = IAC_DIR / "llm_pipeline" / "gay_marriage_1q" / "selected_discussion_metadata.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "audits" / "iac_gay_marriage_observable_replay"
QUOTE_FIELDS = [
    "dataset",
    "discussion_id",
    "post_id",
    "quote_index",
    "parent_quote_index",
    "text_offset",
    "text_id",
    "source_discussion_id",
    "source_post_id",
    "source_start",
    "source_end",
    "truncated",
    "altered",
]
STANCE_FIELDS = [
    "dataset",
    "discussion_id",
    "author_id",
    "topic_id",
    "topic_stance_id_1",
    "topic_stance_votes_1",
    "topic_stance_id_2",
    "topic_stance_votes_2",
    "topic_stance_votes_other",
]
POST_FIELDS = [
    "dataset",
    "discussion_id",
    "post_id",
    "author_id",
    "username",
    "creation_date",
    "parent_post_id",
    "parent_missing",
    "text_id",
    "text",
    "discussion_title",
    "discussion_url",
    "topic_id",
    "topic",
]


def _sha256(path: Path) -> str:
    """分块计算文件摘要，便于复核回放输入和输出。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_metadata(path: Path) -> dict[str, Any]:
    """读取同性婚姻讨论的精确元数据。"""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"元数据根节点必须是 JSON 对象：{path}")
    required = {
        "dataset",
        "discussion_id",
        "discussion_title",
        "post_count",
        "author_count",
        "discussion_start",
        "discussion_end",
        "quarter_time",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"元数据缺少精确字段：{missing}")
    return payload


def _read_jsonl_posts(path: Path, dataset: str, discussion_id: str) -> list[dict[str, Any]]:
    """流式筛选指定数据集和讨论，不加载无关帖子。"""

    posts = []
    seen_post_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} 第 {line_number} 行不是有效 JSON：{exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path} 第 {line_number} 行必须是 JSON 对象。")
            if item.get("dataset") != dataset or str(item.get("discussion_id")) != discussion_id:
                continue
            missing = [field for field in POST_FIELDS if field not in item]
            if missing:
                raise ValueError(f"{path} 第 {line_number} 行缺少精确字段：{missing}")
            post_id = item["post_id"]
            if not isinstance(post_id, str) or not post_id.isdecimal():
                raise ValueError(f"{path} 第 {line_number} 行 post_id 必须是十进制数字字符串。")
            if post_id in seen_post_ids:
                raise ValueError(f"讨论中存在重复 post_id：{post_id}")
            seen_post_ids.add(post_id)
            _parse_creation_date(item["creation_date"], path, line_number)
            posts.append({field: item[field] for field in POST_FIELDS})
    return sorted(posts, key=lambda item: (_parse_creation_date(item["creation_date"], path, 0), int(item["post_id"])))


def _parse_creation_date(value: object, path: Path, line_number: int) -> datetime:
    """严格解析当前 IAC 处理文件中的时间字段。"""

    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} 第 {line_number} 行 creation_date 必须是非空字符串。")
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{path} 第 {line_number} 行 creation_date 无法解析：{value!r}") from exc


def _read_csv_matches(
    path: Path,
    expected_fields: list[str],
    dataset: str,
    discussion_id: str,
) -> list[dict[str, str]]:
    """按精确表头和键值读取讨论关联行。"""

    matches = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_fields:
            raise ValueError(f"{path} 表头不匹配；实际={reader.fieldnames}，预期={expected_fields}")
        for row in reader:
            if row["dataset"] == dataset and row["discussion_id"] == discussion_id:
                matches.append(dict(row))
    return matches


def build_replay(
    *,
    posts_path: Path,
    quotes_path: Path,
    stances_path: Path,
    metadata_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """构建只包含原始数据可观测字段的讨论回放。"""

    metadata = _load_metadata(metadata_path)
    dataset = str(metadata["dataset"])
    discussion_id = str(metadata["discussion_id"])
    posts = _read_jsonl_posts(posts_path, dataset, discussion_id)
    quotes = _read_csv_matches(quotes_path, QUOTE_FIELDS, dataset, discussion_id)
    stances = _read_csv_matches(stances_path, STANCE_FIELDS, dataset, discussion_id)

    post_ids = {post["post_id"] for post in posts}
    quotes_by_post: dict[str, list[dict[str, str]]] = defaultdict(list)
    unattached_quote_count = 0
    unresolved_quote_source_count = 0
    for quote in quotes:
        quotes_by_post[quote["post_id"]].append(quote)
        if quote["post_id"] not in post_ids:
            unattached_quote_count += 1
        source_post_id = quote["source_post_id"]
        if (
            quote["source_discussion_id"] == discussion_id
            and source_post_id
            and source_post_id not in post_ids
        ):
            unresolved_quote_source_count += 1

    stances_by_author: dict[str, list[dict[str, str]]] = defaultdict(list)
    for stance in stances:
        stances_by_author[stance["author_id"]].append(stance)

    replay = []
    for sequence, post in enumerate(posts, start=1):
        replay.append({
            "sequence": sequence,
            **post,
            "quotes": sorted(
                quotes_by_post.get(post["post_id"], []),
                key=lambda item: (int(item["quote_index"]), item["parent_quote_index"]),
            ),
            "mturk_author_stances": stances_by_author.get(post["author_id"], []),
        })

    author_ids = {post["author_id"] for post in posts}
    unexpected_missing_parents = 0
    declared_missing_parents = 0
    for post in posts:
        parent_post_id = post["parent_post_id"]
        if parent_post_id is None:
            continue
        if not isinstance(parent_post_id, str):
            raise ValueError(f"post_id={post['post_id']} 的 parent_post_id 必须是字符串或 null。")
        if parent_post_id not in post_ids:
            if post["parent_missing"] == "1":
                declared_missing_parents += 1
            else:
                unexpected_missing_parents += 1

    actual_post_count = len(posts)
    actual_author_count = len(author_ids)
    expected_post_count = int(metadata["post_count"])
    expected_author_count = int(metadata["author_count"])
    completeness = {
        "schema_version": 1,
        "dataset": dataset,
        "discussion_id": discussion_id,
        "expected_post_count": expected_post_count,
        "actual_post_count": actual_post_count,
        "expected_author_count": expected_author_count,
        "actual_author_count": actual_author_count,
        "quote_row_count": len(quotes),
        "posts_with_quotes": len({quote["post_id"] for quote in quotes}),
        "unattached_quote_count": unattached_quote_count,
        "unresolved_same_discussion_quote_source_count": unresolved_quote_source_count,
        "declared_missing_parent_count": declared_missing_parents,
        "unexpected_missing_parent_count": unexpected_missing_parents,
        "mturk_author_stance_row_count": len(stances),
        "authors_with_mturk_stance": len(stances_by_author),
        "mturk_author_stance_author_coverage": len(stances_by_author) / actual_author_count if actual_author_count else 0.0,
        "complete_definition": (
            "仅检查可观测帖子与作者数量、引用目标连接和未声明父帖缺失；"
            "不表示原平台关注、推荐、曝光或互动数据完整。"
        ),
        "complete": (
            actual_post_count == expected_post_count
            and actual_author_count == expected_author_count
            and unattached_quote_count == 0
            and unexpected_missing_parents == 0
        ),
    }
    manifest = {
        "schema_version": 1,
        "replay_scope": "IAC v2 FourForums 可观测讨论回放",
        "dataset": dataset,
        "discussion_id": discussion_id,
        "discussion_title": metadata["discussion_title"],
        "discussion_start": metadata["discussion_start"],
        "discussion_end": metadata["discussion_end"],
        "quarter_time": metadata["quarter_time"],
        "ordering": ["creation_date", "post_id 的整数值"],
        "observable_fields": {
            "post": POST_FIELDS,
            "quote": QUOTE_FIELDS,
            "mturk_author_stance": STANCE_FIELDS,
        },
        "unobservable_platform_fields": [
            "原平台关注关系",
            "原平台推荐算法、特征、分数与排序结果",
            "原平台曝光、浏览和点击日志",
            "原平台点赞、点踩和转发事件",
        ],
        "source_files": {
            "posts": _source_manifest(posts_path),
            "quotes": _source_manifest(quotes_path),
            "mturk_author_stances": _source_manifest(stances_path),
            "discussion_metadata": _source_manifest(metadata_path),
        },
    }
    return replay, manifest, completeness


def _source_manifest(path: Path) -> dict[str, Any]:
    """记录输入文件的绝对路径、大小和摘要。"""

    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _render_report(manifest: dict[str, Any], completeness: dict[str, Any]) -> str:
    """生成回放范围、完整性与保真度边界报告。"""

    lines = [
        "# IAC v2 FourForums 可观测讨论回放报告",
        "",
        "## 回放范围",
        "",
        f"- 数据集：`{manifest['dataset']}`",
        f"- 讨论：`{manifest['discussion_id']}` / {manifest['discussion_title']}",
        f"- 原始时间范围：`{manifest['discussion_start']}` 至 `{manifest['discussion_end']}`",
        f"- 当前仿真初始化分界时间：`{manifest['quarter_time']}`",
        "- 排序：先按 `creation_date`，同一时间再按 `post_id` 的整数值。",
        "",
        "## 完整性",
        "",
        "| 检查项 | 数量或结果 |",
        "|---|---|",
        f"| 元数据帖子数 / 回放帖子数 | {completeness['expected_post_count']} / {completeness['actual_post_count']} |",
        f"| 元数据作者数 / 回放作者数 | {completeness['expected_author_count']} / {completeness['actual_author_count']} |",
        f"| 引用记录数 | {completeness['quote_row_count']} |",
        f"| 含引用的帖子数 | {completeness['posts_with_quotes']} |",
        f"| 无法连接到回放帖子的引用记录数 | {completeness['unattached_quote_count']} |",
        f"| 同讨论内无法连接来源帖的引用记录数 | {completeness['unresolved_same_discussion_quote_source_count']} |",
        f"| 原数据已声明缺失的父帖数 | {completeness['declared_missing_parent_count']} |",
        f"| 原数据未声明但缺失的父帖数 | {completeness['unexpected_missing_parent_count']} |",
        f"| MTurk 作者立场记录数 | {completeness['mturk_author_stance_row_count']} |",
        f"| 有 MTurk 立场的作者覆盖率 | {completeness['mturk_author_stance_author_coverage']:.6f} |",
        f"| 可观测回放完整性通过 | {'是' if completeness['complete'] else '否'} |",
        "",
        "## 保真度边界",
        "",
        "回放直接保留 IAC 处理数据中的发帖时间、作者、父帖、正文、引用来源和现有 MTurk 作者立场记录。",
        "本讨论在 `mturk_author_stance.csv` 中没有记录，因此每条回放记录的 `mturk_author_stances` 均为空数组。",
        "以下项目没有原始数据证据，不能称为原平台还原：",
        "",
    ]
    lines.extend(f"- {item}" for item in manifest["unobservable_platform_fields"])
    lines.extend(
        [
            "",
            "项目中的关注图、信任、投放账号和信息流日志属于仿真实现或实验处理，必须与本回放分开报告。",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    replay: list[dict[str, Any]],
    manifest: dict[str, Any],
    completeness: dict[str, Any],
    output_dir: Path,
) -> list[Path]:
    """写出回放 JSONL、清单、完整性 JSON 和中文报告。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    replay_path = output_dir / "iac_observable_replay.jsonl"
    completeness_path = output_dir / "iac_observable_replay_completeness.json"
    manifest_path = output_dir / "iac_observable_replay_manifest.json"
    report_path = output_dir / "iac_observable_replay_report.md"
    with replay_path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in replay:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    completeness_path.write_text(
        json.dumps(completeness, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest["output"] = {
        "replay_path": str(replay_path.resolve()),
        "replay_size_bytes": replay_path.stat().st_size,
        "replay_sha256": _sha256(replay_path),
        "record_count": len(replay),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(_render_report(manifest, completeness), encoding="utf-8")
    return [replay_path, manifest_path, completeness_path, report_path]


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 IAC v2 FourForums 可观测讨论回放。")
    parser.add_argument("--posts", type=Path, default=IAC_DIR / "posts.jsonl")
    parser.add_argument("--quotes", type=Path, default=IAC_DIR / "quote.csv")
    parser.add_argument("--mturk-author-stances", type=Path, default=IAC_DIR / "mturk_author_stance.csv")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    replay, manifest, completeness = build_replay(
        posts_path=args.posts,
        quotes_path=args.quotes,
        stances_path=args.mturk_author_stances,
        metadata_path=args.metadata,
    )
    for path in write_outputs(replay, manifest, completeness, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
