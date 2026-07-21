from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "iac_v2" / "fourforums_no_parse_2016_05_18.sql.gz"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "audits"

# 每项能力只绑定已经核实的原始表和列，不按名称相似度推断。
SOURCE_CAPABILITIES = [
    {
        "id": "authors",
        "name": "发帖者身份",
        "evidence": {"author": ["author_id", "username"]},
    },
    {
        "id": "discussions",
        "name": "讨论页面与发起者",
        "evidence": {"discussion": ["discussion_id", "url", "title", "initiating_author_id"]},
    },
    {
        "id": "post_timeline",
        "name": "发帖时间线",
        "evidence": {"post": ["discussion_id", "post_id", "author_id", "creation_date", "text_id"]},
    },
    {
        "id": "reply_tree",
        "name": "父帖回复关系",
        "evidence": {"post": ["discussion_id", "post_id", "parent_post_id", "parent_missing"]},
    },
    {
        "id": "quotes",
        "name": "引用及引用来源",
        "evidence": {
            "quote": [
                "discussion_id", "post_id", "quote_index", "parent_quote_index", "text_id",
                "source_discussion_id", "source_post_id", "source_start", "source_end", "truncated", "altered",
            ]
        },
    },
    {
        "id": "topics",
        "name": "讨论话题与话题立场定义",
        "evidence": {
            "discussion_topic": ["discussion_id", "topic_id"],
            "topic": ["topic_id", "topic"],
            "topic_stance": ["topic_id", "topic_stance_id", "stance"],
        },
    },
    {
        "id": "annotated_author_stance",
        "name": "人工标注的作者立场",
        "evidence": {
            "mturk_author_stance": [
                "discussion_id", "author_id", "topic_id", "topic_stance_id_1", "topic_stance_votes_1",
                "topic_stance_id_2", "topic_stance_votes_2", "topic_stance_votes_other",
            ]
        },
    },
]

SOURCE_MISSING = [
    {"id": "follow_graph", "name": "原平台关注关系"},
    {"id": "recommendation", "name": "原平台推荐算法、特征、分数与排序结果"},
    {"id": "exposure", "name": "原平台曝光、浏览和点击日志"},
    {"id": "engagement", "name": "原平台点赞、点踩和转发事件"},
]


def _sql_unescape(char: str) -> str:
    """还原 mysqldump 字符串中的常用反斜杠转义。"""

    return {"0": "\0", "n": "\n", "r": "\r", "t": "\t", "b": "\b", "Z": "\x1a"}.get(char, char)


def parse_insert_values(line: str) -> list[list[Any]]:
    """解析单行 mysqldump 扩展 INSERT，仅用于精确读取元数据。"""

    marker = " VALUES "
    index = line.find(marker)
    if index < 0:
        return []
    text = line[index + len(marker):].rstrip().rstrip(";")
    rows: list[list[Any]] = []
    row: list[Any] | None = None
    token: list[str] = []
    quoted = in_string = escaped = False

    def finish() -> None:
        nonlocal token, quoted
        if row is None:
            return
        value = "".join(token)
        row.append(value if quoted else (None if value.strip().upper() == "NULL" else value.strip()))
        token = []
        quoted = False

    for char in text:
        if in_string:
            if escaped:
                token.append(_sql_unescape(char))
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == "'":
                in_string = False
            else:
                token.append(char)
            continue
        if char == "'":
            quoted = in_string = True
        elif char == "(" and row is None:
            row, token, quoted = [], [], False
        elif char == "," and row is not None:
            finish()
        elif char == ")" and row is not None:
            finish()
            rows.append(row)
            row = None
        elif row is not None:
            token.append(char)
    return rows


def parse_sql_dump(path: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """流式读取完整建表结构和 dataset_metadata，不展开 SQL 文件。"""

    tables: dict[str, dict[str, Any]] = {}
    metadata: list[dict[str, Any]] = []
    table_name = ""
    columns: list[dict[str, str]] = []
    constraints: list[str] = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("CREATE TABLE `"):
                table_name = line.split("`", 2)[1]
                columns, constraints = [], []
                continue
            if table_name:
                stripped = line.strip().rstrip(",")
                if stripped.startswith("`"):
                    parts = stripped.split("`", 2)
                    columns.append({"name": parts[1], "definition": parts[2].strip()})
                elif line.startswith(") ENGINE="):
                    tables[table_name] = {"columns": columns, "constraints": constraints, "engine": line.strip()}
                    table_name = ""
                elif stripped:
                    constraints.append(stripped)
                continue
            if line.startswith("INSERT INTO `dataset_metadata`"):
                for row in parse_insert_values(line):
                    if len(row) == 3:
                        metadata.append({"row_id": row[0], "metadata_field": row[1], "metadata_value": row[2]})
    return tables, metadata


def _sha256(path: Path) -> str:
    """分块计算源文件摘要，保证审计输入可复核。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_exact_columns(tables: dict[str, dict[str, Any]], evidence: dict[str, list[str]]) -> None:
    """要求精确表列存在；结构变化时直接失败，不进行替代匹配。"""

    for table, required in evidence.items():
        if table not in tables:
            raise ValueError(f"SQL 中缺少精确表名: {table}")
        actual = {item["name"] for item in tables[table]["columns"]}
        missing = [column for column in required if column not in actual]
        if missing:
            raise ValueError(f"SQL 表 {table} 缺少精确列名: {missing}")


def assess_source(tables: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """按已核实的数据字典生成原始数据能力清单。"""

    result = []
    for item in SOURCE_CAPABILITIES:
        _require_exact_columns(tables, item["evidence"])
        result.append({**item, "status": "restored", "basis": "原始 SQL 直接提供精确表列"})
    all_tables = sorted(tables)
    for item in SOURCE_MISSING:
        result.append(
            {
                **item,
                "status": "missing",
                "evidence": {},
                "basis": "完整 SQL 建表结构未提供该平台数据；未用其他字段替代推断",
                "audited_table_names": all_tables,
            }
        )
    return result


def _exact_line(path: Path, text: str) -> int | None:
    """查找已经核实的精确代码标识。"""

    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if text in line:
            return number
    return None


def assess_project(project_root: Path) -> list[dict[str, Any]]:
    """核对当前场景和社交平台对原始平台设置的实现边界。"""

    scene = project_root / "backend" / "scenarios" / "iac_gay_marriage.py"
    platform = project_root / "backend" / "social_sys" / "platform.py"
    config = project_root / "backend" / "persona" / "config.py"
    checks = [
        (scene, "def _follow_edges"),
        (scene, "def _online_trust"),
        (scene, "def _offline_trust"),
        (platform, "def _apply_recommendation_placeholder"),
        (config, "social_recommendation_enabled: bool = False"),
    ]
    lines: dict[str, int] = {}
    for path, marker in checks:
        line = _exact_line(path, marker)
        if line is None:
            raise ValueError(f"未找到精确代码标识: {path}:{marker}")
        lines[f"{path.relative_to(project_root).as_posix()}::{marker}"] = line

    exposure_line = _exact_line(platform, "exposure_events")
    return [
        {
            "id": "follow_graph",
            "status": "derived",
            "basis": "场景函数 _follow_edges 生成；FourForums SQL 未提供原平台关注关系",
            "code_line": lines["backend/scenarios/iac_gay_marriage.py::def _follow_edges"],
        },
        {
            "id": "online_trust",
            "status": "derived",
            "basis": "场景函数 _online_trust 生成；不是原平台字段",
            "code_line": lines["backend/scenarios/iac_gay_marriage.py::def _online_trust"],
        },
        {
            "id": "offline_trust",
            "status": "derived",
            "basis": "场景函数 _offline_trust 生成；不是原平台字段",
            "code_line": lines["backend/scenarios/iac_gay_marriage.py::def _offline_trust"],
        },
        {
            "id": "recommendation",
            "status": "missing",
            "basis": "当前精确入口为 _apply_recommendation_placeholder，配置默认值为 False",
            "code_line": lines["backend/social_sys/platform.py::def _apply_recommendation_placeholder"],
        },
        {
            "id": "exposure_instrumentation",
            "status": "derived" if exposure_line is not None else "missing",
            "basis": "项目观测日志不等于原平台曝光数据",
            "code_line": exposure_line,
        },
    ]


def build_audit(source: Path, project_root: Path) -> dict[str, Any]:
    """生成确定性的机器可读审计对象。"""

    if not source.is_file():
        raise FileNotFoundError(f"IAC SQL 压缩包不存在: {source}")
    tables, metadata = parse_sql_dump(source)
    return {
        "schema_version": 1,
        "audit_scope": "IAC v2 FourForums 原始平台保真度",
        "source": {
            "path": str(source.resolve()),
            "size_bytes": source.stat().st_size,
            "sha256": _sha256(source),
            "table_count": len(tables),
            "dataset_metadata": metadata,
        },
        "source_schema": tables,
        "source_capabilities": assess_source(tables),
        "project_fidelity": assess_project(project_root),
        "status_contract": {
            "restored": "原始 SQL 直接提供精确表列",
            "derived": "项目代码生成或测量，不能称为原平台还原",
            "missing": "原始 SQL 或当前项目未提供，不以其他字段替代",
        },
    }


def render_report(audit: dict[str, Any]) -> str:
    """把机器结果渲染为中文审计报告。"""

    source = audit["source"]
    lines = [
        "# IAC v2 FourForums 平台保真度审计",
        "",
        "## 审计输入",
        "",
        f"- SQL 压缩包：`{source['path']}`",
        f"- 文件大小：`{source['size_bytes']}` 字节",
        f"- SHA-256：`{source['sha256']}`",
        f"- 完整建表数量：`{source['table_count']}`",
        "",
        "## 结论",
        "",
        "FourForums SQL 可以直接还原作者、讨论、帖子时间线、父帖、引用、话题及人工立场标注。",
        "SQL 不提供原平台关注关系、推荐排序、曝光日志、点赞、点踩或转发事件。",
        "当前场景中的关注关系、线上信任和线下信任均由项目代码生成，必须标记为推导设置。",
        "",
        "## 原始数据能力",
        "",
        "| 状态 | 能力 | 精确证据 |",
        "|---|---|---|",
    ]
    for item in audit["source_capabilities"]:
        evidence = "; ".join(f"{table}({', '.join(columns)})" for table, columns in item["evidence"].items()) or "无"
        lines.append(f"| `{item['status']}` | {item['name']} | `{evidence}` |")
    lines.extend(["", "## 当前项目保真度", "", "| 状态 | 项目项 | 依据 |", "|---|---|---|"])
    for item in audit["project_fidelity"]:
        line = item.get("code_line")
        line_text = f"；代码行 {line}" if line is not None else ""
        lines.append(f"| `{item['status']}` | `{item['id']}` | {item['basis']}{line_text} |")
    lines.extend(["", "## 原始 SQL 表", ""])
    for table, detail in audit["source_schema"].items():
        columns = ", ".join(column["name"] for column in detail["columns"])
        lines.append(f"- `{table}`：`{columns}`")
    lines.extend(
        [
            "",
            "## 后续输入要求",
            "",
            "若要声称还原原平台关注或推荐设置，必须补充原平台数据库表、平台代码、技术文档或可复核抓包。",
            "在补充证据前，后续实验应分别命名为“项目生成关注图”和“项目实现推荐策略”，不能命名为原平台还原。",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(audit: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    """写出固定文件名的 JSON 与 Markdown。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "iac_fourforums_platform_fidelity.json"
    report_path = output_dir / "iac_fourforums_platform_fidelity.md"
    json_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(render_report(audit), encoding="utf-8")
    return json_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="审计 IAC v2 FourForums 原始平台设置与当前项目实现边界。")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="fourforums no_parse SQL 压缩包路径。")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT, help="项目根目录。")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="审计产物目录。")
    args = parser.parse_args()
    json_path, report_path = write_outputs(build_audit(args.source, args.project_root), args.output_dir)
    print(json_path)
    print(report_path)


if __name__ == "__main__":
    main()
