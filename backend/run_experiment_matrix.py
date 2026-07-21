from __future__ import annotations

import argparse
from pathlib import Path

from experiment_matrix import run_matrix


def parse_args() -> argparse.Namespace:
    """解析批量实验入口参数。"""

    parser = argparse.ArgumentParser(description="按结构化矩阵批量运行实验并生成配对汇总。")
    parser.add_argument("matrix", help="矩阵 JSON 文件。")
    parser.add_argument("--output-dir", required=True, help="批量运行根目录。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_matrix(args.matrix, args.output_dir)
    complete = sum(record["status"] == "succeeded" for record in manifest["runs"])
    total = len(manifest["runs"])
    print(f"matrix_output_dir={Path(args.output_dir).resolve()}")
    print(f"matrix_complete_runs={complete}/{total}")


if __name__ == "__main__":
    main()
