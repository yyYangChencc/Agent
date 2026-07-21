from __future__ import annotations

import csv
import sys


def configure_csv_field_size_limit() -> int:
    """将 CSV 单字段上限设置为当前解释器可接受的最大值。"""

    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return limit
        except OverflowError:
            # 兼容 csv 底层整数范围小于 sys.maxsize 的解释器。
            limit //= 10
            if limit <= 0:
                raise
