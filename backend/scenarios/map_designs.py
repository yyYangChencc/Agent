from __future__ import annotations

from copy import deepcopy
from typing import Any


MapDesignData = dict[str, Any]


def _horizontal(row: int, col_start: int, col_end: int) -> list[list[int]]:
    return [[row, col] for col in range(col_start, col_end + 1)]


def _vertical(col: int, row_start: int, row_end: int) -> list[list[int]]:
    return [[row, col] for row in range(row_start, row_end + 1)]


TERRAIN = [
    {"id": "grass_base", "name": "草地底色", "kind": "grass", "bounds": [0, 0, 24, 24], "color": "#203a2f", "alpha": 1.0},
    {"id": "home_yards", "name": "住宅庭院", "kind": "yard", "bounds": [0, 0, 9, 12], "color": "#315a3b", "alpha": 0.45},
    {"id": "market_floor", "name": "商业铺装", "kind": "plaza", "bounds": [0, 15, 8, 24], "color": "#4a3a2a", "alpha": 0.45},
    {"id": "office_ground", "name": "办公地块", "kind": "office_ground", "bounds": [16, 0, 24, 10], "color": "#293d52", "alpha": 0.4},
    {"id": "park_ground", "name": "公园地块", "kind": "park", "bounds": [16, 16, 24, 24], "color": "#1f5a43", "alpha": 0.5},
]


REGIONS = [
    {"id": "residential_area", "name": "住宅区", "kind": "residential", "bounds": [0, 0, 9, 12], "label_pos": [1, 1], "color": "#7c3aed", "description": "智能体起始生活区，包含床和住宅周边活动空间。"},
    {"id": "commercial_area", "name": "商业区", "kind": "commercial", "bounds": [0, 15, 8, 24], "label_pos": [1, 16], "color": "#f59e0b", "description": "补充食物的区域，进入食品店后会自动交互。"},
    {"id": "central_food_area", "name": "中心食物区", "kind": "resource", "bounds": [9, 7, 14, 16], "label_pos": [9, 8], "color": "#22c55e", "description": "地图中部的可采集食物区域。"},
    {"id": "work_area", "name": "工作区", "kind": "work", "bounds": [16, 0, 24, 10], "label_pos": [17, 1], "color": "#3b82f6", "description": "打工赚钱的区域。"},
    {"id": "recreation_area", "name": "娱乐区", "kind": "recreation", "bounds": [16, 16, 24, 24], "label_pos": [17, 17], "color": "#10b981", "description": "恢复放松度的区域。"},
]


ROADS = [
    {"id": "north_road", "name": "北侧横路", "kind": "road", "cells": _horizontal(5, 0, 24), "color": "#8a7356", "width": 1},
    {"id": "central_road", "name": "中心横路", "kind": "road", "cells": _horizontal(12, 0, 24), "color": "#8a7356", "width": 1},
    {"id": "south_road", "name": "南侧横路", "kind": "road", "cells": _horizontal(20, 0, 24), "color": "#8a7356", "width": 1},
    {"id": "central_axis", "name": "中轴竖路", "kind": "road", "cells": _vertical(12, 0, 24), "color": "#8a7356", "width": 1},
]


BASE_OBJECT_REGIONS = {
    "company_1": {"region_id": "work_area", "entrance": [20, 4]},
    "company_2": {"region_id": "work_area", "entrance": [22, 7]},
    "shop_1": {"region_id": "commercial_area", "entrance": [2, 17]},
    "shop_2": {"region_id": "commercial_area", "entrance": [4, 20]},
    "playground_1": {"region_id": "recreation_area", "entrance": [20, 19]},
    "food_1": {"region_id": "central_food_area", "entrance": [10, 9]},
    "food_2": {"region_id": "central_food_area", "entrance": [12, 13]},
    "food_3": {"region_id": "commercial_area", "entrance": [8, 17]},
    "food_4": {"region_id": "work_area", "entrance": [15, 6]},
    "food_5": {"region_id": "recreation_area", "entrance": [16, 19]},
}


def _base_design(object_regions: dict[str, dict[str, Any]]) -> MapDesignData:
    """生成可序列化地图设计，场景只需传入对象和区域绑定。"""

    return {
        "version": 1,
        "map_size": [25, 25],
        "position_format": "[row, col]",
        "bounds_format": "[row_start, col_start, row_end, col_end]",
        "terrain": deepcopy(TERRAIN),
        "regions": deepcopy(REGIONS),
        "roads": deepcopy(ROADS),
        "object_regions": deepcopy(object_regions),
    }


def default_town_map_design() -> MapDesignData:
    object_regions = dict(BASE_OBJECT_REGIONS)
    object_regions.update({
        "bed_1": {"region_id": "residential_area", "entrance": [2, 3]},
        "bed_2": {"region_id": "residential_area", "entrance": [2, 3]},
        "bed_3": {"region_id": "residential_area", "entrance": [2, 5]},
        "bed_4": {"region_id": "residential_area", "entrance": [2, 7]},
        "bed_5": {"region_id": "residential_area", "entrance": [2, 9]},
    })
    return _base_design(object_regions)


def polarization_map_design() -> MapDesignData:
    object_regions = dict(BASE_OBJECT_REGIONS)
    for index in range(1, 11):
        row = 2 if index <= 5 else 4
        col = 1 + ((index - 1) % 5) * 2
        object_regions[f"bed_{index}"] = {"region_id": "residential_area", "entrance": [row, col + 1]}
    return _base_design(object_regions)
