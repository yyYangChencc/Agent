from copy import deepcopy
from typing import Any

from persona.logger import get_logger

logger = get_logger(__name__)

MapDesignData = dict[str, Any]

class Map:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.grid = [['0' for _ in range(width)] for _ in range(height)]

    def is_empty(self, x, y):
        return self.grid[x][y] == '0'

    def place(self, x, y, obj_id):
        self.grid[x][y] = obj_id

    def remove(self, x, y):
        self.grid[x][y] = '0'

    def get_e(self, x, y):
        return self.grid[x][y]

    def print_map(self):
        rows = []
        for i in range(self.height):
            rows.append(' '.join(self.grid[i][j] for j in range(self.width)))
        logger.debug("地图状态:\n%s", '\n'.join(rows))


def _horizontal(row: int, col_start: int, col_end: int) -> list[list[int]]:
    return [[row, col] for col in range(col_start, col_end + 1)]


def _vertical(col: int, row_start: int, row_end: int) -> list[list[int]]:
    return [[row, col] for row in range(row_start, row_end + 1)]


DEFAULT_TERRAIN = [
    {
        "id": "grass_base",
        "name": "草地底色",
        "kind": "grass",
        "bounds": [0, 0, 24, 24],
        "color": "#203a2f",
        "alpha": 1.0,
    },
    {
        "id": "home_yards",
        "name": "住宅庭院",
        "kind": "yard",
        "bounds": [0, 0, 9, 12],
        "color": "#315a3b",
        "alpha": 0.45,
    },
    {
        "id": "market_floor",
        "name": "商业铺装",
        "kind": "plaza",
        "bounds": [0, 15, 8, 24],
        "color": "#4a3a2a",
        "alpha": 0.45,
    },
    {
        "id": "office_ground",
        "name": "办公地块",
        "kind": "office_ground",
        "bounds": [16, 0, 24, 10],
        "color": "#293d52",
        "alpha": 0.4,
    },
    {
        "id": "park_ground",
        "name": "公园地块",
        "kind": "park",
        "bounds": [16, 16, 24, 24],
        "color": "#1f5a43",
        "alpha": 0.5,
    },
]


DEFAULT_REGIONS = [
    {
        "id": "residential_area",
        "name": "住宅区",
        "kind": "residential",
        "bounds": [0, 0, 9, 12],
        "label_pos": [1, 1],
        "color": "#7c3aed",
        "description": "智能体起始生活区，包含 bed_1 到 bed_5。",
    },
    {
        "id": "commercial_area",
        "name": "商业区",
        "kind": "commercial",
        "bounds": [0, 15, 8, 24],
        "label_pos": [1, 16],
        "color": "#f59e0b",
        "description": "购买食物的区域，包含 shop_1 和 shop_2。",
    },
    {
        "id": "central_food_area",
        "name": "中心食物区",
        "kind": "resource",
        "bounds": [9, 7, 14, 16],
        "label_pos": [9, 8],
        "color": "#22c55e",
        "description": "地图中部的可采集食物区域，包含 food_1 和 food_2。",
    },
    {
        "id": "work_area",
        "name": "工作区",
        "kind": "work",
        "bounds": [16, 0, 24, 10],
        "label_pos": [17, 1],
        "color": "#3b82f6",
        "description": "打工赚钱的区域，包含 company_1 和 company_2。",
    },
    {
        "id": "recreation_area",
        "name": "娱乐区",
        "kind": "recreation",
        "bounds": [16, 16, 24, 24],
        "label_pos": [17, 17],
        "color": "#10b981",
        "description": "恢复放松度的区域，包含 playground_1。",
    },
]


DEFAULT_ROADS = [
    {
        "id": "north_road",
        "name": "北侧横路",
        "kind": "road",
        "cells": _horizontal(5, 0, 24),
        "color": "#8a7356",
        "width": 1,
    },
    {
        "id": "central_road",
        "name": "中心横路",
        "kind": "road",
        "cells": _horizontal(12, 0, 24),
        "color": "#8a7356",
        "width": 1,
    },
    {
        "id": "south_road",
        "name": "南侧横路",
        "kind": "road",
        "cells": _horizontal(20, 0, 24),
        "color": "#8a7356",
        "width": 1,
    },
    {
        "id": "central_axis",
        "name": "中轴竖路",
        "kind": "road",
        "cells": _vertical(12, 0, 24),
        "color": "#8a7356",
        "width": 1,
    },
]


DEFAULT_OBJECT_REGIONS = {
    "bed_1": {"region_id": "residential_area", "entrance": [2, 3]},
    "bed_2": {"region_id": "residential_area", "entrance": [2, 3]},
    "bed_3": {"region_id": "residential_area", "entrance": [2, 5]},
    "bed_4": {"region_id": "residential_area", "entrance": [2, 7]},
    "bed_5": {"region_id": "residential_area", "entrance": [2, 9]},
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


def build_default_map_design(width: int, height: int) -> MapDesignData:
    """Return a serializable design for the current grid map."""
    return {
        "version": 1,
        "map_size": [width, height],
        "position_format": "[row, col]",
        "bounds_format": "[row_start, col_start, row_end, col_end]",
        "terrain": deepcopy(DEFAULT_TERRAIN),
        "regions": deepcopy(DEFAULT_REGIONS),
        "roads": deepcopy(DEFAULT_ROADS),
        "object_regions": deepcopy(DEFAULT_OBJECT_REGIONS),
    }


def serialize_map_design(map_design: MapDesignData | None) -> MapDesignData | None:
    if map_design is None:
        return None
    return deepcopy(map_design)
