from __future__ import annotations

from importlib import import_module
from types import ModuleType


SCENARIO_MODULES = {
    "default_town": "scenarios.default_town",
    "iac_gay_marriage": "scenarios.iac_gay_marriage",
    "iac_gay_marriage_10": "scenarios.iac_gay_marriage_10",
    "jiang_ping_polarization": "scenarios.jiang_ping_polarization",
}


def list_scenarios() -> list[str]:
    """返回可用场景名，供 CLI choices 和前端选择使用。"""

    return sorted(SCENARIO_MODULES)


def get_scenario(name: str) -> ModuleType:
    """按精确场景名加载场景模块。"""

    if name not in SCENARIO_MODULES:
        raise ValueError(f"unknown scenario: {name}")
    return import_module(SCENARIO_MODULES[name])
