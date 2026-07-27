from __future__ import annotations

from copy import deepcopy

from .builder import build_runtime_from_spec
from .iac_gay_marriage_30 import BALANCED_MODE, build_spec


SPEC = build_spec("iac_gay_marriage_30_balanced", BALANCED_MODE)
AGENT_IDS = list(SPEC["agent_ids"])


def build_runtime(**kwargs):
    """使用共享真实历史初始化构建两方入边平均的 30 人场景。"""

    return build_runtime_from_spec(deepcopy(SPEC), **kwargs)
