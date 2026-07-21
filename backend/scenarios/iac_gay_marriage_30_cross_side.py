from __future__ import annotations

from copy import deepcopy

from .builder import build_runtime_from_spec
from .iac_gay_marriage_30 import CROSS_SIDE_MODE, build_spec


SPEC = build_spec("iac_gay_marriage_30_cross_side", CROSS_SIDE_MODE)
AGENT_IDS = list(SPEC["agent_ids"])


def build_runtime(**kwargs):
    """构建异侧关注且保留原社区分配的 30 人场景。"""

    return build_runtime_from_spec(deepcopy(SPEC), **kwargs)
