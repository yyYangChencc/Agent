from __future__ import annotations
import csv
import json
import os
from datetime import datetime
from typing import TYPE_CHECKING, IO

if TYPE_CHECKING:
    from persona.agents.agent import Agent

FIELDS = [
    "tick", "opinion", "satiety", "relax", "money",
    "satiety_urgency", "relax_urgency", "money_urgency",
    "satiety_gap", "relax_gap", "money_gap",
    "satiety_pressure_memory", "relax_pressure_memory", "money_pressure_memory",
    "satiety_effective_pressure", "relax_effective_pressure", "money_effective_pressure",
    "opinion_assessment_topic", "opinion_assessment_score", "opinion_scores",
    "emotion", "task",
]

# 历史记录默认存储在 backend/history/<timestamp>/ 下
_DEFAULT_BASE = os.path.join(os.path.dirname(__file__), "..", "history")


class HistoryRecorder:
    """把每个智能体的 tick 状态写入 CSV。

    这些 CSV 是实验分析入口，重点保存需求、压力、观念评测和任务字段；
    前端实时历史只保留最近点，完整轨迹以这里为准。
    """

    def __init__(self, base_dir: str = _DEFAULT_BASE):
        # 每次实例化时，在 base_dir 下创建以当前时间命名的子文件夹
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(base_dir, run_name)
        os.makedirs(self.output_dir, exist_ok=True)
        self._writers: dict[str, csv.DictWriter] = {}
        self._files: dict[str, IO] = {}

    def record(self, tick: int, agents: list[Agent]) -> None:
        for agent in agents:
            if agent.id not in self._writers:
                # 每个智能体单独一个 CSV，便于后续按个体画时间序列。
                path = os.path.join(self.output_dir, f"{agent.id}.csv")
                f = open(path, "w", newline="", encoding="utf-8")
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                writer.writeheader()
                self._files[agent.id] = f
                self._writers[agent.id] = writer
            self._writers[agent.id].writerow({
                "tick": tick,
                "opinion": round(agent.opinion, 4),
                "satiety": round(agent.satisfaction.get("satiety", 0), 2),
                "relax": round(agent.satisfaction.get("relax", 0), 2),
                "money": round(agent.satisfaction.get("money", 0), 2),
                "satiety_urgency": round(agent.urgency.get("satiety", 0), 4),
                "relax_urgency": round(agent.urgency.get("relax", 0), 4),
                "money_urgency": round(agent.urgency.get("money", 0), 4),
                "satiety_gap": round(agent.need_gap.get("satiety", 0), 4),
                "relax_gap": round(agent.need_gap.get("relax", 0), 4),
                "money_gap": round(agent.need_gap.get("money", 0), 4),
                "satiety_pressure_memory": round(agent.pressure_memory.get("satiety", 0), 4),
                "relax_pressure_memory": round(agent.pressure_memory.get("relax", 0), 4),
                "money_pressure_memory": round(agent.pressure_memory.get("money", 0), 4),
                "satiety_effective_pressure": round(agent.effective_pressure.get("satiety", 0), 4),
                "relax_effective_pressure": round(agent.effective_pressure.get("relax", 0), 4),
                "money_effective_pressure": round(agent.effective_pressure.get("money", 0), 4),
                "opinion_assessment_topic": (
                    agent.last_opinion_assessment or {}
                ).get("topic", ""),
                "opinion_assessment_score": (
                    agent.last_opinion_assessment or {}
                ).get("score", ""),
                "opinion_scores": json.dumps(agent.opinion_scores, ensure_ascii=False),
                "emotion": agent.emotion,
                "task": agent.task,
            })
            self._files[agent.id].flush()

    def close(self) -> None:
        for f in self._files.values():
            f.close()
        self._files.clear()
        self._writers.clear()
