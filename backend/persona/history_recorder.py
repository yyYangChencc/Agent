from __future__ import annotations
import csv
import os
from datetime import datetime
from typing import TYPE_CHECKING, IO

if TYPE_CHECKING:
    from persona.agents.agent import Agent

FIELDS = ["tick", "opinion", "satiety", "relax", "money",
          "satiety_demand", "relax_demand", "emotion", "task"]

# 历史记录默认存储在 backend/history/<timestamp>/ 下
_DEFAULT_BASE = os.path.join(os.path.dirname(__file__), "..", "history")


class HistoryRecorder:
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
                path = os.path.join(self.output_dir, f"{agent.id}.csv")
                f = open(path, "w", newline="", encoding="utf-8")
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                writer.writeheader()
                self._files[agent.id] = f
                self._writers[agent.id] = writer
            self._writers[agent.id].writerow({
                "tick": tick,
                "opinion": round(agent.opinion, 4),
                "satiety": round(agent.need.get("satiety", 0), 2),
                "relax": round(agent.need.get("relax", 0), 2),
                "money": round(agent.need.get("money", 0), 2),
                "satiety_demand": round(agent.demand.get("satiety", 0), 4),
                "relax_demand": round(agent.demand.get("relax", 0), 4),
                "emotion": agent.emotion,
                "task": agent.task,
            })
            self._files[agent.id].flush()

    def close(self) -> None:
        for f in self._files.values():
            f.close()
        self._files.clear()
        self._writers.clear()
