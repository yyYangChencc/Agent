from __future__ import annotations
import csv
import json
import os
from datetime import datetime
from typing import TYPE_CHECKING, IO

if TYPE_CHECKING:
    from persona.agents.agent import Agent

RUNTIME_NEED_KEYS = ["satiety", "relax", "money", "belonging", "esteem", "self_actualization"]

FIELDS = (
    ["tick", "agent_id", "opinion"]
    + RUNTIME_NEED_KEYS
    + [f"{key}_urgency" for key in RUNTIME_NEED_KEYS]
    + [f"{key}_gap" for key in RUNTIME_NEED_KEYS]
    + [f"{key}_pressure_memory" for key in RUNTIME_NEED_KEYS]
    + [f"{key}_load_saturation" for key in RUNTIME_NEED_KEYS]
    + [f"{key}_effective_pressure" for key in RUNTIME_NEED_KEYS]
    + [
        "mediators", "role_card_delta", "active_role_cards",
        "action_tool", "post_id", "post_content", "comment_content", "social_action",
        "post_opinion_index", "agreement_to_post",
        "conversation_messages", "need_events",
        "opinion_before", "opinion_after",
        "opinion_assessment_topic", "opinion_assessment_score", "opinion_scores",
        "opinion_assessment_reason", "opinion_assessment_evidence",
        "seen_topic_posts", "visible_topic_posts",
        "emotion", "task",
    ]
)

# 历史记录默认存储在 backend/history/<timestamp>/ 下
_DEFAULT_BASE = os.path.join(os.path.dirname(__file__), "..", "history")


class HistoryRecorder:
    """把每个智能体的 tick 状态写入 CSV。

    这些 CSV 是实验分析入口，重点保存需求、压力、观念评测和任务字段；
    前端实时历史只保留最近点，完整轨迹以这里为准。
    """

    def __init__(self, base_dir: str = _DEFAULT_BASE):
        # 只生成本次运行的目标路径；真正写入第一条记录时再创建目录。
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir = base_dir
        self.run_name = run_name
        self.output_dir = os.path.join(base_dir, run_name)
        self._output_dir_ready = False
        self._writers: dict[str, csv.DictWriter] = {}
        self._files: dict[str, IO] = {}
        self._jsonl_files: dict[str, IO] = {}

    def ensure_output_dir(self) -> str:
        """确保历史输出目录存在；场景仅加载时不会调用该方法。"""

        if not self._output_dir_ready:
            os.makedirs(self.output_dir, exist_ok=True)
            self._output_dir_ready = True
        return self.output_dir

    def record(self, tick: int, agents: list[Agent], platform=None) -> None:
        self.ensure_output_dir()
        for agent in agents:
            if agent.id not in self._writers:
                # 每个智能体单独一个 CSV，便于后续按个体画时间序列。
                path = os.path.join(self.output_dir, f"{agent.id}.csv")
                f = open(path, "w", newline="", encoding="utf-8")
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                writer.writeheader()
                self._files[agent.id] = f
                self._writers[agent.id] = writer
            row = self._build_row(tick, agent, platform=platform)
            self._writers[agent.id].writerow(row)
            self._files[agent.id].flush()
            self._write_jsonl(agent.id, self._build_jsonl_record(row))

    def _build_row(self, tick: int, agent: Agent, platform=None) -> dict:
        """把运行时对象压平为 CSV/JSONL 共享行。"""

        last_assessment = agent.last_opinion_assessment or {}
        psychological = agent.last_psychological_assessment or {}
        role_card = psychological.get("role_card_delta", {})
        mediators = self._collect_mediators(psychological)
        action = getattr(agent, "last_action", {}) or {}
        social_action = getattr(agent, "last_social_action", {}) or {}
        opinion_before = last_assessment.get(
            "before_score",
            getattr(agent, "last_opinion_before_assessment", agent.opinion),
        )
        row = {
                "tick": tick,
                "agent_id": agent.id,
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
                "satiety_load_saturation": round(agent.load_saturation.get("satiety", 0), 4),
                "relax_load_saturation": round(agent.load_saturation.get("relax", 0), 4),
                "money_load_saturation": round(agent.load_saturation.get("money", 0), 4),
                "satiety_effective_pressure": round(agent.effective_pressure.get("satiety", 0), 4),
                "relax_effective_pressure": round(agent.effective_pressure.get("relax", 0), 4),
                "money_effective_pressure": round(agent.effective_pressure.get("money", 0), 4),
                "mediators": self._json(mediators),
                "role_card_delta": self._json(role_card),
                "active_role_cards": self._json({
                    "status": psychological.get("status"),
                    "activated_needs": psychological.get("activated_needs", []),
                    "recovering_needs": psychological.get("recovering_needs", []),
                    "role_card_delta": role_card,
                }),
                "action_tool": action.get("tool", ""),
                "post_id": social_action.get("post_id", action.get("args", {}).get("post_id", "")),
                "post_content": social_action.get("post_content", ""),
                "comment_content": social_action.get("comment_content", ""),
                "social_action": social_action.get("action", ""),
                "post_opinion_index": social_action.get("opinion_index", ""),
                "agreement_to_post": social_action.get("agreement_to_post", ""),
                "opinion_before": opinion_before,
                "opinion_after": last_assessment.get("score", agent.opinion),
                "opinion_assessment_topic": last_assessment.get("topic", ""),
                "opinion_assessment_score": last_assessment.get("score", ""),
                "opinion_scores": json.dumps(agent.opinion_scores, ensure_ascii=False),
                "opinion_assessment_reason": last_assessment.get("reason", ""),
                "opinion_assessment_evidence": self._json(last_assessment.get("evidence", [])),
                "seen_topic_posts": self._json(self._seen_topic_posts(agent)),
                "visible_topic_posts": self._json(self._visible_topic_posts(agent, platform)),
                "emotion": agent.emotion,
                "task": agent.task,
            }
        # 动态补齐所有运行时需求字段，避免新增需求只出现在 JSONL 而不出现在 CSV。
        for need_key in RUNTIME_NEED_KEYS:
            row[need_key] = round(agent.satisfaction.get(need_key, 0), 2)
            row[f"{need_key}_urgency"] = round(agent.urgency.get(need_key, 0), 4)
            row[f"{need_key}_gap"] = round(agent.need_gap.get(need_key, 0), 4)
            row[f"{need_key}_pressure_memory"] = round(agent.pressure_memory.get(need_key, 0), 4)
            row[f"{need_key}_load_saturation"] = round(agent.load_saturation.get(need_key, 0), 4)
            row[f"{need_key}_effective_pressure"] = round(agent.effective_pressure.get(need_key, 0), 4)
        row["conversation_messages"] = self._json(self._conversation_messages(agent, tick))
        row["need_events"] = self._json(self._need_events(agent, tick))
        return row

    def _write_jsonl(self, agent_id: str, record: dict) -> None:
        self.ensure_output_dir()
        if agent_id not in self._jsonl_files:
            path = os.path.join(self.output_dir, f"{agent_id}.jsonl")
            self._jsonl_files[agent_id] = open(path, "w", encoding="utf-8")
        f = self._jsonl_files[agent_id]
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        f.flush()

    def _build_jsonl_record(self, row: dict) -> dict:
        """JSONL 保留嵌套结构，便于追踪完整解释链。"""

        record = dict(row)
        # JSONL 面向解释链追踪，额外补回按需求分组的嵌套状态。
        record["satisfaction"] = self._group_need_fields(row, {
            "satiety": "satiety",
            "relax": "relax",
            "money": "money",
        })
        record["need_gap"] = self._group_need_fields(row, {
            "satiety": "satiety_gap",
            "relax": "relax_gap",
            "money": "money_gap",
        })
        record["pressure_memory"] = self._group_need_fields(row, {
            "satiety": "satiety_pressure_memory",
            "relax": "relax_pressure_memory",
            "money": "money_pressure_memory",
        })
        record["load_saturation"] = self._group_need_fields(row, {
            "satiety": "satiety_load_saturation",
            "relax": "relax_load_saturation",
            "money": "money_load_saturation",
        })
        record["effective_pressure"] = self._group_need_fields(row, {
            "satiety": "satiety_effective_pressure",
            "relax": "relax_effective_pressure",
            "money": "money_effective_pressure",
        })
        # 覆盖旧三项需求分组，最终以完整运行时需求键为准。
        record["satisfaction"] = self._group_need_fields(row, {
            need_key: need_key for need_key in RUNTIME_NEED_KEYS
        })
        record["need_gap"] = self._group_need_fields(row, {
            need_key: f"{need_key}_gap" for need_key in RUNTIME_NEED_KEYS
        })
        record["pressure_memory"] = self._group_need_fields(row, {
            need_key: f"{need_key}_pressure_memory" for need_key in RUNTIME_NEED_KEYS
        })
        record["load_saturation"] = self._group_need_fields(row, {
            need_key: f"{need_key}_load_saturation" for need_key in RUNTIME_NEED_KEYS
        })
        record["effective_pressure"] = self._group_need_fields(row, {
            need_key: f"{need_key}_effective_pressure" for need_key in RUNTIME_NEED_KEYS
        })
        for key in [
            "mediators",
            "role_card_delta",
            "active_role_cards",
            "opinion_scores",
            "opinion_assessment_evidence",
            "seen_topic_posts",
            "visible_topic_posts",
            "conversation_messages",
            "need_events",
        ]:
            record[key] = self._loads_json_field(row.get(key))
        return record

    def _group_need_fields(self, row: dict, mapping: dict[str, str]) -> dict[str, float]:
        """把 CSV 扁平字段还原为按需求键分组的 JSONL 字段。"""

        grouped = {}
        for need_key, field_name in mapping.items():
            value = row.get(field_name, 0.0)
            try:
                grouped[need_key] = float(value)
            except (TypeError, ValueError):
                grouped[need_key] = 0.0
        return grouped

    def _conversation_messages(self, agent: Agent, tick: int) -> list[dict]:
        """导出当前 tick 已落地的结构化线下对话消息。"""

        messages = []
        for item in getattr(agent, "conversation_event_log", []) or []:
            if isinstance(item, dict) and item.get("time") == tick:
                messages.append(dict(item))
        return messages

    def _need_events(self, agent: Agent, tick: int) -> list[dict]:
        """导出当前 tick 由对话等事件触发的需求变化。"""

        events = []
        for item in getattr(agent, "need_event_log", []) or []:
            if isinstance(item, dict) and item.get("tick") == tick:
                events.append(dict(item))
        return events

    def _collect_mediators(self, psychological: dict) -> dict:
        if not isinstance(psychological, dict):
            return {}
        mediators = psychological.get("mediators")
        if isinstance(mediators, dict):
            return mediators
        out = {}
        evaluator_results = psychological.get("evaluator_results")
        if isinstance(evaluator_results, dict):
            for need_key, result in evaluator_results.items():
                if not isinstance(result, dict):
                    continue
                for key, value in (result.get("mediators") or {}).items():
                    out[f"{need_key}.{key}"] = value
        return out

    def _seen_topic_posts(self, agent: Agent) -> list[dict]:
        """记录当前观念主题下实际看过的帖子来源，便于追踪极化暴露。"""

        out = []
        for post in getattr(agent, "opinion_seen_posts_buffer", []) or []:
            if not isinstance(post, dict):
                continue
            out.append({
                "id": post.get("id"),
                "author_id": post.get("author_id"),
                "topic": post.get("topic"),
                "time": post.get("time"),
                "opinion_index": post.get("opinion_index"),
                "is_news": post.get("is_news"),
                "is_rumor": post.get("is_rumor"),
                "source_type": post.get("source_type"),
                "content": post.get("content"),
            })
        return out

    def _visible_topic_posts(self, agent: Agent, platform) -> list[dict]:
        """记录当前 tick 结束时该智能体可见的当前主题帖子。"""

        if platform is None or not hasattr(platform, "get_visible_posts"):
            return []
        topic = str(getattr(agent.config, "default_opinion_topic", "") or "")
        if not topic:
            return []
        out = []
        for post in platform.get_visible_posts(agent.id):
            if str(getattr(post, "topic", "") or "") != topic:
                continue
            out.append({
                "id": getattr(post, "id", None),
                "author_id": getattr(post, "author_id", None),
                "topic": getattr(post, "topic", None),
                "time": getattr(post, "time", None),
                "opinion_index": getattr(post, "opinion_index", None),
                "is_news": getattr(post, "is_news", None),
                "is_rumor": getattr(post, "is_rumor", None),
                "source_type": getattr(post, "source_type", None),
                "content": getattr(post, "content", None),
            })
        return out

    def _json(self, value) -> str:
        return json.dumps(value, ensure_ascii=False, default=str)

    def _loads_json_field(self, value):
        """把 CSV 中的 JSON 字符串还原为 JSONL 中的对象/数组。"""

        if isinstance(value, (dict, list)):
            return value
        if not isinstance(value, str) or value == "":
            return value
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    def close(self) -> None:
        for f in self._files.values():
            f.close()
        for f in self._jsonl_files.values():
            f.close()
        self._files.clear()
        self._writers.clear()
        self._jsonl_files.clear()
