from __future__ import annotations
import csv
import json
import os
from datetime import datetime
from typing import TYPE_CHECKING, IO

if TYPE_CHECKING:
    from persona.agents.agent import Agent

RUNTIME_NEED_KEYS = ["satiety", "relax", "money", "belonging", "esteem", "self_actualization"]

CSV_TOPIC_POST_FIELDS = (
    "id",
    "author_id",
    "topic",
    "time",
    "opinion_index",
    "is_news",
    "is_rumor",
    "source_type",
    "repost_of_post_id",
    "root_post_id",
    "source_author_id",
    "content",
)

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
        "repost_of_post_id", "root_post_id", "source_author_id",
        "comment_id", "parent_comment_id", "root_comment_id",
        "conversation_messages", "need_events",
        "opinion_before", "opinion_after",
        "opinion_assessment_topic", "opinion_assessment_score", "opinion_scores",
        "opinion_assessment_reason", "opinion_assessment_evidence",
        "opinion_assessment_method", "current_honest_belief", "opinion_flan_rating", "opinion_flan_model",
        "opinion_voting_tick", "opinion_voting_options", "opinion_voting_choice_counts",
        "opinion_voting_option_roles", "opinion_voting_choice_shares",
        "opinion_voting_votes", "opinion_voting_requested_voters",
        "opinion_voting_successful_votes", "opinion_voting_failed_votes",
        "opinion_voting_support_share", "opinion_voting_oppose_share",
        "opinion_voting_unknown_share", "opinion_voting_known_share",
        "opinion_voting_decisiveness",
        "opinion_voting_stance", "opinion_voting_stance_valid",
        "opinion_voting_stance_agreement", "opinion_voting_stance_direction_margin",
        "opinion_voting_stance_success_rate",
        "opinion_voting_window_start_tick", "opinion_voting_window_end_tick",
        "opinion_voting_speech_history",
        "seen_topic_posts", "visible_topic_posts",
        "active_memory_events", "active_derived_memories", "active_memory_vectors",
        "pruned_low_value_events", "pruned_events_to_limit", "pruned_vector_memories",
        "deleted_memory_vectors", "vector_delete_failures",
        "emotion", "task",
    ]
)

# 历史记录默认存储在 backend/history/<scenario_name>_<timestamp>/ 下
_DEFAULT_BASE = os.path.join(os.path.dirname(__file__), "..", "history")
_INVALID_RUN_NAME_CHARS = '<>:"/\\|?*'


class HistoryRecorder:
    """把每个智能体的 tick 状态写入 CSV。

    这些 CSV 是实验分析入口，重点保存需求、压力、观念评测和任务字段；
    前端实时历史只保留最近点，完整轨迹以这里为准。
    """

    def __init__(self, base_dir: str = _DEFAULT_BASE, *, scenario_name: str):
        # 只生成本次运行的目标路径；真正写入第一条记录时再创建目录。
        if (
            not scenario_name
            or scenario_name in {".", ".."}
            or any(char in _INVALID_RUN_NAME_CHARS or ord(char) < 32 for char in scenario_name)
        ):
            raise ValueError(f"scenario_name 不能用于历史目录名：{scenario_name!r}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{scenario_name}_{timestamp}"
        self.base_dir = base_dir
        self.scenario_name = scenario_name
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
            # CSV 仅写帖子紧凑快照，完整评论继续写入 JSONL。
            self._writers[agent.id].writerow(self._build_csv_row(row))
            self._files[agent.id].flush()
            self._write_jsonl(agent.id, self._build_jsonl_record(row))

    def _build_row(self, tick: int, agent: Agent, platform=None) -> dict:
        """把运行时对象压平为 CSV/JSONL 共享行。"""

        last_assessment = agent.last_opinion_assessment or {}
        last_voting = agent.last_opinion_voting or {}
        current_voting = last_voting if last_voting.get("tick") == tick else {}
        psychological = agent.last_psychological_assessment or {}
        role_card = psychological.get("role_card_delta", {})
        mediators = self._collect_mediators(psychological)
        action = getattr(agent, "last_action", {}) or {}
        social_action = getattr(agent, "last_social_action", {}) or {}
        social_relations = self._social_relation_fields(social_action, platform)
        memory_counts, memory_maintenance = self._memory_stats(agent)
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
                "repost_of_post_id": social_relations["repost_of_post_id"],
                "root_post_id": social_relations["root_post_id"],
                "source_author_id": social_relations["source_author_id"],
                "comment_id": social_relations["comment_id"],
                "parent_comment_id": social_relations["parent_comment_id"],
                "root_comment_id": social_relations["root_comment_id"],
                "opinion_before": opinion_before,
                "opinion_after": last_assessment.get("score", agent.opinion),
                "opinion_assessment_topic": last_assessment.get("topic", ""),
                "opinion_assessment_score": last_assessment.get("score", ""),
                "opinion_scores": json.dumps(agent.opinion_scores, ensure_ascii=False),
                "opinion_assessment_reason": last_assessment.get("reason", ""),
                "opinion_assessment_evidence": self._json(last_assessment.get("evidence", [])),
                "opinion_assessment_method": current_voting.get("method", last_assessment.get("source", "")),
                "current_honest_belief": last_assessment.get("current_honest_belief", ""),
                "opinion_flan_rating": last_assessment.get("flan_rating", ""),
                "opinion_flan_model": last_assessment.get("flan_model", ""),
                "opinion_voting_tick": current_voting.get("tick", ""),
                "opinion_voting_options": self._json(current_voting.get("options", [])),
                "opinion_voting_choice_counts": self._json(current_voting.get("choice_counts", {})),
                "opinion_voting_option_roles": self._json(current_voting.get("option_roles", {})),
                "opinion_voting_choice_shares": self._json(current_voting.get("choice_shares", {})),
                "opinion_voting_votes": self._json(current_voting.get("votes", [])),
                "opinion_voting_requested_voters": current_voting.get("requested_voters", ""),
                "opinion_voting_successful_votes": current_voting.get("successful_votes", ""),
                "opinion_voting_failed_votes": current_voting.get("failed_votes", ""),
                "opinion_voting_support_share": current_voting.get("support_share", ""),
                "opinion_voting_oppose_share": current_voting.get("oppose_share", ""),
                "opinion_voting_unknown_share": current_voting.get("unknown_share", ""),
                "opinion_voting_known_share": current_voting.get("known_share", ""),
                "opinion_voting_decisiveness": current_voting.get("decisiveness", ""),
                "opinion_voting_stance": current_voting.get("stance", ""),
                "opinion_voting_stance_valid": current_voting.get("stance_valid", ""),
                "opinion_voting_stance_agreement": current_voting.get("stance_agreement", ""),
                "opinion_voting_stance_direction_margin": current_voting.get("stance_direction_margin", ""),
                "opinion_voting_stance_success_rate": current_voting.get("stance_success_rate", ""),
                "opinion_voting_window_start_tick": current_voting.get("window_start_tick", ""),
                "opinion_voting_window_end_tick": current_voting.get("window_end_tick", ""),
                "opinion_voting_speech_history": self._json(current_voting.get("speech_history", [])),
                "seen_topic_posts": self._json(self._seen_topic_posts(agent)),
                "visible_topic_posts": self._json(self._visible_topic_posts(agent, platform)),
                "active_memory_events": memory_counts["active_memory_events"],
                "active_derived_memories": memory_counts["active_derived_memories"],
                "active_memory_vectors": memory_counts["active_memory_vectors"],
                "pruned_low_value_events": memory_maintenance.get("pruned_low_value_events", 0),
                "pruned_events_to_limit": memory_maintenance.get("pruned_events_to_limit", 0),
                "pruned_vector_memories": memory_maintenance.get("pruned_vector_memories", 0),
                "deleted_memory_vectors": memory_maintenance.get("deleted_memory_vectors", 0),
                "vector_delete_failures": memory_maintenance.get("vector_delete_failures", 0),
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

    def _memory_stats(self, agent: Agent) -> tuple[dict[str, int], dict]:
        """读取当前活跃记忆数量和本轮轻量维护结果。"""

        empty = {"active_memory_events": 0, "active_derived_memories": 0, "active_memory_vectors": 0}
        mem = getattr(agent, "mem", None)
        if mem is None:
            return empty, {}
        try:
            counts = mem.get_active_memory_counts(agent.id) if hasattr(mem, "get_active_memory_counts") else empty
        except Exception:
            counts = empty
        try:
            maintenance = mem.get_last_memory_maintenance(agent.id) if hasattr(mem, "get_last_memory_maintenance") else {}
        except Exception:
            maintenance = {}
        return {key: int(counts.get(key, 0) or 0) for key in empty}, maintenance

    def _social_relation_fields(self, social_action: dict, platform) -> dict:
        """从本轮真实帖子和评论对象读取传播关系。"""

        fields = {
            "repost_of_post_id": "",
            "root_post_id": "",
            "source_author_id": "",
            "comment_id": social_action.get("comment_id", ""),
            "parent_comment_id": social_action.get("parent_comment_id", ""),
            "root_comment_id": "",
        }
        if platform is None:
            return fields
        post_id = social_action.get("post_id")
        posts_lock = getattr(platform, "_posts_lock", None)
        if posts_lock is None:
            post = next((item for item in platform.posts if item.id == post_id), None)
        else:
            with posts_lock:
                post = next((item for item in platform.posts if item.id == post_id), None)
        if post is None:
            return fields

        for field_name in ("repost_of_post_id", "root_post_id", "source_author_id"):
            value = getattr(post, field_name)
            fields[field_name] = value if value is not None else ""

        comment_id = fields["comment_id"]
        if not comment_id:
            return fields
        comment = post.get_comment(comment_id)
        if comment is None:
            return fields
        fields["parent_comment_id"] = comment.parent_comment_id if comment.parent_comment_id is not None else ""
        fields["root_comment_id"] = comment.root_comment_id if comment.root_comment_id is not None else ""
        return fields

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
            "opinion_voting_options",
            "opinion_voting_choice_counts",
            "opinion_voting_option_roles",
            "opinion_voting_choice_shares",
            "opinion_voting_votes",
            "opinion_voting_speech_history",
            "seen_topic_posts",
            "visible_topic_posts",
            "conversation_messages",
            "need_events",
        ]:
            record[key] = self._loads_json_field(row.get(key))
        return record

    def _build_csv_row(self, row: dict) -> dict:
        """为 CSV 移除评论正文，同时保留可分析的帖子字段。"""

        csv_row = dict(row)
        for field_name in ("seen_topic_posts", "visible_topic_posts"):
            posts = self._loads_json_field(row.get(field_name))
            csv_row[field_name] = self._json(self._compact_topic_posts(posts))
        return csv_row

    @staticmethod
    def _compact_topic_posts(posts) -> list[dict]:
        """生成不含评论正文的帖子快照。"""

        if not isinstance(posts, list):
            return []
        compact_posts = []
        for post in posts:
            if not isinstance(post, dict):
                continue
            comments = post.get("comments")
            compact_post = {field_name: post.get(field_name) for field_name in CSV_TOPIC_POST_FIELDS}
            compact_post["comments_count"] = len(comments) if isinstance(comments, list) else 0
            compact_posts.append(compact_post)
        return compact_posts

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
                "repost_of_post_id": post.get("repost_of_post_id"),
                "root_post_id": post.get("root_post_id"),
                "source_author_id": post.get("source_author_id"),
                "content": post.get("content"),
                "comments": [
                    self._comment_history_snapshot(comment)
                    for comment in post.get("comments", [])
                    if isinstance(comment, dict)
                ],
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
            post_data = post.to_dict()
            if str(post_data.get("topic") or "") != topic:
                continue
            out.append({
                "id": post_data.get("id"),
                "author_id": post_data.get("author_id"),
                "topic": post_data.get("topic"),
                "time": post_data.get("time"),
                "opinion_index": post_data.get("opinion_index"),
                "is_news": post_data.get("is_news"),
                "is_rumor": post_data.get("is_rumor"),
                "source_type": post_data.get("source_type"),
                "repost_of_post_id": post_data.get("repost_of_post_id"),
                "root_post_id": post_data.get("root_post_id"),
                "source_author_id": post_data.get("source_author_id"),
                "content": post_data.get("content"),
                "comments": [
                    self._comment_history_snapshot(comment)
                    for comment in post_data.get("comments", [])
                    if isinstance(comment, dict)
                ],
            })
        return out

    @staticmethod
    def _comment_history_snapshot(comment: dict) -> dict:
        """保留评论正文及回复链字段。"""

        return {
            "id": comment.get("id"),
            "author_id": comment.get("author_id"),
            "content": comment.get("content"),
            "time": comment.get("time"),
            "agreement_to_post": comment.get("agreement_to_post"),
            "parent_comment_id": comment.get("parent_comment_id"),
            "root_comment_id": comment.get("root_comment_id"),
        }

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
