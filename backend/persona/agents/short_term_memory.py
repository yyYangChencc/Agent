from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any


SUMMARY_RECORD_TYPE = "short_term_summary"


def _stable_text(value: Any) -> str:
    """把结构化内容稳定转换为可读文本。"""

    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _project_observation(value: dict[str, Any]) -> dict[str, Any]:
    """保留观察中与连续决策直接相关的精确字段。"""

    people = []
    for item in value.get("people") or []:
        if not isinstance(item, dict):
            continue
        people.append({
            "id": item.get("id"),
            "position": item.get("position"),
            "inside_building_id": item.get("inside_building_id"),
        })

    objects = []
    for item in value.get("objects") or []:
        if not isinstance(item, dict):
            continue
        objects.append({
            "id": item.get("id"),
            "kind": item.get("kind"),
            "position": item.get("position"),
            "owner_agent_id": item.get("owner_agent_id"),
            "free_num": item.get("free_num"),
            "occupant_id": item.get("occupant_id"),
        })

    actions = []
    for item in value.get("actions") or []:
        if not isinstance(item, dict):
            continue
        actions.append({
            "type": item.get("type"),
            "actor_id": item.get("actor_id"),
            "acted_id": item.get("acted_id"),
            "info": item.get("info"),
            "time": item.get("time"),
        })

    social = value.get("social") if isinstance(value.get("social"), dict) else {}
    notifications = []
    for item in social.get("notification_events") or []:
        if not isinstance(item, dict):
            continue
        notifications.append({
            "event_id": item.get("event_id"),
            "event_type": item.get("event_type"),
            "actor_id": item.get("actor_id"),
            "related_agent_id": item.get("related_agent_id"),
            "post_id": item.get("post_id"),
            "topic": item.get("topic"),
            "content": item.get("content"),
            "time": item.get("time"),
        })

    return {
        "time": value.get("time"),
        "position": value.get("position"),
        "region": value.get("region"),
        "people": people,
        "objects": objects,
        "actions": actions,
        "notification_events": notifications,
    }


@dataclass(frozen=True)
class ShortTermMemoryEntry:
    """一条带时间和来源的短期记忆记录。"""

    entry_id: int
    world_time: int
    record_type: str
    content: Any
    episode_id: str = ""
    task: str = ""
    action_tool: str = ""
    success: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        """生成兼容旧 Prompt 和日志的单行文本。"""

        return f"t={self.world_time} {self.record_type}: {_stable_text(self.content)}"

    def summary_payload(self) -> dict[str, Any]:
        """生成供 LLM 压缩的有界结构化记录。"""

        content = self.content
        if self.record_type == "observation" and isinstance(content, dict):
            content = _project_observation(content)
        return {
            "entry_id": self.entry_id,
            "world_time": self.world_time,
            "record_type": self.record_type,
            "content": copy.deepcopy(content),
            "episode_id": self.episode_id,
            "task": self.task,
            "action_tool": self.action_tool,
            "success": self.success,
            "metadata": copy.deepcopy(self.metadata),
        }


@dataclass(frozen=True)
class ShortTermCompactionSnapshot:
    """LLM 调用前冻结的压缩区间。"""

    revision: int
    entry_ids: tuple[int, ...]
    start_tick: int
    end_tick: int
    entries: tuple[ShortTermMemoryEntry, ...]

    def payload(self) -> list[dict[str, Any]]:
        return [entry.summary_payload() for entry in self.entries]


class ShortTermMemoryBuffer:
    """按完整时间步保存并原子压缩短期记忆。"""

    def __init__(self) -> None:
        self._entries: list[ShortTermMemoryEntry] = []
        self._next_entry_id = 1
        self._revision = 0

    @property
    def entries(self) -> tuple[ShortTermMemoryEntry, ...]:
        return tuple(self._entries)

    @property
    def revision(self) -> int:
        return self._revision

    def append(
        self,
        *,
        world_time: int,
        record_type: str,
        content: Any,
        episode_id: str = "",
        task: str = "",
        action_tool: str = "",
        success: bool | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ShortTermMemoryEntry:
        """追加记录；删除只允许通过压缩提交发生。"""

        entry = ShortTermMemoryEntry(
            entry_id=self._next_entry_id,
            world_time=int(world_time),
            record_type=str(record_type),
            content=copy.deepcopy(content),
            episode_id=str(episode_id or ""),
            task=str(task or ""),
            action_tool=str(action_tool or ""),
            success=success,
            metadata=copy.deepcopy(metadata or {}),
        )
        self._next_entry_id += 1
        self._entries.append(entry)
        self._revision += 1
        return entry

    def history_strings(self) -> list[str]:
        return [entry.render() for entry in self._entries]

    def latest(self, record_types: set[str] | None = None) -> ShortTermMemoryEntry | None:
        for entry in reversed(self._entries):
            if record_types is None or entry.record_type in record_types:
                return entry
        return None

    def raw_ticks(self) -> list[int]:
        return sorted({
            entry.world_time
            for entry in self._entries
            if entry.record_type != SUMMARY_RECORD_TYPE
        })

    def needs_compaction(self, max_ticks: int) -> bool:
        return len(self.raw_ticks()) >= max(1, int(max_ticks))

    def compaction_snapshot(
        self,
        *,
        max_ticks: int,
        hot_ticks: int,
    ) -> ShortTermCompactionSnapshot | None:
        """选择较早完整 tick，并把旧滚动总结一并纳入。"""

        ticks = self.raw_ticks()
        if len(ticks) < max(1, int(max_ticks)):
            return None
        keep_count = max(1, min(int(hot_ticks), len(ticks) - 1))
        compact_ticks = set(ticks[:-keep_count])
        selected = [
            entry
            for entry in self._entries
            if entry.record_type == SUMMARY_RECORD_TYPE or entry.world_time in compact_ticks
        ]
        if not selected:
            return None
        raw_selected_ticks = [
            entry.world_time
            for entry in selected
            if entry.record_type != SUMMARY_RECORD_TYPE
        ]
        if not raw_selected_ticks:
            return None
        start_tick = min(
            int(entry.metadata.get("start_tick", entry.world_time))
            if entry.record_type == SUMMARY_RECORD_TYPE
            else entry.world_time
            for entry in selected
        )
        end_tick = max(raw_selected_ticks)
        return ShortTermCompactionSnapshot(
            revision=self._revision,
            entry_ids=tuple(entry.entry_id for entry in selected),
            start_tick=start_tick,
            end_tick=end_tick,
            entries=tuple(selected),
        )

    def commit_summary(
        self,
        snapshot: ShortTermCompactionSnapshot,
        summary: dict[str, Any],
    ) -> bool:
        """仅在缓冲区未变化时原子替换压缩区间。"""

        if snapshot.revision != self._revision:
            return False
        selected_ids = set(snapshot.entry_ids)
        indexes = [
            index for index, entry in enumerate(self._entries)
            if entry.entry_id in selected_ids
        ]
        if len(indexes) != len(selected_ids):
            return False
        insert_at = min(indexes)
        summary_entry = ShortTermMemoryEntry(
            entry_id=self._next_entry_id,
            world_time=snapshot.end_tick,
            record_type=SUMMARY_RECORD_TYPE,
            content=copy.deepcopy(summary),
            episode_id="",
            task=str(summary.get("task_progress") or ""),
            metadata={
                "start_tick": snapshot.start_tick,
                "end_tick": snapshot.end_tick,
                "source_entry_ids": list(snapshot.entry_ids),
            },
        )
        self._next_entry_id += 1
        remaining = [entry for entry in self._entries if entry.entry_id not in selected_ids]
        remaining.insert(insert_at, summary_entry)
        self._entries = remaining
        self._revision += 1
        return True

    def recent_entries(self, tick_count: int) -> list[ShortTermMemoryEntry]:
        ticks = self.raw_ticks()
        selected_ticks = set(ticks[-max(1, int(tick_count)):])
        return [
            entry for entry in self._entries
            if entry.record_type != SUMMARY_RECORD_TYPE and entry.world_time in selected_ticks
        ]

    def summaries(self) -> list[ShortTermMemoryEntry]:
        return [entry for entry in self._entries if entry.record_type == SUMMARY_RECORD_TYPE]

