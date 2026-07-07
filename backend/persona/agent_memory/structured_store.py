from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


def _json_dumps(value: Any) -> str:
    """统一 JSON 序列化，保证中文原文和键顺序稳定写入 SQLite。"""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _json_loads(value: str | None) -> Any:
    """读取 SQLite 中的 JSON 字段；历史脏数据不是 JSON 时保留原文。"""

    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _text_id(value: Any) -> str:
    """把实体、帖子等索引字段统一转成文本，避免 SQLite 主键类型漂移。"""

    if value is None:
        return ""
    return str(value)


def _text_ids(values: list[Any] | None, *, exclude_system: bool = False) -> list[str]:
    """把一组索引字段转成非空文本 ID。"""

    ids: list[str] = []
    for value in values or []:
        text = _text_id(value)
        if not text:
            continue
        if exclude_system and text == "system":
            continue
        ids.append(text)
    return ids


def _region_id(region: Any) -> str:
    if isinstance(region, dict):
        return _text_id(region.get("id"))
    return ""


class StructuredMemoryStore:
    """结构化记忆的 SQLite 存储层。

    SQLite 负责动态状态和可追踪证据：原始事件、实体当前状态、实体关系、
    社交帖子快照、场景快照、衍生记忆、冲突记录和访问日志。Chroma 只承担
    语义召回，本类不做 embedding。
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _init_schema(self) -> None:
        """初始化结构化记忆表。

        memory_events 保存不可变原始事实；entity_states/social_posts 保存最新状态；
        derived_memories 保存压缩后的长期记忆；memory_conflicts 和 memory_access_log
        分别服务冲突调解与检索评测。
        """

        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS memory_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    world_time INTEGER NOT NULL DEFAULT 0,
                    entity_id TEXT,
                    related_agent_id TEXT,
                    object_id TEXT,
                    post_id TEXT,
                    summary TEXT NOT NULL DEFAULT '',
                    payload_json TEXT NOT NULL,
                    importance REAL NOT NULL DEFAULT 0.5,
                    confidence REAL NOT NULL DEFAULT 0.5,
                    valid INTEGER NOT NULL DEFAULT 1,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_memory_events_agent_time
                    ON memory_events(agent_id, world_time DESC, id DESC);
                CREATE INDEX IF NOT EXISTS idx_memory_events_agent_type
                    ON memory_events(agent_id, memory_type, source_type);
                CREATE INDEX IF NOT EXISTS idx_memory_events_entity
                    ON memory_events(agent_id, entity_id, related_agent_id, object_id, post_id);

                CREATE TABLE IF NOT EXISTS entity_states (
                    agent_id TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    name TEXT NOT NULL DEFAULT '',
                    position_json TEXT,
                    region_json TEXT,
                    region_id TEXT NOT NULL DEFAULT '',
                    first_seen_at INTEGER NOT NULL DEFAULT 0,
                    last_seen_at INTEGER NOT NULL DEFAULT 0,
                    source_type TEXT NOT NULL DEFAULT '',
                    payload_json TEXT NOT NULL,
                    importance REAL NOT NULL DEFAULT 0.5,
                    confidence REAL NOT NULL DEFAULT 0.5,
                    valid INTEGER NOT NULL DEFAULT 1,
                    update_count INTEGER NOT NULL DEFAULT 1,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(agent_id, entity_type, entity_id)
                );

                CREATE INDEX IF NOT EXISTS idx_entity_states_agent_region
                    ON entity_states(agent_id, region_id, entity_type);
                CREATE INDEX IF NOT EXISTS idx_entity_states_agent_seen
                    ON entity_states(agent_id, last_seen_at DESC);

                CREATE TABLE IF NOT EXISTS person_profiles (
                    agent_id TEXT NOT NULL,
                    target_agent_id TEXT NOT NULL,
                    name TEXT NOT NULL DEFAULT '',
                    last_position_json TEXT,
                    last_region_json TEXT,
                    last_seen_at INTEGER NOT NULL DEFAULT 0,
                    last_social_seen_at INTEGER NOT NULL DEFAULT 0,
                    last_reflected_at INTEGER NOT NULL DEFAULT 0,
                    recent_post_summary TEXT NOT NULL DEFAULT '',
                    actions_impression TEXT NOT NULL DEFAULT '',
                    opinion_impression TEXT NOT NULL DEFAULT '',
                    relationship_impression TEXT NOT NULL DEFAULT '',
                    confidence REAL NOT NULL DEFAULT 0.5,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(agent_id, target_agent_id)
                );

                CREATE INDEX IF NOT EXISTS idx_person_profiles_agent_seen
                    ON person_profiles(agent_id, last_seen_at DESC, last_social_seen_at DESC);
                CREATE INDEX IF NOT EXISTS idx_person_profiles_reflect
                    ON person_profiles(agent_id, last_reflected_at, last_seen_at, last_social_seen_at);

                CREATE TABLE IF NOT EXISTS entity_relations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT NOT NULL,
                    subject_id TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    object_id TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    first_seen_at INTEGER NOT NULL DEFAULT 0,
                    last_seen_at INTEGER NOT NULL DEFAULT 0,
                    payload_json TEXT NOT NULL,
                    importance REAL NOT NULL DEFAULT 0.5,
                    confidence REAL NOT NULL DEFAULT 0.5,
                    valid INTEGER NOT NULL DEFAULT 1,
                    update_count INTEGER NOT NULL DEFAULT 1,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE(agent_id, subject_id, relation_type, object_id)
                );

                CREATE INDEX IF NOT EXISTS idx_entity_relations_agent_subject
                    ON entity_relations(agent_id, subject_id, last_seen_at DESC);
                CREATE INDEX IF NOT EXISTS idx_entity_relations_agent_object
                    ON entity_relations(agent_id, object_id, last_seen_at DESC);

                CREATE TABLE IF NOT EXISTS social_posts (
                    agent_id TEXT NOT NULL,
                    post_id TEXT NOT NULL,
                    author_id TEXT NOT NULL DEFAULT '',
                    topic TEXT NOT NULL DEFAULT '',
                    content TEXT NOT NULL DEFAULT '',
                    post_time INTEGER,
                    likes INTEGER NOT NULL DEFAULT 0,
                    dislikes INTEGER NOT NULL DEFAULT 0,
                    reposts INTEGER NOT NULL DEFAULT 0,
                    comments_count INTEGER NOT NULL DEFAULT 0,
                    opinion_index REAL NOT NULL DEFAULT 0.0,
                    is_news INTEGER NOT NULL DEFAULT 0,
                    is_rumor INTEGER NOT NULL DEFAULT 0,
                    first_seen_at INTEGER NOT NULL DEFAULT 0,
                    last_seen_at INTEGER NOT NULL DEFAULT 0,
                    source_type TEXT NOT NULL DEFAULT '',
                    payload_json TEXT NOT NULL,
                    importance REAL NOT NULL DEFAULT 0.5,
                    confidence REAL NOT NULL DEFAULT 0.5,
                    valid INTEGER NOT NULL DEFAULT 1,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(agent_id, post_id)
                );

                CREATE INDEX IF NOT EXISTS idx_social_posts_agent_author
                    ON social_posts(agent_id, author_id, last_seen_at DESC);
                CREATE INDEX IF NOT EXISTS idx_social_posts_agent_time
                    ON social_posts(agent_id, last_seen_at DESC);

                CREATE TABLE IF NOT EXISTS scene_snapshots (
                    agent_id TEXT NOT NULL,
                    world_time INTEGER NOT NULL,
                    observer_position_json TEXT,
                    region_json TEXT,
                    people_count INTEGER NOT NULL DEFAULT 0,
                    objects_count INTEGER NOT NULL DEFAULT 0,
                    actions_count INTEGER NOT NULL DEFAULT 0,
                    notifications_count INTEGER NOT NULL DEFAULT 0,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY(agent_id, world_time)
                );

                CREATE TABLE IF NOT EXISTS derived_memories (
                    memory_id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    world_time INTEGER NOT NULL DEFAULT 0,
                    entity_id TEXT,
                    related_agent_id TEXT,
                    object_id TEXT,
                    post_id TEXT,
                    task TEXT,
                    need_key TEXT,
                    summary TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    importance REAL NOT NULL DEFAULT 0.5,
                    confidence REAL NOT NULL DEFAULT 0.5,
                    valid INTEGER NOT NULL DEFAULT 1,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    last_accessed_at INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_derived_agent_type_time
                    ON derived_memories(agent_id, memory_type, world_time DESC);
                CREATE INDEX IF NOT EXISTS idx_derived_agent_entities
                    ON derived_memories(agent_id, entity_id, related_agent_id, object_id, post_id);

                CREATE TABLE IF NOT EXISTS memory_conflicts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT NOT NULL,
                    conflict_type TEXT NOT NULL,
                    subject_id TEXT NOT NULL DEFAULT '',
                    old_payload_json TEXT NOT NULL,
                    new_payload_json TEXT NOT NULL,
                    resolution TEXT NOT NULL DEFAULT '',
                    resolved INTEGER NOT NULL DEFAULT 0,
                    world_time INTEGER NOT NULL DEFAULT 0,
                    confidence REAL NOT NULL DEFAULT 0.5,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS memory_access_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT NOT NULL,
                    context TEXT NOT NULL,
                    memory_kind TEXT NOT NULL,
                    memory_ref TEXT NOT NULL,
                    world_time INTEGER NOT NULL DEFAULT 0,
                    query_text TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_access_log_agent_time
                    ON memory_access_log(agent_id, world_time DESC, id DESC);
                """
            )
            self._ensure_columns_locked("social_posts", {"topic": "TEXT NOT NULL DEFAULT ''"})
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_social_posts_agent_topic
                    ON social_posts(agent_id, topic, last_seen_at DESC)
                """
            )
            self._migrate_person_entity_states_locked()
            self._conn.commit()

    def _ensure_columns_locked(self, table_name: str, columns: dict[str, str]) -> None:
        """为已有 SQLite 库补齐新增列，避免升级后必须手动删库。"""

        existing_columns = {
            str(row["name"])
            for row in self._conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        }
        for column_name, column_type in columns.items():
            if column_name in existing_columns:
                continue
            self._conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")

    def _migrate_person_entity_states_locked(self) -> None:
        """把旧人物状态迁移为人物档案，并让旧 person entity_states 不再参与默认检索。"""

        now = time.time()
        self._conn.execute(
            """
            INSERT INTO person_profiles (
                agent_id, target_agent_id, name, last_position_json,
                last_region_json, last_seen_at, confidence, created_at, updated_at
            )
            SELECT
                agent_id, entity_id, name, position_json,
                region_json, last_seen_at, confidence,
                COALESCE(created_at, ?), COALESCE(updated_at, ?)
            FROM entity_states
            WHERE entity_type = 'person'
              AND COALESCE(entity_id, '') != ''
            ON CONFLICT(agent_id, target_agent_id)
            DO UPDATE SET
                name = CASE
                    WHEN excluded.name != '' THEN excluded.name
                    ELSE person_profiles.name
                END,
                last_position_json = COALESCE(excluded.last_position_json, person_profiles.last_position_json),
                last_region_json = COALESCE(excluded.last_region_json, person_profiles.last_region_json),
                last_seen_at = MAX(person_profiles.last_seen_at, excluded.last_seen_at),
                confidence = MAX(person_profiles.confidence, excluded.confidence),
                updated_at = excluded.updated_at
            """,
            (now, now),
        )
        self._conn.execute(
            """
            UPDATE entity_states
            SET valid = 0, updated_at = ?
            WHERE entity_type = 'person' AND valid = 1
            """,
            (now,),
        )

    def reset_all(self) -> None:
        """清空所有结构化记忆表，用于服务 reset 时和 Chroma reset 保持一致。"""

        tables = [
            "memory_events",
            "entity_states",
            "person_profiles",
            "entity_relations",
            "social_posts",
            "scene_snapshots",
            "derived_memories",
            "memory_conflicts",
            "memory_access_log",
        ]
        with self._lock:
            for table in tables:
                self._conn.execute(f"DELETE FROM {table}")
            self._conn.commit()

    def record_event(
        self,
        agent_id: str,
        *,
        memory_type: str,
        source_type: str,
        event_type: str,
        world_time: int,
        payload: Any,
        summary: str = "",
        entity_id: Any = None,
        related_agent_id: Any = None,
        object_id: Any = None,
        post_id: Any = None,
        importance: float = 0.5,
        confidence: float = 0.5,
        valid: bool = True,
    ) -> int:
        """写入一条原始事件。

        原始事件保留 observe、action、social、conversation、opinion 等来源的证据，
        后续状态更新、冲突调解和评测都可以追溯到这里的 payload。
        """

        now = time.time()
        with self._lock:
            cursor = self._conn.execute(
                """
                INSERT INTO memory_events (
                    agent_id, memory_type, source_type, event_type, world_time,
                    entity_id, related_agent_id, object_id, post_id, summary,
                    payload_json, importance, confidence, valid, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    agent_id,
                    memory_type,
                    source_type,
                    event_type,
                    int(world_time or 0),
                    _text_id(entity_id) or None,
                    _text_id(related_agent_id) or None,
                    _text_id(object_id) or None,
                    _text_id(post_id) or None,
                    summary,
                    _json_dumps(payload),
                    float(importance),
                    float(confidence),
                    1 if valid else 0,
                    now,
                    now,
                ),
            )
            self._conn.commit()
            return int(cursor.lastrowid)

    def upsert_entity_state(
        self,
        agent_id: str,
        *,
        entity_type: str,
        entity_id: Any,
        name: str = "",
        position: Any = None,
        region: Any = None,
        source_type: str,
        world_time: int,
        payload: Any,
        importance: float = 0.5,
        confidence: float = 0.8,
        valid: bool = True,
    ) -> None:
        """更新人物、物品或建筑的当前状态。

        状态表只保留最新位置和区域；旧状态不会覆盖原始事件，而是在发生高置信
        位置变化时写入 temporal_state_change 冲突记录，说明这是时间变化事实。
        """

        entity_id_text = _text_id(entity_id)
        if not entity_id_text:
            return
        now = time.time()
        region_json = _json_dumps(region) if region is not None else None
        position_json = _json_dumps(position) if position is not None else None
        payload_json = _json_dumps(payload)
        with self._lock:
            existing = self._conn.execute(
                """
                SELECT position_json, region_json, payload_json, confidence
                FROM entity_states
                WHERE agent_id = ? AND entity_type = ? AND entity_id = ?
                """,
                (agent_id, entity_type, entity_id_text),
            ).fetchone()
            if existing is not None:
                old_position = existing["position_json"]
                old_region = existing["region_json"]
                old_payload = existing["payload_json"]
                old_confidence = float(existing["confidence"] or 0.5)
                changed = old_position != position_json or old_region != region_json
                self._conn.execute(
                    """
                    UPDATE entity_states
                    SET name = ?, position_json = ?, region_json = ?, region_id = ?,
                        last_seen_at = ?, source_type = ?, payload_json = ?,
                        importance = MAX(importance, ?),
                        confidence = MAX(confidence, ?),
                        valid = ?, update_count = update_count + 1, updated_at = ?
                    WHERE agent_id = ? AND entity_type = ? AND entity_id = ?
                    """,
                    (
                        name,
                        position_json,
                        region_json,
                        _region_id(region),
                        int(world_time or 0),
                        source_type,
                        payload_json,
                        float(importance),
                        float(confidence),
                        1 if valid else 0,
                        now,
                        agent_id,
                        entity_type,
                        entity_id_text,
                    ),
                )
                if changed and old_confidence >= 0.7 and confidence >= 0.7:
                    self._record_conflict_locked(
                        agent_id=agent_id,
                        conflict_type="temporal_state_change",
                        subject_id=entity_id_text,
                        old_payload=_json_loads(old_payload),
                        new_payload=payload,
                        resolution="latest_observation_is_current_state",
                        resolved=True,
                        world_time=int(world_time or 0),
                        confidence=min(old_confidence, float(confidence)),
                        now=now,
                    )
            else:
                self._conn.execute(
                    """
                    INSERT INTO entity_states (
                        agent_id, entity_type, entity_id, name, position_json,
                        region_json, region_id, first_seen_at, last_seen_at,
                        source_type, payload_json, importance, confidence,
                        valid, update_count, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                    """,
                    (
                        agent_id,
                        entity_type,
                        entity_id_text,
                        name,
                        position_json,
                        region_json,
                        _region_id(region),
                        int(world_time or 0),
                        int(world_time or 0),
                        source_type,
                        payload_json,
                        float(importance),
                        float(confidence),
                        1 if valid else 0,
                        now,
                        now,
                    ),
                )
            self._conn.commit()

    def upsert_person_profile_seen(
        self,
        agent_id: str,
        *,
        target_agent_id: Any,
        name: str = "",
        position: Any = None,
        region: Any = None,
        world_time: int,
        confidence: float = 0.8,
    ) -> None:
        """记录线下 observe 看到的人物基础档案，不再写 person entity_state。"""

        target_id = _text_id(target_agent_id)
        if not target_id or target_id == agent_id or target_id == "system":
            return
        now = time.time()
        position_json = _json_dumps(position) if position is not None else None
        region_json = _json_dumps(region) if region is not None else None
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO person_profiles (
                    agent_id, target_agent_id, name, last_position_json,
                    last_region_json, last_seen_at, confidence, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(agent_id, target_agent_id)
                DO UPDATE SET
                    name = CASE
                        WHEN excluded.name != '' THEN excluded.name
                        ELSE person_profiles.name
                    END,
                    last_position_json = COALESCE(excluded.last_position_json, person_profiles.last_position_json),
                    last_region_json = COALESCE(excluded.last_region_json, person_profiles.last_region_json),
                    last_seen_at = MAX(person_profiles.last_seen_at, excluded.last_seen_at),
                    confidence = MAX(person_profiles.confidence, excluded.confidence),
                    updated_at = excluded.updated_at
                """,
                (
                    agent_id,
                    target_id,
                    name or target_id,
                    position_json,
                    region_json,
                    int(world_time or 0),
                    float(confidence),
                    now,
                    now,
                ),
            )
            self._conn.commit()

    def upsert_person_profile_social(
        self,
        agent_id: str,
        *,
        target_agent_id: Any,
        name: str = "",
        world_time: int,
        recent_post_summary: str = "",
        confidence: float = 0.75,
    ) -> None:
        """记录社交平台看到的人物基础档案，作者和评论者共用同一张表。"""

        target_id = _text_id(target_agent_id)
        if not target_id or target_id == agent_id or target_id == "system":
            return
        now = time.time()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO person_profiles (
                    agent_id, target_agent_id, name, last_social_seen_at,
                    recent_post_summary, confidence, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(agent_id, target_agent_id)
                DO UPDATE SET
                    name = CASE
                        WHEN excluded.name != '' THEN excluded.name
                        ELSE person_profiles.name
                    END,
                    last_social_seen_at = MAX(person_profiles.last_social_seen_at, excluded.last_social_seen_at),
                    recent_post_summary = CASE
                        WHEN excluded.recent_post_summary != '' THEN excluded.recent_post_summary
                        ELSE person_profiles.recent_post_summary
                    END,
                    confidence = MAX(person_profiles.confidence, excluded.confidence),
                    updated_at = excluded.updated_at
                """,
                (
                    agent_id,
                    target_id,
                    name or target_id,
                    int(world_time or 0),
                    recent_post_summary,
                    float(confidence),
                    now,
                    now,
                ),
            )
            self._conn.commit()

    def update_person_profile_impressions(
        self,
        agent_id: str,
        target_agent_id: Any,
        *,
        actions_impression: str,
        opinion_impression: str,
        relationship_impression: str,
        reflected_at: int,
        confidence: float = 0.7,
    ) -> None:
        """在 reflect 阶段更新人物档案的三类印象。"""

        target_id = _text_id(target_agent_id)
        if not target_id or target_id == agent_id or target_id == "system":
            return
        now = time.time()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO person_profiles (
                    agent_id, target_agent_id, name, actions_impression,
                    opinion_impression, relationship_impression,
                    last_reflected_at, confidence, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(agent_id, target_agent_id)
                DO UPDATE SET
                    actions_impression = excluded.actions_impression,
                    opinion_impression = excluded.opinion_impression,
                    relationship_impression = excluded.relationship_impression,
                    last_reflected_at = excluded.last_reflected_at,
                    confidence = MAX(person_profiles.confidence, excluded.confidence),
                    updated_at = excluded.updated_at
                """,
                (
                    agent_id,
                    target_id,
                    target_id,
                    actions_impression,
                    opinion_impression,
                    relationship_impression,
                    int(reflected_at or 0),
                    float(confidence),
                    now,
                    now,
                ),
            )
            self._conn.commit()

    def upsert_relation(
        self,
        agent_id: str,
        *,
        subject_id: Any,
        relation_type: str,
        object_id: Any,
        source_type: str,
        world_time: int,
        payload: Any,
        importance: float = 0.5,
        confidence: float = 0.7,
        valid: bool = True,
    ) -> None:
        """更新实体关系。

        关系由 subject_id、relation_type、object_id 唯一确定，重复观察只刷新
        last_seen_at、payload、重要性和置信度。
        """

        subject_id_text = _text_id(subject_id)
        object_id_text = _text_id(object_id)
        if not subject_id_text or not object_id_text:
            return
        now = time.time()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO entity_relations (
                    agent_id, subject_id, relation_type, object_id, source_type,
                    first_seen_at, last_seen_at, payload_json, importance, confidence,
                    valid, update_count, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                ON CONFLICT(agent_id, subject_id, relation_type, object_id)
                DO UPDATE SET
                    source_type = excluded.source_type,
                    last_seen_at = excluded.last_seen_at,
                    payload_json = excluded.payload_json,
                    importance = MAX(entity_relations.importance, excluded.importance),
                    confidence = MAX(entity_relations.confidence, excluded.confidence),
                    valid = excluded.valid,
                    update_count = entity_relations.update_count + 1,
                    updated_at = excluded.updated_at
                """,
                (
                    agent_id,
                    subject_id_text,
                    relation_type,
                    object_id_text,
                    source_type,
                    int(world_time or 0),
                    int(world_time or 0),
                    _json_dumps(payload),
                    float(importance),
                    float(confidence),
                    1 if valid else 0,
                    now,
                    now,
                ),
            )
            self._conn.commit()

    def upsert_social_post(
        self,
        agent_id: str,
        post: dict[str, Any],
        *,
        source_type: str,
        world_time: int,
        importance: float = 0.55,
        confidence: float = 0.8,
        valid: bool = True,
    ) -> None:
        """保存智能体可见的社交帖子快照。

        同一个 agent_id/post_id 只保留最新互动计数和内容快照，原始浏览和反馈
        仍保存在 memory_events 中。
        """

        post_id = _text_id(post.get("id"))
        if not post_id:
            return
        now = time.time()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO social_posts (
                    agent_id, post_id, author_id, topic, content, post_time, likes,
                    dislikes, reposts, comments_count, opinion_index,
                    is_news, is_rumor, first_seen_at, last_seen_at,
                    source_type, payload_json, importance, confidence, valid,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(agent_id, post_id)
                DO UPDATE SET
                    author_id = excluded.author_id,
                    topic = excluded.topic,
                    content = excluded.content,
                    post_time = excluded.post_time,
                    likes = excluded.likes,
                    dislikes = excluded.dislikes,
                    reposts = excluded.reposts,
                    comments_count = excluded.comments_count,
                    opinion_index = excluded.opinion_index,
                    is_news = excluded.is_news,
                    is_rumor = excluded.is_rumor,
                    last_seen_at = excluded.last_seen_at,
                    source_type = excluded.source_type,
                    payload_json = excluded.payload_json,
                    importance = MAX(social_posts.importance, excluded.importance),
                    confidence = MAX(social_posts.confidence, excluded.confidence),
                    valid = excluded.valid,
                    updated_at = excluded.updated_at
                """,
                (
                    agent_id,
                    post_id,
                    _text_id(post.get("author_id")),
                    _text_id(post.get("topic")),
                    _text_id(post.get("content")),
                    post.get("time"),
                    int(post.get("likes") or 0),
                    int(post.get("dislikes") or 0),
                    int(post.get("reposts") or 0),
                    int(post.get("comments_count") or 0),
                    float(post.get("opinion_index") or 0.0),
                    1 if post.get("is_news") else 0,
                    1 if post.get("is_rumor") else 0,
                    int(world_time or 0),
                    int(world_time or 0),
                    source_type,
                    _json_dumps(post),
                    float(importance),
                    float(confidence),
                    1 if valid else 0,
                    now,
                    now,
                ),
            )
            self._conn.commit()

    def record_scene_snapshot(self, agent_id: str, observation: dict[str, Any]) -> None:
        """按时间步保存一次 observe 的场景摘要和完整结构化观察。"""

        world_time = int(observation.get("time") or 0)
        social = observation.get("social") if isinstance(observation.get("social"), dict) else {}
        now = time.time()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO scene_snapshots (
                    agent_id, world_time, observer_position_json, region_json,
                    people_count, objects_count, actions_count, notifications_count,
                    payload_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(agent_id, world_time)
                DO UPDATE SET
                    observer_position_json = excluded.observer_position_json,
                    region_json = excluded.region_json,
                    people_count = excluded.people_count,
                    objects_count = excluded.objects_count,
                    actions_count = excluded.actions_count,
                    notifications_count = excluded.notifications_count,
                    payload_json = excluded.payload_json,
                    created_at = excluded.created_at
                """,
                (
                    agent_id,
                    world_time,
                    _json_dumps(observation.get("position")),
                    _json_dumps(observation.get("region")),
                    len(observation.get("people") or []),
                    len(observation.get("objects") or []),
                    len(observation.get("actions") or []),
                    len(social.get("notifications") or []),
                    _json_dumps(observation),
                    now,
                ),
            )
            self._conn.commit()

    def record_derived_memory(
        self,
        *,
        memory_id: str,
        agent_id: str,
        memory_type: str,
        source_type: str,
        world_time: int,
        summary: str,
        payload: Any,
        entity_id: Any = None,
        related_agent_id: Any = None,
        object_id: Any = None,
        post_id: Any = None,
        task: Any = None,
        need_key: Any = None,
        importance: float = 0.5,
        confidence: float = 0.5,
        valid: bool = True,
    ) -> None:
        """保存压缩后的长期记忆。

        这里承接 Chroma 写入后的双写结果，也承接不需要 embedding 的反思类记忆。
        对稳定类型记忆做同主体冲突检测，冲突只记录不自动覆盖结论。
        """

        now = time.time()
        with self._lock:
            stable_memory = memory_type in {"semantic", "procedural", "social"}
            subject_id = _text_id(entity_id) or _text_id(related_agent_id) or _text_id(object_id) or _text_id(post_id)
            if stable_memory and subject_id:
                # 稳定事实如果同主体已有不同摘要，先进入冲突表，后续由规则或 LLM 调解。
                existing = self._conn.execute(
                    """
                    SELECT memory_id, summary, payload_json, confidence
                    FROM derived_memories
                    WHERE agent_id = ?
                      AND memory_type = ?
                      AND valid = 1
                      AND COALESCE(entity_id, '') = COALESCE(?, '')
                      AND COALESCE(related_agent_id, '') = COALESCE(?, '')
                      AND COALESCE(object_id, '') = COALESCE(?, '')
                      AND COALESCE(post_id, '') = COALESCE(?, '')
                      AND memory_id != ?
                    ORDER BY confidence DESC, world_time DESC
                    LIMIT 1
                    """,
                    (
                        agent_id,
                        memory_type,
                        _text_id(entity_id) or None,
                        _text_id(related_agent_id) or None,
                        _text_id(object_id) or None,
                        _text_id(post_id) or None,
                        memory_id,
                    ),
                ).fetchone()
                if existing is not None and existing["summary"] != summary:
                    self._record_conflict_locked(
                        agent_id=agent_id,
                        conflict_type="stable_fact_conflict",
                        subject_id=subject_id,
                        old_payload={
                            "memory_id": existing["memory_id"],
                            "summary": existing["summary"],
                            "payload": _json_loads(existing["payload_json"]),
                        },
                        new_payload={
                            "memory_id": memory_id,
                            "summary": summary,
                            "payload": payload,
                        },
                        resolution="",
                        resolved=False,
                        world_time=int(world_time or 0),
                        confidence=min(float(existing["confidence"] or 0.5), float(confidence)),
                        now=now,
                    )
            self._conn.execute(
                """
                INSERT INTO derived_memories (
                    memory_id, agent_id, memory_type, source_type, world_time,
                    entity_id, related_agent_id, object_id, post_id, task,
                    need_key, summary, payload_json, importance, confidence,
                    valid, access_count, last_accessed_at, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?)
                ON CONFLICT(memory_id)
                DO UPDATE SET
                    memory_type = excluded.memory_type,
                    source_type = excluded.source_type,
                    world_time = excluded.world_time,
                    entity_id = excluded.entity_id,
                    related_agent_id = excluded.related_agent_id,
                    object_id = excluded.object_id,
                    post_id = excluded.post_id,
                    task = excluded.task,
                    need_key = excluded.need_key,
                    summary = excluded.summary,
                    payload_json = excluded.payload_json,
                    importance = excluded.importance,
                    confidence = excluded.confidence,
                    valid = excluded.valid,
                    updated_at = excluded.updated_at
                """,
                (
                    memory_id,
                    agent_id,
                    memory_type,
                    source_type,
                    int(world_time or 0),
                    _text_id(entity_id) or None,
                    _text_id(related_agent_id) or None,
                    _text_id(object_id) or None,
                    _text_id(post_id) or None,
                    _text_id(task) or None,
                    _text_id(need_key) or None,
                    summary,
                    _json_dumps(payload),
                    float(importance),
                    float(confidence),
                    1 if valid else 0,
                    int(world_time or 0),
                    now,
                    now,
                ),
            )
            self._conn.commit()

    def get_unresolved_conflicts(self, agent_id: str, limit: int = 20) -> list[dict[str, Any]]:
        """读取尚未解决的冲突，供定期 consolidation 或调试界面使用。"""

        return self._fetch_all(
            """
            SELECT * FROM memory_conflicts
            WHERE agent_id = ? AND resolved = 0
            ORDER BY world_time DESC, id DESC
            LIMIT ?
            """,
            [agent_id, int(limit)],
        )

    def resolve_conflict(
        self,
        conflict_id: int,
        *,
        resolution: str,
        resolved: bool = True,
        invalidate_memory_ids: list[str] | None = None,
    ) -> None:
        """写入冲突调解决议，并按需将被判定失效的衍生记忆标记为 invalid。"""

        now = time.time()
        with self._lock:
            self._conn.execute(
                """
                UPDATE memory_conflicts
                SET resolution = ?, resolved = ?, updated_at = ?
                WHERE id = ?
                """,
                (resolution, 1 if resolved else 0, now, int(conflict_id)),
            )
            for memory_id in invalidate_memory_ids or []:
                self._conn.execute(
                    """
                    UPDATE derived_memories
                    SET valid = 0, updated_at = ?
                    WHERE memory_id = ?
                    """,
                    (now, memory_id),
                )
            self._conn.commit()

    def prune_low_value_events(
        self,
        agent_id: str,
        *,
        before_time: int,
        max_importance: float = 0.3,
        max_confidence: float = 0.5,
    ) -> int:
        """把长期未用且低重要性、低置信度的原始事件标记为失效。"""

        with self._lock:
            cursor = self._conn.execute(
                """
                UPDATE memory_events
                SET valid = 0, updated_at = ?
                WHERE agent_id = ?
                  AND valid = 1
                  AND world_time < ?
                  AND importance <= ?
                  AND confidence <= ?
                """,
                (time.time(), agent_id, int(before_time), float(max_importance), float(max_confidence)),
            )
            self._conn.commit()
            return int(cursor.rowcount or 0)

    def record_conflict(
        self,
        *,
        agent_id: str,
        conflict_type: str,
        subject_id: str,
        old_payload: Any,
        new_payload: Any,
        resolution: str = "",
        resolved: bool = False,
        world_time: int = 0,
        confidence: float = 0.5,
    ) -> int:
        """公开冲突写入入口；内部保持和自动冲突检测相同的字段格式。"""

        now = time.time()
        with self._lock:
            row_id = self._record_conflict_locked(
                agent_id=agent_id,
                conflict_type=conflict_type,
                subject_id=subject_id,
                old_payload=old_payload,
                new_payload=new_payload,
                resolution=resolution,
                resolved=resolved,
                world_time=world_time,
                confidence=confidence,
                now=now,
            )
            self._conn.commit()
            return row_id

    def _record_conflict_locked(
        self,
        *,
        agent_id: str,
        conflict_type: str,
        subject_id: str,
        old_payload: Any,
        new_payload: Any,
        resolution: str,
        resolved: bool,
        world_time: int,
        confidence: float,
        now: float,
    ) -> int:
        """在调用方已经持锁时写入冲突，避免嵌套事务破坏一致性。"""

        cursor = self._conn.execute(
            """
            INSERT INTO memory_conflicts (
                agent_id, conflict_type, subject_id, old_payload_json,
                new_payload_json, resolution, resolved, world_time,
                confidence, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                agent_id,
                conflict_type,
                subject_id,
                _json_dumps(old_payload),
                _json_dumps(new_payload),
                resolution,
                1 if resolved else 0,
                int(world_time or 0),
                float(confidence),
                now,
                now,
            ),
        )
        return int(cursor.lastrowid)

    def record_access(
        self,
        *,
        agent_id: str,
        context: str,
        memory_kind: str,
        memory_ref: Any,
        world_time: int,
        query_text: str = "",
    ) -> None:
        """记录一次结构化记忆检索命中，并更新衍生记忆访问统计。"""

        now = time.time()
        memory_ref_text = _text_id(memory_ref)
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO memory_access_log (
                    agent_id, context, memory_kind, memory_ref, world_time,
                    query_text, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    agent_id,
                    context,
                    memory_kind,
                    memory_ref_text,
                    int(world_time or 0),
                    query_text,
                    now,
                ),
            )
            if memory_kind == "derived":
                self._conn.execute(
                    """
                    UPDATE derived_memories
                    SET access_count = access_count + 1,
                        last_accessed_at = ?,
                        updated_at = ?
                    WHERE memory_id = ?
                    """,
                    (int(world_time or 0), now, memory_ref_text),
                )
            self._conn.commit()

    def get_person_profiles(self, agent_id: str, target_agent_ids: list[Any], limit: int = 20) -> list[dict[str, Any]]:
        """按人物 id 精确读取人物档案，用于 world/social/conversation 召回。"""

        ids = _text_ids(target_agent_ids, exclude_system=True)
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        sql = (
            "SELECT * FROM person_profiles "
            f"WHERE agent_id = ? AND target_agent_id IN ({placeholders}) "
            "ORDER BY MAX(last_seen_at, last_social_seen_at) DESC LIMIT ?"
        )
        return self._fetch_all(sql, [agent_id, *ids, int(limit)])

    def get_recent_person_profiles(self, agent_id: str, limit: int = 10) -> list[dict[str, Any]]:
        """读取近期人物档案，作为缺少明确人物 id 时的补充。"""

        return self._fetch_all(
            """
            SELECT * FROM person_profiles
            WHERE agent_id = ?
            ORDER BY MAX(last_seen_at, last_social_seen_at) DESC, updated_at DESC
            LIMIT ?
            """,
            [agent_id, int(limit)],
        )

    def get_profiles_pending_reflection(self, agent_id: str, limit: int = 5) -> list[dict[str, Any]]:
        """找出已有新事实但尚未在 reflect 阶段更新印象的人物档案。"""

        return self._fetch_all(
            """
            SELECT * FROM person_profiles
            WHERE agent_id = ?
              AND MAX(last_seen_at, last_social_seen_at) > last_reflected_at
            ORDER BY MAX(last_seen_at, last_social_seen_at) DESC, updated_at DESC
            LIMIT ?
            """,
            [agent_id, int(limit)],
        )

    def get_person_profile_facts(
        self,
        agent_id: str,
        target_agent_id: Any,
        *,
        since_time: int = 0,
        limit: int = 12,
    ) -> list[dict[str, Any]]:
        """收集某个人物档案更新所需的新增事实。"""

        target_id = _text_id(target_agent_id)
        if not target_id:
            return []
        return self._fetch_all(
            """
            SELECT * FROM memory_events
            WHERE agent_id = ?
              AND valid = 1
              AND world_time > ?
              AND (
                    entity_id = ?
                 OR related_agent_id = ?
                 OR object_id = ?
                 OR payload_json LIKE ?
              )
            ORDER BY world_time DESC, id DESC
            LIMIT ?
            """,
            [agent_id, int(since_time or 0), target_id, target_id, target_id, f"%{target_id}%", int(limit)],
        )

    def get_entity_states(self, agent_id: str, entity_ids: list[Any], limit: int = 20) -> list[dict[str, Any]]:
        """按实体 id 精确读取当前状态，用于 world/conversation 检索路由。"""

        ids = _text_ids(entity_ids)
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        sql = (
            "SELECT * FROM entity_states "
            f"WHERE agent_id = ? AND entity_id IN ({placeholders}) AND valid = 1 AND entity_type != 'person' "
            "ORDER BY last_seen_at DESC LIMIT ?"
        )
        return self._fetch_all(sql, [agent_id, *ids, int(limit)])

    def get_recent_entity_states(self, agent_id: str, limit: int = 10) -> list[dict[str, Any]]:
        return self._fetch_all(
            """
            SELECT * FROM entity_states
            WHERE agent_id = ? AND valid = 1 AND entity_type != 'person'
            ORDER BY last_seen_at DESC, update_count DESC
            LIMIT ?
            """,
            [agent_id, int(limit)],
        )

    def search_entity_states(
        self,
        agent_id: str,
        *,
        entity_type: str = "",
        kinds: list[Any] | None = None,
        exclude_entity_ids: list[Any] | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """按受控条件查询实体状态；调用方只能传字段值，不能传 SQL。"""

        clauses = ["agent_id = ?", "valid = 1", "entity_type != 'person'"]
        params: list[Any] = [agent_id]
        if entity_type:
            clauses.append("entity_type = ?")
            params.append(_text_id(entity_type))
        kind_texts = [_text_id(value) for value in kinds or [] if _text_id(value)]
        if kind_texts:
            kind_clauses = []
            for kind in kind_texts:
                kind_clauses.append("(payload_json LIKE ? OR name = ? OR entity_id = ?)")
                params.extend([f'%"{kind}"%', kind, kind])
            clauses.append("(" + " OR ".join(kind_clauses) + ")")
        exclude_ids = [_text_id(value) for value in exclude_entity_ids or [] if _text_id(value)]
        if exclude_ids:
            placeholders = ",".join("?" for _ in exclude_ids)
            clauses.append(f"entity_id NOT IN ({placeholders})")
            params.extend(exclude_ids)
        params.append(int(limit))
        sql = (
            "SELECT * FROM entity_states WHERE "
            + " AND ".join(clauses)
            + " ORDER BY importance DESC, confidence DESC, last_seen_at DESC LIMIT ?"
        )
        return self._fetch_all(sql, params)

    def get_social_posts(self, agent_id: str, post_ids: list[Any], limit: int = 20) -> list[dict[str, Any]]:
        ids = _text_ids(post_ids)
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        sql = (
            "SELECT * FROM social_posts "
            f"WHERE agent_id = ? AND post_id IN ({placeholders}) AND valid = 1 "
            "ORDER BY last_seen_at DESC LIMIT ?"
        )
        return self._fetch_all(sql, [agent_id, *ids, int(limit)])

    def get_recent_social_posts(self, agent_id: str, limit: int = 10) -> list[dict[str, Any]]:
        return self._fetch_all(
            """
            SELECT * FROM social_posts
            WHERE agent_id = ? AND valid = 1
            ORDER BY last_seen_at DESC
            LIMIT ?
            """,
            [agent_id, int(limit)],
        )

    def get_social_posts_by_authors(self, agent_id: str, author_ids: list[Any], limit: int = 10) -> list[dict[str, Any]]:
        """按作者读取近期帖子，用于社交 planner 查询人物相关线上内容。"""

        ids = _text_ids(author_ids, exclude_system=True)
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        sql = (
            "SELECT * FROM social_posts "
            f"WHERE agent_id = ? AND author_id IN ({placeholders}) AND valid = 1 "
            "ORDER BY last_seen_at DESC LIMIT ?"
        )
        return self._fetch_all(sql, [agent_id, *ids, int(limit)])

    def get_recent_events(
        self,
        agent_id: str,
        *,
        source_types: list[str] | None = None,
        memory_types: list[str] | None = None,
        entity_ids: list[Any] | None = None,
        post_ids: list[Any] | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """按来源、记忆类型、实体和帖子过滤近期原始事件。"""

        clauses = ["agent_id = ?", "valid = 1"]
        params: list[Any] = [agent_id]
        if source_types:
            placeholders = ",".join("?" for _ in source_types)
            clauses.append(f"source_type IN ({placeholders})")
            params.extend(source_types)
        if memory_types:
            placeholders = ",".join("?" for _ in memory_types)
            clauses.append(f"memory_type IN ({placeholders})")
            params.extend(memory_types)
        entity_texts = [_text_id(value) for value in entity_ids or [] if _text_id(value)]
        if entity_texts:
            placeholders = ",".join("?" for _ in entity_texts)
            clauses.append(
                f"(entity_id IN ({placeholders}) OR related_agent_id IN ({placeholders}) OR object_id IN ({placeholders}))"
            )
            params.extend(entity_texts)
            params.extend(entity_texts)
            params.extend(entity_texts)
        post_texts = [_text_id(value) for value in post_ids or [] if _text_id(value)]
        if post_texts:
            placeholders = ",".join("?" for _ in post_texts)
            clauses.append(f"post_id IN ({placeholders})")
            params.extend(post_texts)
        params.append(int(limit))
        sql = (
            "SELECT * FROM memory_events WHERE "
            + " AND ".join(clauses)
            + " ORDER BY world_time DESC, id DESC LIMIT ?"
        )
        return self._fetch_all(sql, params)

    def get_recent_derived(
        self,
        agent_id: str,
        *,
        memory_types: list[str] | None = None,
        task: str = "",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """读取近期衍生记忆，优先返回高重要性和高置信度内容。"""

        clauses = ["agent_id = ?", "valid = 1"]
        params: list[Any] = [agent_id]
        if memory_types:
            placeholders = ",".join("?" for _ in memory_types)
            clauses.append(f"memory_type IN ({placeholders})")
            params.extend(memory_types)
        if task:
            clauses.append("(task = ? OR task IS NULL)")
            params.append(task)
        params.append(int(limit))
        sql = (
            "SELECT * FROM derived_memories WHERE "
            + " AND ".join(clauses)
            + " ORDER BY importance DESC, confidence DESC, world_time DESC LIMIT ?"
        )
        return self._fetch_all(sql, params)

    def get_recent_relations(
        self,
        agent_id: str,
        *,
        subject_ids: list[Any] | None = None,
        object_ids: list[Any] | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """读取和指定人物、物品或帖子相关的近期关系。"""

        clauses = ["agent_id = ?", "valid = 1"]
        params: list[Any] = [agent_id]
        subject_texts = [_text_id(value) for value in subject_ids or [] if _text_id(value)]
        if subject_texts:
            placeholders = ",".join("?" for _ in subject_texts)
            clauses.append(f"subject_id IN ({placeholders})")
            params.extend(subject_texts)
        object_texts = [_text_id(value) for value in object_ids or [] if _text_id(value)]
        if object_texts:
            placeholders = ",".join("?" for _ in object_texts)
            clauses.append(f"object_id IN ({placeholders})")
            params.extend(object_texts)
        params.append(int(limit))
        sql = (
            "SELECT * FROM entity_relations WHERE "
            + " AND ".join(clauses)
            + " ORDER BY last_seen_at DESC, update_count DESC LIMIT ?"
        )
        return self._fetch_all(sql, params)

    def list_agent_memory(self, agent_id: str, limit: int = 200) -> dict[str, list[dict[str, Any]]]:
        """按表分类列出某个智能体的结构化记忆，供 API 和调试界面展示。"""

        return {
            "events": self._fetch_all(
                """
                SELECT * FROM memory_events
                WHERE agent_id = ?
                ORDER BY world_time DESC, id DESC
                LIMIT ?
                """,
                [agent_id, int(limit)],
            ),
            "entity_states": self._fetch_all(
                """
                SELECT * FROM entity_states
                WHERE agent_id = ?
                ORDER BY last_seen_at DESC, entity_type, entity_id
                LIMIT ?
                """,
                [agent_id, int(limit)],
            ),
            "person_profiles": self._fetch_all(
                """
                SELECT * FROM person_profiles
                WHERE agent_id = ?
                ORDER BY MAX(last_seen_at, last_social_seen_at) DESC, target_agent_id
                LIMIT ?
                """,
                [agent_id, int(limit)],
            ),
            "entity_relations": self._fetch_all(
                """
                SELECT * FROM entity_relations
                WHERE agent_id = ?
                ORDER BY last_seen_at DESC, id DESC
                LIMIT ?
                """,
                [agent_id, int(limit)],
            ),
            "social_posts": self._fetch_all(
                """
                SELECT * FROM social_posts
                WHERE agent_id = ?
                ORDER BY last_seen_at DESC, post_id
                LIMIT ?
                """,
                [agent_id, int(limit)],
            ),
            "scene_snapshots": self._fetch_all(
                """
                SELECT * FROM scene_snapshots
                WHERE agent_id = ?
                ORDER BY world_time DESC
                LIMIT ?
                """,
                [agent_id, int(limit)],
            ),
            "derived_memories": self._fetch_all(
                """
                SELECT * FROM derived_memories
                WHERE agent_id = ?
                ORDER BY world_time DESC, memory_type
                LIMIT ?
                """,
                [agent_id, int(limit)],
            ),
            "memory_conflicts": self._fetch_all(
                """
                SELECT * FROM memory_conflicts
                WHERE agent_id = ?
                ORDER BY world_time DESC, id DESC
                LIMIT ?
                """,
                [agent_id, int(limit)],
            ),
            "memory_access_log": self._fetch_all(
                """
                SELECT * FROM memory_access_log
                WHERE agent_id = ?
                ORDER BY world_time DESC, id DESC
                LIMIT ?
                """,
                [agent_id, int(limit)],
            ),
        }

    def _fetch_all(self, sql: str, params: list[Any]) -> list[dict[str, Any]]:
        """执行只读查询并统一转换为普通 dict。"""

        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        """把 *_json 字段解码为同名去后缀字段，保留原始 JSON 文本便于追踪。"""

        data = dict(row)
        for key in list(data.keys()):
            if key.endswith("_json"):
                decoded_key = key[:-5]
                data[decoded_key] = _json_loads(data.get(key))
        return data
