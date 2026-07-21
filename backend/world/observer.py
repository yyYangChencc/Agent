from __future__ import annotations

from typing import Any

from world.event import SpeakingEvent
from persona.logger import get_logger

logger = get_logger(__name__)


Observation = dict[str, Any]


def observe(agent, r: int) -> Observation:
    people: list[dict[str, Any]] = []
    objects: list[dict[str, Any]] = []
    x, y = agent.position

    for i in range(max(0, x - r), min(agent.world.map.height, x + r)):
        for j in range(max(0, y - r), min(agent.world.map.width, y + r)):
            if agent.world.map.is_empty(i, j):
                continue
            obj_id = agent.world.map.get_e(i, j)
            if obj_id == agent.id:
                continue
            if obj_id in agent.world.agents:
                other = agent.world.agents[obj_id]
                people.append(_agent_to_observation(other))
            elif obj_id in agent.world.objects:
                obj = agent.world.objects[obj_id]
                objects.append(_object_to_observation(obj))

    actions = [_event_to_observation(agent, event) for event in agent.observed_events]
    agent.observed_events.clear()

    raw_notifications = list(agent._pending_social_notifications)
    notification_events = [
        _structured_notification(agent, note)
        for note in raw_notifications
    ]
    notifications = [
        {
            "type": "notification",
            "content": event["content"],
            "time": agent.world.time,
        }
        for event in notification_events
    ]
    agent._pending_social_notifications.clear()

    observation = {
        "schema_version": 1,
        "observer_id": agent.id,
        "episode_id": str(getattr(agent, "_current_episode_id", "") or ""),
        "time": agent.world.time,
        "radius": r,
        "position": list(agent.position),
        "region": _region_for_agent(agent),
        "people": people,
        "objects": objects,
        "actions": actions,
        "social": {
            "notifications": notifications,
            "notification_events": notification_events,
        },
    }
    agent.observation = observation
    logger.debug("[%s] structured observation: %s", agent.id, observation)
    return observation


def _structured_notification(agent, note) -> dict[str, Any]:
    """兼容旧字符串通知，并展开平台事件中的常用关联字段。"""

    if isinstance(note, dict):
        event = dict(note)
        details = dict(event.get("details") or {})
    else:
        event = {"content": str(note or "")}
        details = {}

    platform_event_id = str(event.get("platform_event_id") or event.get("event_id") or "")
    actor_id = str(event.get("actor_id") or "")
    source_author_id = event.get("source_author_id")
    if source_author_id in (None, ""):
        source_author_id = details.get("source_author_id")
    event_time = event.get("time")
    if event_time is None:
        event_time = event.get("tick")
    if event_time is None:
        event_time = agent.world.time

    event.update(
        {
            "schema_version": int(event.get("schema_version") or 1),
            "type": "notification",
            "content": str(event.get("content") or ""),
            "time": event_time,
            "episode_id": str(
                event.get("episode_id")
                or getattr(agent, "_current_episode_id", "")
                or ""
            ),
            "event_id": str(event.get("event_id") or platform_event_id),
            "event_type": str(event.get("event_type") or "notification"),
            "platform_event_id": platform_event_id,
            "feed_request_id": str(event.get("feed_request_id") or ""),
            "actor_id": actor_id,
            "related_agent_id": str(event.get("related_agent_id") or actor_id or source_author_id or ""),
            "post_id": event.get("post_id"),
            "comment_id": event.get("comment_id", details.get("comment_id")),
            "parent_comment_id": event.get("parent_comment_id", details.get("parent_comment_id")),
            "root_comment_id": event.get("root_comment_id", details.get("root_comment_id")),
            "target_agent_id": str(event.get("target_agent_id") or agent.id),
            "source_post_id": event.get("source_post_id", details.get("source_post_id")),
            "root_post_id": event.get("root_post_id", details.get("root_post_id")),
            "source_author_id": source_author_id,
            "topic": str(event.get("topic") or details.get("topic") or ""),
            "details": details,
        }
    )
    return event


def _agent_to_observation(agent) -> dict[str, Any]:
    return {
        "id": agent.id,
        "name": agent.id,
        "position": list(agent.position),
        "region": _region_for_agent(agent),
        "inside_building_id": agent.inside_building_id,
    }


def _object_to_observation(obj) -> dict[str, Any]:
    return {
        "id": obj.id,
        "name": obj.id,
        "kind": getattr(obj, "kind", type(obj).__name__),
        "type": type(obj).__name__,
        "position": list(obj.position),
        "region": _region_for_object(obj),
        "owner_agent_id": getattr(obj, "owner_agent_id", None),
        "free_num": getattr(obj, "free_num", None),
        "occupant_id": getattr(obj, "occupant_id", None),
        "description": obj.get_desc() if hasattr(obj, "get_desc") else "",
    }


def _event_to_observation(observer, event) -> dict[str, Any]:
    data = {
        "type": event.type,
        "actor_id": event.actor,
        "acted_id": event.acted,
        "info": event.info,
        "time": event.time,
        "position": list(event.position) if event.position is not None else None,
        "region": _region_for_position(observer.world, event.position),
    }
    if isinstance(event, SpeakingEvent):
        data["response_to"] = event.response_to
    return data


def _region_for_agent(agent) -> dict[str, Any] | None:
    if agent.inside_building_id:
        building_obj = agent.world.objects.get(agent.inside_building_id)
        if building_obj is not None:
            region = _region_for_object(building_obj)
            if region is not None:
                return region
    return _region_for_position(agent.world, agent.position)


def _region_for_object(obj) -> dict[str, Any] | None:
    design = getattr(obj.world, "map_design", None)
    if isinstance(design, dict):
        object_regions = design.get("object_regions", {})
        if isinstance(object_regions, dict):
            region_info = object_regions.get(obj.id)
            if isinstance(region_info, dict):
                region_id = region_info.get("region_id")
                region = _region_by_id(obj.world, region_id)
                if region is not None:
                    return region
    return _region_for_position(obj.world, obj.position)


def _region_for_position(world, position) -> dict[str, Any] | None:
    if position is None:
        return None
    row, col = position
    design = getattr(world, "map_design", None)
    if not isinstance(design, dict):
        return None
    for region in design.get("regions", []):
        bounds = region.get("bounds")
        if not isinstance(bounds, list) or len(bounds) != 4:
            continue
        row_start, col_start, row_end, col_end = bounds
        if row_start <= row <= row_end and col_start <= col <= col_end:
            return _public_region(region)
    return None


def _region_by_id(world, region_id) -> dict[str, Any] | None:
    if not region_id:
        return None
    design = getattr(world, "map_design", None)
    if not isinstance(design, dict):
        return None
    for region in design.get("regions", []):
        if region.get("id") == region_id:
            return _public_region(region)
    return None


def _public_region(region: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": region.get("id"),
        "name": region.get("name"),
        "kind": region.get("kind"),
    }
