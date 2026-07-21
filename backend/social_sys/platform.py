import json
import threading
from contextlib import ExitStack

from tools.operator_tools import SocialOperator, register_operator_tools
from social_sys.post import Post
from persona.logger import get_logger
from persona.opinion.scale import OPINION_NEUTRAL, clamp_opinion

logger = get_logger(__name__)


class SocialPlatform:
    """Social platform state and action execution."""

    def __init__(self, llm=None, config=None):
        self.agents = {}
        self.influencers = {}
        self.posts = []
        self.time = 0
        self.llm = llm
        self.config = config
        self._posts_lock = threading.RLock()
        self._action_lock = threading.RLock()
        self._event_lock = threading.RLock()
        self._next_post_id = 1
        self._next_event_id = 1
        self._next_feed_request_id = 1
        self.exposure_events: list[dict] = []
        self.events: list[dict] = []
        self.social_operator = SocialOperator(self)
        self.tools, self.tools_prompt = register_operator_tools(self.social_operator)

    def get_agent(self, agent_id):
        return self.agents.get(agent_id, None)

    def account_ids(self, *, exclude_id: str = "") -> list[str]:
        """返回平台已注册账号的精确 ID，作为关注动作的受控目标列表。"""

        account_ids = set(self.agents) | set(self.influencers)
        if exclude_id:
            account_ids.discard(exclude_id)
        return sorted(account_ids)

    def add_agent(self, agent):
        self.agents[agent.id] = agent

    def add_influencer(self, influencer: dict) -> None:
        """注册只有线上身份、没有地图实体的投放者账号。"""

        influencer_id = str(influencer.get("id") or "").strip()
        if not influencer_id:
            raise ValueError("influencer id is required")
        data = dict(influencer)
        data["id"] = influencer_id
        self.influencers[influencer_id] = data

    def add_post(self, post):
        with self._posts_lock:
            self.posts.append(post)
            post_id = getattr(post, "id", None)
            if isinstance(post_id, int) and not isinstance(post_id, bool):
                self._next_post_id = max(self._next_post_id, post_id + 1)

    def allocate_post_id(self) -> int:
        """集中分配帖子 ID，兼容测试直接预置的帖子。"""

        with self._posts_lock:
            existing_ids = [
                post.id
                for post in self.posts
                if isinstance(getattr(post, "id", None), int)
                and not isinstance(getattr(post, "id", None), bool)
            ]
            post_id = max(self._next_post_id, max(existing_ids, default=0) + 1)
            self._next_post_id = post_id + 1
            return post_id

    def record_event(
        self,
        event_type: str,
        *,
        actor_id: str = "",
        post_id=None,
        target_agent_id: str = "",
        feed_request_id: str = "",
        details: dict | None = None,
    ) -> dict:
        """追加统一平台事件，供传播链和实验审计回放。"""

        event_type = str(event_type or "").strip()
        if not event_type:
            raise ValueError("event_type is required")
        with self._event_lock:
            event = {
                "schema_version": 1,
                "event_id": f"event-{self._next_event_id:08d}",
                "event_type": event_type,
                "tick": int(self.time),
                "actor_id": str(actor_id or ""),
                "post_id": post_id,
                "target_agent_id": str(target_agent_id or ""),
                "feed_request_id": str(feed_request_id or ""),
                "details": dict(details or {}),
            }
            self._next_event_id += 1
            self.events.append(event)
            return dict(event)

    def _allocate_feed_request_id(self) -> str:
        """为一次完整浏览分配稳定标识。"""

        with self._event_lock:
            request_id = f"feed-{self._next_feed_request_id:08d}"
            self._next_feed_request_id += 1
            return request_id

    def get_visible_posts(self, receiver_id, limit=None):
        stages = self._build_feed_stages(receiver_id, limit)
        return stages["displayed"]

    def _build_feed_stages(self, receiver_id, limit=None) -> dict:
        """返回可见集合、排序结果和最终展示结果，供曝光审计复用。"""

        receiver = self.get_agent(receiver_id)
        if not receiver:
            return {"eligible": [], "ranked": [], "displayed": []}
        with self._posts_lock:
            posts = [
                post
                for post in self.posts
                if post.author_id in receiver.followers or post.is_news
            ]
        eligible = list(posts)
        if self._recommendation_enabled():
            posts = self._apply_recommendation_placeholder(receiver_id, posts)
        ranked = list(posts)
        try:
            limit_value = int(limit) if limit is not None else 0
        except (TypeError, ValueError):
            limit_value = 0
        if limit_value > 0:
            # 只限制帖子数量；post.to_dict() 会保留该帖子的全部评论。
            posts = posts[-limit_value:]
        return {"eligible": eligible, "ranked": ranked, "displayed": list(posts)}

    def _recommendation_enabled(self) -> bool:
        """读取推荐系统配置开关；当前默认关闭。"""

        if self.config is None:
            return False
        return bool(getattr(self.config, "social_recommendation_enabled", False))

    def _apply_recommendation_placeholder(self, receiver_id, posts):
        """推荐算法暂不实现；启用开关时先保持关注流原样。"""

        logger.debug("[SocialPlatform] recommendation placeholder enabled for %s", receiver_id)
        return posts

    def browse(self, receiver_id, limit=None) -> tuple[list, dict]:
        """一次性生成可互动帖子、浏览载荷和曝光审计事件。"""

        if limit is None and self.config is not None:
            limit = getattr(self.config, "social_visible_post_limit", None)
        stages = self._build_feed_stages(receiver_id, limit=limit)
        posts = stages["displayed"]
        feed_request_id = self._allocate_feed_request_id()
        feed_mode = "recommendation_placeholder" if self._recommendation_enabled() else "follow_chronological"
        exposure = {
            "tick": self.time,
            "receiver_id": receiver_id,
            "feed_request_id": feed_request_id,
            "feed_mode": feed_mode,
            "limit": limit,
            "eligible_post_ids": [post.id for post in stages["eligible"]],
            "ranked_post_ids": [post.id for post in stages["ranked"]],
            "displayed_post_ids": [post.id for post in posts],
        }
        event = self.record_event(
            "feed_impression",
            actor_id=receiver_id,
            feed_request_id=feed_request_id,
            details={
                "feed_mode": feed_mode,
                "limit": limit,
                "eligible_post_ids": list(exposure["eligible_post_ids"]),
                "ranked_post_ids": list(exposure["ranked_post_ids"]),
                "displayed_post_ids": list(exposure["displayed_post_ids"]),
            },
        )
        exposure["event_id"] = event["event_id"]
        with self._event_lock:
            self.exposure_events.append(exposure)
        payload = {
            "schema_version": 1,
            "type": "social_browse",
            "receiver_id": receiver_id,
            "time": self.time,
            "feed_request_id": feed_request_id,
            "platform_event_id": event["event_id"],
            "account_ids": self.account_ids(exclude_id=receiver_id),
            "following_ids": list(getattr(self.get_agent(receiver_id), "followers", []) or []),
            "visible_post_ids": [post.id for post in posts],
            "posts": [post.to_dict() for post in posts],
        }
        return list(posts), payload

    def give_post(self, receiver_id, limit=None):
        _, payload = self.browse(receiver_id, limit=limit)
        return payload

    def inject_news(
        self,
        tick: int,
        title: str,
        content: str,
        topic: str = "",
        opinion_index: float = OPINION_NEUTRAL,
    ):
        post_id = self.allocate_post_id()
        news_post = Post(post_id, "system", f"【{title}】{content}", is_news=True, topic=topic)
        news_post.opinion_index = clamp_opinion(opinion_index)
        news_post.time = tick
        self.add_post(news_post)
        event = self.record_event(
            "official_news_created",
            actor_id="system",
            post_id=post_id,
            details={
                "title": title,
                "content": news_post.content,
                "topic": topic,
                "opinion_index": news_post.opinion_index,
                "is_news": True,
                "is_rumor": False,
                "source_type": news_post.source_type,
            },
        )
        # 保留创建事件标识，供世界通知与平台账本建立精确关联。
        news_post.platform_event_id = event["event_id"]
        logger.info("[News] tick=%d inject news: %s", tick, title)
        return news_post

    def inject_influencer_post(
        self,
        tick: int,
        author_id: str,
        content: str,
        topic: str = "",
        opinion_index: float = OPINION_NEUTRAL,
        is_rumor: bool = False,
    ):
        """按场景排期发布投放者帖子，投放者不进入世界行动循环。"""

        author_id = str(author_id or "").strip()
        if not author_id:
            raise ValueError("influencer post author_id is required")
        post_id = self.allocate_post_id()
        post = Post(
            post_id,
            author_id,
            str(content or ""),
            is_rumor=is_rumor,
            is_news=False,
            topic=topic,
            source_type="influencer",
        )
        post.opinion_index = clamp_opinion(opinion_index)
        post.time = tick
        self.add_post(post)
        self.record_event(
            "influencer_post_created",
            actor_id=author_id,
            post_id=post_id,
            details={
                "content": post.content,
                "topic": topic,
                "opinion_index": post.opinion_index,
                "is_news": False,
                "is_rumor": bool(is_rumor),
                "source_type": post.source_type,
            },
        )
        logger.info("[Influencer] tick=%d author=%s post=%s", tick, author_id, post_id)
        return post

    def execute(self, Operator_id, action_str):
        if not action_str:
            return {
                "ok": True,
                "action": None,
                "feedback": "",
            }
        try:
            decision = json.loads(action_str)
        except json.JSONDecodeError:
            logger.error("[SocialPlatform] invalid JSON action: %r", action_str)
            return self._failure_result(Operator_id, None, "invalid social action JSON")
        data, think = self._action_from_decision(decision)
        if not data or "tool" not in data:
            logger.debug("[SocialPlatform] %s selected no action", Operator_id)
            return {
                "ok": True,
                "action": None,
                "think": think,
                "feedback": "",
            }
        tool_name = data.get("tool")
        if not isinstance(tool_name, str) or not tool_name:
            logger.warning("[SocialPlatform] invalid tool name: %r", tool_name)
            return self._failure_result(Operator_id, None, "invalid social tool name", think=think)
        tool = self.tools.get(tool_name, None)
        if not tool:
            logger.warning("[SocialPlatform] unknown tool: %s", tool_name)
            return self._failure_result(Operator_id, tool_name, "tool not found", think=think)

        args = data.get("args", {})
        if not isinstance(args, dict):
            logger.warning("[SocialPlatform] invalid tool args (tool=%s): %r", tool_name, args)
            return self._failure_result(Operator_id, tool_name, "tool args must be an object", think=think)
        args = dict(args)
        args["operator_ID"] = Operator_id
        agent = self.get_agent(Operator_id)
        world_lock = getattr(getattr(agent, "world", None), "_world_lock", None) if agent is not None else None
        with ExitStack() as locks:
            # 与世界动作共用状态锁；平台锁只覆盖快速提交，不影响并发生成 LLM 决策。
            if world_lock is not None:
                locks.enter_context(world_lock)
            locks.enter_context(self._action_lock)
            if agent is not None:
                # 每次执行只读取本次工具写入的动作快照，避免异常路径误用上次成功结果。
                agent.last_social_action = {}
            with self._event_lock:
                event_start_index = len(self.events)
            try:
                feedback = tool.run(**args)
            except Exception as e:
                committed = self._committed_action_event(
                    Operator_id,
                    tool_name,
                    event_start_index=event_start_index,
                )
                if committed is not None:
                    logger.error(
                        "[SocialPlatform] social action committed but post-processing failed "
                        "(tool=%s operator=%s): %s",
                        tool_name,
                        Operator_id,
                        e,
                        exc_info=True,
                    )
                    social_action, committed_event = committed
                    return self._postprocess_failure_result(
                        Operator_id,
                        tool_name,
                        args,
                        think,
                        social_action,
                        committed_event,
                    )
                logger.error(
                    "[SocialPlatform] tool execution failed (tool=%s operator=%s): %s",
                    tool_name,
                    Operator_id,
                    e,
                    exc_info=True,
                )
                return self._failure_result(
                    Operator_id,
                    tool_name,
                    "tool execution failed",
                    think=think,
                    post_id=args.get("post_id"),
                )
            if isinstance(feedback, dict):
                feedback.setdefault("think", think)
                result = feedback
            else:
                result = self._structured_feedback(tool_name, Operator_id, args, feedback, think)
            if not result.get("ok"):
                # 已带事件标识的失败结果已经入账，避免重复记录同一次失败。
                if not result.get("platform_event_id"):
                    failed_event = self._record_failed_action_event(
                        Operator_id,
                        tool_name,
                        result.get("feedback"),
                        post_id=result.get("post_id"),
                    )
                    result["platform_event_id"] = failed_event["event_id"]
                    if failed_event.get("feed_request_id"):
                        result.setdefault("feed_request_id", failed_event["feed_request_id"])
            return result

    def _failure_result(self, operator_id, action, feedback, *, think: str = "", post_id=None) -> dict:
        """统一构造失败反馈并写入平台事件账本。"""

        result = {
            "ok": False,
            "action": action,
            "think": think,
            "feedback": str(feedback or ""),
        }
        agent = self.get_agent(operator_id)
        feed_request_id = str(getattr(agent, "_last_feed_request_id", "") or "") if agent is not None else ""
        if feed_request_id:
            result["feed_request_id"] = feed_request_id
        if post_id is not None:
            result["post_id"] = post_id
        failed_event = self._record_failed_action_event(operator_id, action, feedback, post_id=post_id)
        result["platform_event_id"] = failed_event["event_id"]
        return result

    def _committed_action_event(
        self,
        operator_id: str,
        action: str,
        *,
        event_start_index: int,
    ) -> tuple[dict, dict] | None:
        """用本次动作快照和账本事件共同确认核心动作已经提交。"""

        agent = self.get_agent(operator_id)
        social_action = dict(getattr(agent, "last_social_action", {}) or {}) if agent is not None else {}
        event_id = str(social_action.get("platform_event_id") or "")
        if social_action.get("action") != action or not event_id:
            return None
        with self._event_lock:
            committed_event = next(
                (
                    dict(event)
                    for event in reversed(self.events[event_start_index:])
                    if event.get("event_id") == event_id
                    and event.get("event_type") == action
                    and event.get("actor_id") == operator_id
                ),
                None,
            )
        if committed_event is None:
            return None
        return social_action, committed_event

    def _postprocess_failure_result(
        self,
        operator_id: str,
        action: str,
        args: dict,
        think: str,
        social_action: dict,
        committed_event: dict,
    ) -> dict:
        """核心动作已提交时，单独记录后处理失败而不否定状态变更。"""

        feedback = "social action committed; post-processing failed"
        postprocess_event = self.record_event(
            "social_action_postprocess_failed",
            actor_id=operator_id,
            post_id=social_action.get("post_id"),
            target_agent_id=str(social_action.get("target_agent_id") or ""),
            feed_request_id=str(committed_event.get("feed_request_id") or ""),
            details={
                "action": action,
                "committed_event_id": committed_event["event_id"],
                "feedback": "post-processing failed",
            },
        )
        result = self._structured_feedback(action, operator_id, args, feedback, think)
        result["ok"] = True
        result["side_effects_ok"] = False
        result["postprocess_event_id"] = postprocess_event["event_id"]
        return result

    def _record_failed_action_event(self, operator_id, action, feedback, *, post_id=None) -> dict:
        """记录所有失败入口，保证解析错误和工具异常也可审计。"""

        agent = self.get_agent(operator_id)
        return self.record_event(
            "social_action_failed",
            actor_id=operator_id,
            post_id=post_id,
            feed_request_id=str(getattr(agent, "_last_feed_request_id", "") or "") if agent is not None else "",
            details={
                "action": str(action or ""),
                "feedback": str(feedback or ""),
            },
        )

    def _structured_feedback(self, action: str, operator_id: str, args: dict, feedback, think: str = ""):
        ok = not (isinstance(feedback, str) and feedback.startswith("error:"))
        creation_actions = {"send_post", "repost_post", "quote_post"}
        # 创建动作失败时不读取上一次成功动作，避免回填旧帖子和旧动作字段。
        post = None if action in creation_actions and not ok else self._post_from_action(action, operator_id, args)
        if action in {"comment_post", "reply_comment", "like_post", "dislike_post"} and post is None:
            ok = False
        result = {
            "ok": ok,
            "action": action,
            "think": think,
            "feedback": feedback or "",
        }
        agent = self.get_agent(operator_id)
        feed_request_id = str(getattr(agent, "_last_feed_request_id", "") or "") if agent is not None else ""
        if feed_request_id:
            result["feed_request_id"] = feed_request_id
        if ok and agent is not None:
            social_action = getattr(agent, "last_social_action", {}) or {}
            if social_action.get("action") == action:
                for field in (
                    "state_changed",
                    "previous_reaction",
                    "current_reaction",
                    "source_post_id",
                    "root_post_id",
                    "source_author_id",
                    "target_agent_id",
                    "comment_id",
                    "parent_comment_id",
                    "root_comment_id",
                    "post_topic",
                    "platform_event_id",
                    "online_trust_before",
                    "online_trust_after",
                ):
                    if field in social_action:
                        result[field] = social_action[field]
                if social_action.get("feed_request_id"):
                    result["feed_request_id"] = social_action["feed_request_id"]
        if post is not None:
            result["post_id"] = post.id
            result["post"] = post.to_dict()
        elif "post_id" in args:
            result["post_id"] = args.get("post_id")
            if action in {"repost_post", "quote_post"}:
                result["source_post_id"] = args.get("post_id")
        return result

    def _post_from_action(self, action: str, operator_id: str, args: dict):
        if action in {"send_post", "repost_post", "quote_post"}:
            agent = self.get_agent(operator_id)
            social_action = getattr(agent, "last_social_action", {}) or {} if agent is not None else {}
            created_post_id = social_action.get("post_id") if social_action.get("action") == action else None
            with self._posts_lock:
                for post in reversed(self.posts):
                    if post.author_id == operator_id and (created_post_id is None or post.id == created_post_id):
                        return post
            return None
        post_id = args.get("post_id")
        if post_id is None:
            return None
        if isinstance(post_id, bool):
            return None
        if isinstance(post_id, int):
            normalized_post_id = post_id
        elif isinstance(post_id, str) and post_id.strip().isdigit():
            normalized_post_id = int(post_id.strip())
        else:
            return None
        agent = self.get_agent(operator_id)
        visible_posts = list(getattr(agent, "_last_seen_posts", []) or []) if agent is not None else []
        return next((post for post in visible_posts if post.id == normalized_post_id), None)

    def _action_from_decision(self, decision):
        if not isinstance(decision, dict):
            return {}, ""
        think = str(decision.get("think", "") or "")
        action = decision.get("action", {})
        if not isinstance(action, dict):
            return {}, think
        return action, think
