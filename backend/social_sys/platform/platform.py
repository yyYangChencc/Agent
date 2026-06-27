import json
import threading

from tools.operator_tools import SocialOperator, register_operator_tools
from social_sys.post.post import Post
from persona.logger import get_logger
from persona.opinion.scale import OPINION_NEUTRAL, clamp_opinion

logger = get_logger(__name__)


class SocialPlatform:
    """Social platform state and action execution."""

    def __init__(self):
        self.agents = {}
        self.posts = []
        self.time = 0
        self._posts_lock = threading.Lock()
        self.social_operator = SocialOperator(self)
        self.tools, self.tools_prompt = register_operator_tools(self.social_operator)

    def get_agent(self, agent_id):
        return self.agents.get(agent_id, None)

    def add_agent(self, agent):
        self.agents[agent.id] = agent

    def add_post(self, post):
        self.posts.append(post)

    def get_visible_posts(self, receiver_id):
        receiver = self.get_agent(receiver_id)
        if not receiver:
            return []
        return [
            post
            for post in self.posts
            if post.author_id in receiver.followers or post.is_news
        ]

    def give_post(self, receiver_id):
        posts = self.get_visible_posts(receiver_id)
        return {
            "schema_version": 1,
            "type": "social_browse",
            "receiver_id": receiver_id,
            "time": self.time,
            "visible_post_ids": [post.id for post in posts],
            "posts": [post.to_dict() for post in posts],
        }

    def inject_news(
        self,
        tick: int,
        title: str,
        content: str,
        opinion_index: float = OPINION_NEUTRAL,
    ):
        post_id = len(self.posts) + 1
        news_post = Post(post_id, "system", f"【{title}】{content}", is_news=True)
        news_post.opinion_index = clamp_opinion(opinion_index)
        news_post.time = tick
        self.posts.append(news_post)
        logger.info("[News] tick=%d inject news: %s", tick, title)
        return news_post

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
            return {
                "ok": False,
                "action": None,
                "think": "",
                "feedback": "invalid social action JSON",
            }
        data, think = self._action_from_decision(decision)
        if not data or "tool" not in data:
            logger.debug("[SocialPlatform] %s selected no action", Operator_id)
            return {
                "ok": True,
                "action": None,
                "think": think,
                "feedback": "",
            }
        tool = self.tools.get(data["tool"], None)
        if not tool:
            logger.warning("[SocialPlatform] unknown tool: %s", data.get("tool"))
            return {
                "ok": False,
                "action": data.get("tool"),
                "think": think,
                "feedback": "tool not found",
            }

        args = data.get("args", {})
        args["operator_ID"] = Operator_id
        try:
            feedback = tool.run(**args)
        except Exception as e:
            logger.error(
                "[SocialPlatform] tool execution failed (tool=%s operator=%s): %s",
                data["tool"],
                Operator_id,
                e,
                exc_info=True,
            )
            return {
                "ok": False,
                "action": data["tool"],
                "think": think,
                "feedback": "tool execution failed",
            }
        if isinstance(feedback, dict):
            feedback.setdefault("think", think)
            return feedback
        return self._structured_feedback(data["tool"], Operator_id, args, feedback, think)

    def _structured_feedback(self, action: str, operator_id: str, args: dict, feedback, think: str = ""):
        post = self._post_from_action(action, operator_id, args)
        ok = not (isinstance(feedback, str) and feedback.startswith("error:"))
        if action in {"comment_post", "like_post", "dislike_post"} and post is None:
            ok = False
        result = {
            "ok": ok,
            "action": action,
            "think": think,
            "feedback": feedback or "",
        }
        if post is not None:
            result["post_id"] = post.id
            result["post"] = post.to_dict()
        elif "post_id" in args:
            result["post_id"] = args.get("post_id")
        return result

    def _post_from_action(self, action: str, operator_id: str, args: dict):
        if action == "send_post":
            for post in reversed(self.posts):
                if post.author_id == operator_id:
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
