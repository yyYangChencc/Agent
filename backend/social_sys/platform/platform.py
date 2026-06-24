import json
import threading
from tools.operator_tools import SocialOperator, register_operator_tools
from social_sys.post.post import Post
from persona.logger import get_logger
from persona.opinion.scale import OPINION_NEUTRAL, clamp_opinion

logger = get_logger(__name__)

class SocialPlatform:
    """简化社交平台。

    平台负责保存用户和帖子，并执行 SocialOperator 工具。帖子可见性由
    give_post 控制：关注对象帖子和系统新闻对智能体可见。
    """

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

    def give_post(self, receiver_id):
        """返回某个智能体本轮可见的帖子。"""

        res = []
        receiver = self.get_agent(receiver_id)
        if not receiver:
            return res
        for post in self.posts:
            if post.author_id in receiver.followers or post.is_news:
                res.append(post)
        return res

    def inject_news(self, tick: int, title: str, content: str, opinion_index: float = OPINION_NEUTRAL):
        """投放真实新闻到社交平台，所有智能体自动可见（通过 is_news 标记）。"""
        post_id = len(self.posts) + 1
        news_post = Post(post_id, "system", f"【{title}】{content}", is_news=True)
        news_post.opinion_index = clamp_opinion(opinion_index)
        news_post.time = tick
        self.posts.append(news_post)
        logger.info("[News] tick=%d 投放新闻: %s", tick, title)
        return news_post

    def execute(self, Operator_id, action_str):
        """解析社交动作 JSON，并调用 SocialOperator 中的具体工具。"""

        if not action_str:
            return ""
        try:
            data = json.loads(action_str)
        except json.JSONDecodeError:
            logger.error("[SocialPlatform] JSON 解析失败，原始内容: %r", action_str)
            return ""
        if not data or "tool" not in data:
            logger.debug("[SocialPlatform] %s 本轮选择不行动", Operator_id)
            return ""
        tool = self.tools.get(data["tool"], None)
        if tool:
            args = data.get("args", {})
            args["operator_ID"] = Operator_id
            try:
                feedback = tool.run(**args)
            except Exception as e:
                logger.error("[SocialPlatform] 工具 %s 执行失败 (operator=%s): %s", data["tool"], Operator_id, e, exc_info=True)
                return "error: tool execution failed"
            return feedback
        logger.warning("[SocialPlatform] 未知工具: %s", data.get("tool"))
        return "error: tool not found"
