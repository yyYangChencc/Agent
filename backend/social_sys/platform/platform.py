import json
import threading
from tools.operator_tools import SocialOperator, register_operator_tools
from social_sys.post.post import Post
from persona.logger import get_logger

logger = get_logger(__name__)

class SocialPlatform:
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
        res = []
        receiver = self.get_agent(receiver_id)
        if not receiver:
            return res
        for post in self.posts:
            if post.author_id in receiver.followers or post.is_news:
                res.append(post)
        return res

    def inject_news(self, tick: int, title: str, content: str, opinion_index: float = 0.5):
        """投放真实新闻到社交平台，所有智能体自动可见（通过 is_news 标记）。"""
        post_id = len(self.posts) + 1
        news_post = Post(post_id, "system", f"【{title}】{content}", is_news=True)
        news_post.opinion_index = opinion_index
        news_post.time = tick
        self.posts.append(news_post)
        logger.info("[News] tick=%d 投放新闻: %s", tick, title)
        return news_post

    def execute(self, Operator_id, action_str):
        #执行智能体调用的函数，具体逻辑在 tools/operator_tools.py 中定义，action_str 是一个 JSON 字符串，包含工具名称和参数
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
