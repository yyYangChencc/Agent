import json
import threading
from tools.operator_tools import SocialOperator, register_operator_tools
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
            if post.author_id in receiver.followers:
                res.append(post)
        return res

    def execute(self, Operator_id, action_str):
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
