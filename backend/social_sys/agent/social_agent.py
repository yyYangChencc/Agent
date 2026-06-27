import json

from social_sys.post.post import Post
from social_sys.platform.platform import SocialPlatform
from persona.logger import get_logger

logger = get_logger(__name__)

class SocialAgent:
    def __init__(self, id, mem, platform, social_policy, world):
        self.id = id
        self.post_history = []
        self.followers = []
        self.file = None
        self.mem = mem
        self.platform = platform
        self.social_policy = social_policy
        self.world = world
        self._last_seen_posts = []
        self.platform.add_agent(self)

    def add_follower(self, follower_id):
        if follower_id not in self.followers:
            self.followers.append(follower_id)
            logger.info("%s 关注了 %s", follower_id, self.id)
        else:
            logger.info("%s 已经关注了 %s", follower_id, self.id)

    def add_post_history(self, post):
        self.post_history.append(post)

    def get_post_history(self):
        return "\n".join([post.show() for post in self.post_history])

    def receive_post(self):
        posts = self.platform.get_visible_posts(self.id)
        posts_info = self.platform.give_post(self.id)
        return posts, json.dumps(posts_info, ensure_ascii=False, indent=2)

    def recall(self, obs):
        return self.mem.retrieve_agent_memories(self.id, obs, n_results=3)

    def step(self):
        logger.info("[%s] 正在查看帖子...", self.id)
        posts, posts_info = self.receive_post()
        self._last_seen_posts = posts
        logger.debug("[%s] 帖子内容: %s", self.id, posts_info)
        mem_info = self.recall(posts_info)
        logger.debug("[%s] 相关记忆: %s", self.id, mem_info)
        raw = self.social_policy.decide(self, posts_info, mem_info)
        feedback = self.platform.execute(self.id, raw)
        logger.debug("[%s] 平台反馈: %s", self.id, feedback)
