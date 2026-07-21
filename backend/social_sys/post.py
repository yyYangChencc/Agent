import random
import threading


class Comment:
    """帖子评论，包含评论者对原帖的认同值。"""

    def __init__(
        self,
        id,
        author_id,
        content,
        time=None,
        agreement_to_post=0.0,
        parent_comment_id=None,
        root_comment_id=None,
    ):
        self.id = id
        self.author_id = author_id
        self.content = content
        self.time = time
        self.agreement_to_post = agreement_to_post
        self.parent_comment_id = parent_comment_id
        self.root_comment_id = root_comment_id

    def show(self):
        return (
            f"评论作者:{self.author_id},创建时间：{self.time},"
            f"认同值:{self.agreement_to_post},评论内容:{self.content}"
        )

    def get_id(self):
        return self.id

    def to_dict(self):
        return {
            "id": self.id,
            "author_id": self.author_id,
            "content": self.content,
            "time": self.time,
            "agreement_to_post": self.agreement_to_post,
            "parent_comment_id": self.parent_comment_id,
            "root_comment_id": self.root_comment_id,
        }


class Post:
    def __init__(
        self,
        id,
        author_id,
        content,
        is_rumor=False,
        is_news=False,
        topic="",
        source_type="agent",
        repost_of_post_id=None,
        root_post_id=None,
        source_author_id=None,
    ):
        self.id = id
        self.author_id = author_id
        self.content = content     # 帖子内容
        self.is_rumor = is_rumor   # 是否谣言
        self.is_news = is_news     # 是否为系统投放的真实新闻
        self.topic = topic         # 帖子主题；新闻帖使用系统主题，普通帖由发帖者自定义
        self.source_type = "official_news" if is_news else source_type  # 区分官方新闻、投放者和普通智能体发帖
        self.opinion_index: float = 0.0  # 作者发布时的观念快照，新闻帖表示该新闻对主题正面叙事的立场

        self.repost_of_post_id = repost_of_post_id
        self.root_post_id = root_post_id
        self.source_author_id = source_author_id

        # 公开计数通过只读属性获取，写入统一在锁内完成。
        self._lock = threading.RLock()
        self._likes = 0
        self._reposts = 0
        self.comments = 0
        self._dislikes = 0
        self.time = 0 #发表时间
  
        self.comments_list = []
        self._reaction_by_agent = {}
        self._repost_agent_ids = set()

    @property
    def likes(self):
        with self._lock:
            return self._likes

    @property
    def dislikes(self):
        with self._lock:
            return self._dislikes

    @property
    def reposts(self):
        with self._lock:
            return self._reposts

    def popularity(self):
        with self._lock:
            return self.likes + 2 * self.reposts + self.comments
    
    def add_comment(self, comment):
        with self._lock:
            existing_ids = {item.id for item in self.comments_list}
            if not getattr(comment, "id", None) or comment.id in existing_ids:
                next_index = self.comments + 1
                comment_id = f"{self.id}_c{next_index}"
                while comment_id in existing_ids:
                    next_index += 1
                    comment_id = f"{self.id}_c{next_index}"
                comment.id = comment_id
            self.comments_list.append(comment)
            self.comments += 1
            return comment

    def get_comment(self, comment_id):
        """按精确评论 ID 返回评论对象。"""

        with self._lock:
            return next((item for item in self.comments_list if item.id == comment_id), None)
     
    def show(self):
        tag = "[新闻] " if self.is_news else ""
        with self._lock:
            comments_snapshot = list(self.comments_list)
        comments = "\n".join([comment.show() for comment in comments_snapshot])
        return f"""
        帖子ID：{self.id}
        发布时间：{self.time}  发布作者：{self.author_id}
        主题：{self.topic}
        {tag}帖子内容：{self.content}
        评论：{comments}
        """

    def set_reaction(self, agent_id, reaction):
        """设置单个智能体的反应，并返回变更前后的精确状态。"""
        if reaction not in {"like", "dislike", None}:
            raise ValueError("reaction 必须是 'like'、'dislike' 或 None")

        with self._lock:
            previous = self._reaction_by_agent.get(agent_id)
            if previous == reaction:
                return {"changed": False, "previous": previous, "current": reaction}

            if previous == "like":
                self._likes -= 1
            elif previous == "dislike":
                self._dislikes -= 1

            if reaction is None:
                self._reaction_by_agent.pop(agent_id, None)
            else:
                self._reaction_by_agent[agent_id] = reaction

            if reaction == "like":
                self._likes += 1
            elif reaction == "dislike":
                self._dislikes += 1

            return {"changed": True, "previous": previous, "current": reaction}

    def add_like(self, agent_id):
        return self.set_reaction(agent_id, "like")

    def add_dislike(self, agent_id):
        return self.set_reaction(agent_id, "dislike")

    def clear_reaction(self, agent_id):
        return self.set_reaction(agent_id, None)

    def add_repost(self, agent_id):
        """按智能体标识登记一次转发，重复登记不增加计数。"""
        with self._lock:
            previous = agent_id in self._repost_agent_ids
            if previous:
                return {"changed": False, "previous": True, "current": True}

            self._repost_agent_ids.add(agent_id)
            self._reposts += 1
            return {"changed": True, "previous": False, "current": True}

    def to_dict(self):
        with self._lock:
            comments_snapshot = list(self.comments_list)
            return {
                "id": self.id,
                "author_id": self.author_id,
                "topic": self.topic,
                "content": self.content,
                "time": self.time,
                "likes": self.likes,
                "dislikes": self.dislikes,
                "reposts": self.reposts,
                "comments_count": self.comments,
                "opinion_index": self.opinion_index,
                "is_news": self.is_news,
                "is_rumor": self.is_rumor,
                "source_type": self.source_type,
                "repost_of_post_id": self.repost_of_post_id,
                "root_post_id": self.root_post_id,
                "source_author_id": self.source_author_id,
                "comments": [
                    comment.to_dict() if hasattr(comment, "to_dict") else {
                        "id": getattr(comment, "id", ""),
                        "author_id": getattr(comment, "author_id", ""),
                        "content": getattr(comment, "content", ""),
                        "time": getattr(comment, "time", None),
                        "agreement_to_post": getattr(comment, "agreement_to_post", 0.0),
                        "parent_comment_id": getattr(comment, "parent_comment_id", None),
                        "root_comment_id": getattr(comment, "root_comment_id", None),
                    }
                    for comment in comments_snapshot
                ],
            }
