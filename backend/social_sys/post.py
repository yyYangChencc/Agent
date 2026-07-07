import random


class Comment:
    """帖子评论，包含评论者对原帖的认同值。"""

    def __init__(self, id, author_id, content, time=None, agreement_to_post=0.0):
        self.id = id
        self.author_id = author_id
        self.content = content
        self.time = time
        self.agreement_to_post = agreement_to_post

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
        }


class Post:
    def __init__(self, id, author_id, content, is_rumor=False, is_news=False, topic="", source_type="agent"):
        self.id = id
        self.author_id = author_id
        self.content = content     # 帖子内容
        self.is_rumor = is_rumor   # 是否谣言
        self.is_news = is_news     # 是否为系统投放的真实新闻
        self.topic = topic         # 帖子主题；新闻帖使用系统主题，普通帖由发帖者自定义
        self.source_type = "official_news" if is_news else source_type  # 区分官方新闻、投放者和普通智能体发帖
        self.opinion_index: float = 0.0  # 作者发布时的观念快照，新闻帖表示该新闻对主题正面叙事的立场

        self.likes = 0
        self.reposts = 0
        self.comments = 0
        self.dislikes = 0
        self.time = 0 #发表时间
  
        self.comments_list = []

    def popularity(self):
        return self.likes + 2*self.reposts + self.comments
    
    def add_comment(self,comment):
        self.comments_list.append(comment)
        self.comments += 1
     
    def show(self):
        tag = "[新闻] " if self.is_news else ""
        comments = "\n".join([comment.show() for comment in self.comments_list])
        return f"""
        帖子ID：{self.id}
        发布时间：{self.time}  发布作者：{self.author_id}
        主题：{self.topic}
        {tag}帖子内容：{self.content}
        评论：{comments}
        """
    
    def add_like(self, agent_id):
        self.likes += 1 

    def add_dislike(self, agent_id):
        self.dislikes += 1

    def to_dict(self):
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
            "comments": [
                comment.to_dict() if hasattr(comment, "to_dict") else {
                    "id": getattr(comment, "id", ""),
                    "author_id": getattr(comment, "author_id", ""),
                    "content": getattr(comment, "content", ""),
                    "time": getattr(comment, "time", None),
                    "agreement_to_post": getattr(comment, "agreement_to_post", 0.0),
                }
                for comment in self.comments_list
            ],
        }
