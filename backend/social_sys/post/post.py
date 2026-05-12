import random
import numpy as np

class Post:
    def __init__(self,id, author_id, content, is_rumor=False):
        self.id = id
        self.author_id = author_id
        self.content = content     # 帖子内容
        self.is_rumor = is_rumor   # 是否谣言
        self.opinion_index: float = 0.0  # author's opinion snapshot at post creation time

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
        comments = "\n".join([comment.show() for comment in self.comments_list])
        return f"""
        发布时间：{self.time}  发布作者：{self.author_id} 
        帖子内容：{self.content}
        评论：{comments}
        """
    
    def add_like(self, agent_id):
        self.likes += 1 

    def add_dislike(self, agent_id):
        self.dislikes += 1