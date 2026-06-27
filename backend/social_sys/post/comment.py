class Comment:
    def __init__(self,id,author_id,content,time=None):
        self.id = id
        self.author_id = author_id
        self.content  = content
        self.time = time         #创建时间
    def show(self):  
        return f"评论作者:{self.author_id},创建时间：{self.time},评论内容:{self.content}"
    def get_id(self):
        return self.id

    def to_dict(self):
        return {
            "id": self.id,
            "author_id": self.author_id,
            "content": self.content,
            "time": self.time,
        }
