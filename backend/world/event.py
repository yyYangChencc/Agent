class Event:
    def __init__(self, type, actor, info, time, position=None, acted=None):
        self.type = type          # "move", "speak", ...
        self.actor = actor        # agent ID
        self.info = info          # 事件内容
        self.position = position  # 发生位置
        self.time = time          # 发生时间
        self.acted = acted        # 行动对象

class SpeakingEvent(Event):
    def __init__(
        self,
        type,
        actor,
        info,
        time,
        response_to,
        position=None,
        acted=None,
        intent=None,
        social_valence=None,
        topic="",
        topic_stance=None,
    ):
        super().__init__(type, actor, info, time, position, acted)
        self.response_to = response_to
        self.intent = intent
        self.social_valence = social_valence
        self.topic = topic
        self.topic_stance = topic_stance
