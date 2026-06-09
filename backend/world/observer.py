from world.event import SpeakingEvent,Event
from persona.logger import get_logger

logger = get_logger(__name__)

def observe(agent,r:int) -> list:
    res = []
    res.append("周边环境物品及人物如下：")
    x,y = agent.position
    for i in range(max(0,x-r),min(agent.world.map.height,x+r)):
        for j in range(max(0,y-r),min(agent.world.map.width,y+r)):
            if not agent.world.map.is_empty(i,j):
                id = agent.world.map.get_e(i,j)
                if id == agent.id:
                    continue  
                if id in agent.world.agents:
                    other = agent.world.agents[id]
                    res.append(f"ID:{id} 位置:({i},{j}) 类别:agent")
                else:
                    obj = agent.world.objects[id]
                    desc = obj.get_desc() if hasattr(obj, "get_desc") else ""
                    line = f"位置:({i},{j})"
                    if desc:
                        line += f" 描述:{desc}"
                    res.append(line)
    if len(res) == 1:
        res.append("无可见物体或人物")

    res.append("周边可观测动作如下：")
    for event in agent.observed_events:
        if isinstance(event,SpeakingEvent):
            res.append(f"行动人:{event.actor} 行动对象:{event.acted}  动作类型:{event.type}  动作内容:{event.info} 发生时间:{event.time} 发生位置:{event.position}  回复内容：{event.response_to}")
        else:
            res.append(f"行动人:{event.actor} 行动对象:{event.acted}  动作类型:{event.type}  动作内容:{event.info} 发生时间:{event.time} 发生位置:{event.position}")
    if len(res) == 3:
        res.append("无可观测动作")
    agent.observed_events.clear()

    # 社交通知（评论/点赞/点踩 + 新闻推送）
    if agent._pending_social_notifications:
        res.append("社交通知如下：")
        for note in agent._pending_social_notifications:
            res.append(note)
        agent._pending_social_notifications.clear()

    agent.observation = "\n".join(res)
    res = "\n".join(res)
    logger.debug("[%s] 观测结果:\n%s", agent.id, res)
    return res
