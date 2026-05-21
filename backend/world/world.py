import json
import threading
import concurrent.futures
from world.event import Event
from world.map import Map
from tools.operator_tools import register_operator_tools, Operator
from world.observer import observe
from persona.logger import get_logger

logger = get_logger(__name__)

class World:
    def __init__(self, opinion_updater=None, history_recorder=None):
        self.time = 0
        self.map = Map(25, 25)
        self.agents = {}
        self.objects = {}
        self._world_lock = threading.Lock()
        self.conversation_policy = None   # set externally to enable conversation phase
        self.conversation_max_rounds = 3  # max conversation rounds per time step
        self.opinion_updater = opinion_updater
        self.history_recorder = history_recorder
        operator = Operator(self)
        self.tools, self.tools_prompt = register_operator_tools(operator)

    def add_agent(self, agent):
        self.agents[agent.id] = agent
        x, y = agent.position
        self.map.place(x, y, agent.id)

    def add_object(self, obj):
        self.objects[obj.id] = obj
        x, y = obj.position
        self.map.place(x, y, obj.id)

    def step(self):
        self.time += 1
        self.map.print_map()
        agents = list(self.agents.values())

        def _agent_full_step(agent):
            try:
                if agent.sleeping:
                    agent.sleep_ticks_remaining -= 1
                    agent.tick_needs()
                    if agent.sleep_ticks_remaining <= 0:
                        bed = self.objects.get(agent.sleeping_on_bed_id)
                        agent.wakeup(bed)
                    return
                obs    = observe(agent, agent.config.observation_radius)
                action = agent.step(obs)
                reward = self.execute(agent, action)
                agent.append_trajectory(obs, action, reward)
                agent.get_reflect()     #反思总结
                agent.tick_needs()      #需求的自然衰减
            except Exception as e:
                logger.error("[World] agent %s 本轮执行失败: %s", agent.id, e, exc_info=True)

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(agents) or 1) as ex:
            futs = [ex.submit(_agent_full_step, a) for a in agents]
            for f in concurrent.futures.as_completed(futs):
                f.result()

        if self.opinion_updater:
            for agent in self.agents.values():
                self.opinion_updater.online_update(agent)
            if self.time % self.opinion_updater.config.offline_update_interval == 0:
                for agent in self.agents.values():
                    self.opinion_updater.offline_update(agent, self.agents)

        if self.conversation_policy:
            self._conversation_phase(agents)

        if self.history_recorder:
            self.history_recorder.record(self.time, list(self.agents.values()))

    def _conversation_phase(self, agents):
        max_rounds = self.conversation_max_rounds
        for a in agents:
            a.conversation_opted_out = False  # reset each time step

        conv_history = []
        # Seed history with speaks that happened during the main parallel step
        for a in agents:
            for msg in a.inbox:
                conv_history.append({
                    "round": 0,
                    "sender": msg["sender"],
                    "target": a.id,
                    "content": msg["content"],
                    "response_to": msg.get("response_to"),
                })

        # Resolve mutual speaks: if A→B and B→A both happened in the main step,
        # clear the inbox of the lex-smaller agent so only one conversation thread is active
        main_step_senders = {entry["sender"] for entry in conv_history if entry["round"] == 0}
        for a in agents:
            if a.id in main_step_senders and a.inbox:
                if all(msg["sender"] in main_step_senders for msg in a.inbox):
                    if all(a.id < msg["sender"] for msg in a.inbox):
                        logger.info("[World] 检测到互相说话，清空 %s 的收件箱以避免双向对话循环", a.id)
                        a.inbox.clear()

        for round_n in range(1, max_rounds + 1):
            respondents = [a for a in agents if a.inbox and not a.conversation_opted_out]
            if not respondents:
                break
            logger.info("[World] 对话轮次 %d/%d，%d 个智能体待回复",
                        round_n, max_rounds, len(respondents))

            round_history = []

            def _respond(agent, rn=round_n, mr=max_rounds):
                try:
                    action = agent.conversation_step(
                        self.conversation_policy, rn, mr, list(conv_history)
                    )
                    if not action:
                        return
                    try:
                        data = json.loads(action)
                        if data.get("tool") == "speak":
                            self.execute(agent, action)
                            args = data.get("args", {})
                            round_history.append({
                                "round": rn,
                                "sender": agent.id,
                                "target": args.get("ID", ""),
                                "content": args.get("content", ""),
                                "response_to": args.get("response_to"),
                            })
                    except (json.JSONDecodeError, KeyError):
                        pass
                except Exception as e:
                    logger.error("[World] agent %s 对话轮次 %d 失败: %s", agent.id, rn, e, exc_info=True)

            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=len(respondents) or 1) as ex:
                futs = [ex.submit(_respond, a) for a in respondents]
                for f in concurrent.futures.as_completed(futs):
                    f.result()

            conv_history.extend(round_history)

    def execute(self, agent, action_str):
        old_need = agent.need.copy()
        old_demand = agent.demand.copy()
        if not action_str:
            return
        try:
            data = json.loads(action_str)
        except json.JSONDecodeError:
            logger.error("[World] JSON 解析失败，原始内容: %r", action_str)
            return

        if not data or "tool" not in data:
            logger.debug("[%s] 本轮选择不行动", agent.id)
            return

        tool = self.tools.get(data["tool"])
        if not tool:
            logger.warning("[World] 未知工具: %s", data.get("tool"))
            return

        args = data.get("args", {})
        args["operator_ID"] = agent.id
        try:
            feedback = tool.run(**args)
        except Exception as e:
            logger.error("[World] 工具 %s 执行失败 (agent=%s): %s", data["tool"], agent.id, e, exc_info=True)
            return
        agent.add_history("feedback: ", feedback)
        event = Event(
            type=data["tool"],
            actor=agent.id,
            info=feedback,
            time=self.time,
            position=agent.position
        )

        for other in self.agents.values():
            if other.id != agent.id and other.can_perceive(event):
                other.perceive(event)
                if data["tool"] == "speak":
                    target_str = args.get("ID", "")
                    if target_str == "<all>" or other.id in target_str.split():
                        other.receive_message(
                            agent.id,
                            args.get("content", ""),
                            args.get("response_to"),
                        )
        reward = sum(
            (agent.need.get(k, 0.0) - old_need.get(k, 0.0)) * old_demand.get(k, 0.0)
            for k in agent.need
        )
        agent.add_history("reward: ", reward)
        return reward
