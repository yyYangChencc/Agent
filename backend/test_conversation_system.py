from __future__ import annotations

import json
import os
import sys
import unittest


BACKEND_DIR = os.path.dirname(__file__)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


from persona.conversation import ConversationIntent, ConversationManager
from persona.conversation.session import infer_conversation_intent


class _World:
    def __init__(self):
        self.time = 7
        self.agents = {}
        self.executed_actions: list[tuple[str, dict]] = []

    def execute(self, agent, action_str):
        self.executed_actions.append((agent.id, json.loads(action_str)))


class _Agent:
    def __init__(self, agent_id: str, actions: list[str] | None = None):
        self.id = agent_id
        self.inbox: list[dict] = []
        self.conversation_opted_out = False
        self.actions = list(actions or [])
        self.seen_histories: list[list[dict]] = []

    def conversation_step(self, policy, round_n: int, max_rounds: int, conv_history: list[dict]):
        self.seen_histories.append(conv_history)
        if conv_history:
            session_ids = {entry["session_id"] for entry in conv_history}
            self.inbox = [msg for msg in self.inbox if msg.get("session_id") not in session_ids]
        else:
            self.inbox.clear()
        if not self.actions:
            self.conversation_opted_out = True
            return ""
        return self.actions.pop(0)


def _speak(content: str, target: str, response_to: str | None = None) -> str:
    args = {"content": content, "ID": target}
    if response_to is not None:
        args["response_to"] = response_to
    return json.dumps({"tool": "speak", "args": args}, ensure_ascii=False)


class ConversationSystemTest(unittest.TestCase):
    def test_intent_inference(self):
        self.assertEqual(
            infer_conversation_intent("你知道食物在哪里吗？"),
            ConversationIntent.ASK_INFO,
        )
        self.assertEqual(
            infer_conversation_intent("食物在东边", response_to="你知道食物在哪里吗？"),
            ConversationIntent.ANSWER_INFO,
        )
        self.assertEqual(
            infer_conversation_intent("谢谢，知道了"),
            ConversationIntent.END,
        )
        self.assertEqual(
            infer_conversation_intent("我们一起去商店吧"),
            ConversationIntent.COORDINATION,
        )

    def test_manager_creates_session_from_inbox_and_records_reply(self):
        world = _World()
        agent_1 = _Agent("agent_1")
        agent_2 = _Agent(
            "agent_2",
            actions=[_speak("食物在东边三格", "agent_1", response_to="你知道食物在哪里吗？")],
        )
        world.agents = {agent.id: agent for agent in [agent_1, agent_2]}
        agent_2.inbox.append({
            "sender": "agent_1",
            "content": "你知道食物在哪里吗？",
            "response_to": None,
            "time": world.time,
        })

        manager = ConversationManager(world)
        manager.run_phase([agent_1, agent_2], policy=None, max_rounds=2)

        self.assertEqual(len(manager.sessions), 1)
        session = next(iter(manager.sessions.values()))
        self.assertEqual(session.session_id, "conv_7_1")
        self.assertEqual(session.participants, ["agent_1", "agent_2"])
        self.assertEqual(session.status, "resolved")
        self.assertEqual(session.termination_reason, "resolved_intent")
        self.assertEqual([message.intent for message in session.messages], [
            ConversationIntent.ASK_INFO,
            ConversationIntent.ANSWER_INFO,
        ])
        self.assertEqual(len(world.executed_actions), 1)
        self.assertEqual(world.executed_actions[0][0], "agent_2")
        self.assertEqual(world.executed_actions[0][1]["args"]["content"], "食物在东边三格")

        first_history = agent_2.seen_histories[0]
        self.assertEqual(first_history[0]["session_id"], "conv_7_1")
        self.assertEqual(first_history[0]["intent"], "ask_info")

    def test_manager_resolves_session_when_agent_opts_out(self):
        world = _World()
        agent_1 = _Agent("agent_1")
        agent_2 = _Agent("agent_2")
        world.agents = {agent.id: agent for agent in [agent_1, agent_2]}
        agent_2.inbox.append({
            "sender": "agent_1",
            "content": "今天过得怎么样",
            "response_to": None,
            "time": world.time,
        })

        manager = ConversationManager(world)
        manager.run_phase([agent_1, agent_2], policy=None, max_rounds=2)

        session = next(iter(manager.sessions.values()))
        self.assertEqual(session.status, "resolved")
        self.assertEqual(session.termination_reason, "speaker_opted_out")
        self.assertEqual(len(session.messages), 1)
        self.assertEqual(session.messages[0].intent, ConversationIntent.SOCIAL_BONDING)
        self.assertEqual(world.executed_actions, [])

    def test_manager_selects_only_targeted_speaker_for_session(self):
        world = _World()
        agent_1 = _Agent("agent_1")
        agent_2 = _Agent("agent_2", actions=[_speak("我知道，食物在东边", "agent_1", response_to="食物在哪？")])
        agent_3 = _Agent("agent_3", actions=[_speak("我不应该被选中", "agent_1")])
        world.agents = {agent.id: agent for agent in [agent_1, agent_2, agent_3]}
        agent_2.inbox.append({
            "sender": "agent_1",
            "content": "食物在哪？",
            "response_to": None,
            "time": world.time,
        })
        agent_3.inbox.append({
            "sender": "agent_9",
            "content": "今天过得怎么样",
            "response_to": None,
            "time": world.time,
        })

        manager = ConversationManager(world)
        manager.run_phase([agent_1, agent_2, agent_3], policy=None, max_rounds=1)

        self.assertEqual(len(manager.sessions), 2)
        self.assertEqual(len(world.executed_actions), 2)
        self.assertEqual([actor for actor, _ in world.executed_actions], ["agent_2", "agent_3"])

    def test_broadcast_message_creates_one_session_with_multiple_possible_speakers(self):
        world = _World()
        agent_1 = _Agent("agent_1")
        agent_2 = _Agent("agent_2", actions=[_speak("我知道一些线索", "agent_1", response_to="大家知道食物在哪吗？")])
        agent_3 = _Agent("agent_3", actions=[_speak("我也有线索", "agent_1", response_to="大家知道食物在哪吗？")])
        world.agents = {agent.id: agent for agent in [agent_1, agent_2, agent_3]}
        for receiver in [agent_2, agent_3]:
            receiver.inbox.append({
                "sender": "agent_1",
                "target": "<all>",
                "content": "大家知道食物在哪吗？",
                "response_to": None,
                "time": world.time,
            })

        manager = ConversationManager(world)
        manager.run_phase([agent_1, agent_2, agent_3], policy=None, max_rounds=1)

        self.assertEqual(len(manager.sessions), 1)
        session = next(iter(manager.sessions.values()))
        self.assertEqual(session.participants, ["agent_1", "agent_2", "agent_3"])
        self.assertEqual(len(session.messages), 2)
        self.assertEqual(len(world.executed_actions), 1)
        self.assertEqual(world.executed_actions[0][0], "agent_2")


if __name__ == "__main__":
    unittest.main()
