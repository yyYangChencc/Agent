from __future__ import annotations

import json
import re
from typing import Any

from persona.llm.interface import LLMClient
from persona.opinion.scale import OPINION_NEUTRAL, clamp_opinion


class MockLLMClient(LLMClient):
    """实验验收用的确定性 LLM 客户端。

    该客户端只在 `run_experiment.py --llm mock` 中显式启用，用于跑无外部
    LLM 的集成测试；默认真实实验仍使用 OpenAI 兼容客户端。
    """

    def __init__(self, config=None):
        self._config = config

    def generate(self, system: str, user: str, *, response_format: dict | None = None) -> str:
        system_text = str(system or "")
        user_text = str(user or "")
        if "记忆查询规划器" in system_text:
            return self._memory_plan(user_text)
        if "任务名称" in system_text and "<UrgencyKey>" in system_text:
            return self._task_decision(user_text)
        if "微反思" in system_text:
            return "<Insight>按当前任务继续执行最近可见的目标。</Insight><Focus>继续完成当前需求任务</Focus>"
        if "行为轨迹" in user_text:
            return "完成了一段任务轨迹，后续优先选择最近且可交互的目标。"
        if "社交平台智能体" in system_text:
            return self._social_action(user_text)
        if "线下对话结构化要求" in system_text:
            return self._conversation_action(user_text)
        if "新闻观念调研" in system_text:
            return self._opinion_assessment(user_text)
        if "社交平台文本立场评分器" in system_text:
            return self._post_opinion_score(user_text)
        if "心理评测器" in system_text:
            return self._psychology_assessment(user_text)
        if "记忆整理模块" in system_text:
            return self._person_profile_summary()
        if "自主智能体" in system_text:
            return self._world_action(user_text)
        return json.dumps({"think": "mock 默认不行动", "action": {}}, ensure_ascii=False)

    def get_embeddings(self, text: str) -> list[float]:
        """返回固定维度向量，保证 Chroma 流程可运行。"""

        seed = sum(ord(ch) for ch in str(text or "")[:200])
        return [((seed + index * 17) % 97) / 97.0 for index in range(64)]

    def _memory_plan(self, user_text: str) -> str:
        context = self._extract_after_label(user_text, "## 当前上下文类型") or "world"
        return json.dumps(
            {
                "think": "mock 查询计划：优先使用结构化记忆，避免外部模型调用。",
                "context": context.strip().splitlines()[0],
                "queries": [],
            },
            ensure_ascii=False,
        )

    def _task_decision(self, user_text: str) -> str:
        needs = self._parse_need_lines(user_text)
        target_key = "satiety"
        lowest_margin = float("inf")
        for key, values in needs.items():
            margin = values["satisfaction"] - values["threshold"]
            if margin < lowest_margin:
                lowest_margin = margin
                target_key = key
        task_names = {
            "satiety": "寻找食物",
            "relax": "寻找休息方式",
            "money": "赚钱打工",
        }
        task_names.update({
            "belonging": "寻找线下交流",
            "esteem": "寻找被认可的行动",
            "self_actualization": "寻找自主探索目标",
        })
        return (
            f"<Think>mock 根据 satisfaction 与阈值选择最缺的需求 {target_key}。</Think>"
            f"<Task>{task_names.get(target_key, '处理需求')}</Task>"
            f"<UrgencyKey>{target_key}</UrgencyKey>"
        )

    def _world_action(self, user_text: str) -> str:
        state = self._parse_world_state(user_text)
        observation = self._extract_json_block_after_label(user_text, "## 观测（半径5格）")
        objects = observation.get("objects") if isinstance(observation.get("objects"), list) else []
        notifications = ((observation.get("social") or {}).get("notifications") or [])
        if notifications and state["tick"] % 5 == 0:
            return self._decision("查看系统新闻通知并浏览社交平台。", "social_step", {})
        if state["inside_building_id"] != "none":
            return self._decision("已在建筑内，离开建筑避免重复停留。", "exit_building", {})

        need_key = self._task_need_from_text(user_text)
        if need_key in {"belonging", "esteem", "self_actualization"}:
            return self._decision("高层需求任务优先通过线下交流或线上表达寻找社会反馈。", "social_step", {})
        if need_key == "money":
            building = self._nearest_object(objects, {"company"}, state["position"])
            if building and self._adjacent(state["position"], building["position"]):
                return self._decision("公司在交互范围内，进入公司获取 money。", "enter_building", {"ID": building["id"]})
            if building:
                return self._decision("前往最近公司获取 money。", "move", self._xy_args(building["position"]))
            return self._decision("未观察到公司，本轮等待。", "", {})

        if need_key == "relax":
            bed_obj = self._nearest_object(objects, {"bed"}, state["position"])
            playground_obj = self._nearest_object(objects, {"playground"}, state["position"])
            target = bed_obj or playground_obj
            if target and self._adjacent(state["position"], target["position"]):
                tool = "sleep" if target.get("kind") == "bed" else "enter_building"
                return self._decision("休息目标在交互范围内，执行休息动作。", tool, {"ID": target["id"]})
            if target:
                return self._decision("前往最近休息目标。", "move", self._xy_args(target["position"]))
            return self._decision("未观察到休息目标，本轮等待。", "", {})

        food_obj = self._nearest_object(objects, {"food"}, state["position"])
        shop_obj = self._nearest_object(objects, {"food_shop"}, state["position"])
        target = food_obj or shop_obj
        if target and self._adjacent(state["position"], target["position"]):
            tool = "eat" if target.get("kind") == "food" else "enter_building"
            return self._decision("食物目标在交互范围内，执行补充饱腹度动作。", tool, {"ID": target["id"]})
        if target:
            return self._decision("前往最近食物目标。", "move", self._xy_args(target["position"]))
        return self._decision("未观察到食物目标，本轮等待。", "", {})

    def _social_action(self, user_text: str) -> str:
        payload = self._extract_json_block_after_label(user_text, "## 当前浏览的帖子")
        posts = payload.get("posts") if isinstance(payload.get("posts"), list) else []
        news = next((post for post in posts if post.get("is_news")), None)
        if news is not None:
            topic = news.get("topic") or "新闻讨论"
            history_text = self._extract_section(user_text, "## 你的发帖历史", "## 当前浏览的帖子")
            # mock 客户端也遵守社交 prompt：同一主题没有新增想法时不重复发帖。
            if topic in history_text:
                return self._decision("已经围绕该主题发过观点，本轮没有新增理由或求助需求，不再重复发帖。", "", {})
            opinion_index = self._float(news.get("opinion_index"), 0.0)
            if opinion_index >= 0.2:
                stance_text = "我谨慎支持这个正面叙事，也会继续关注后续证据。"
            elif opinion_index <= -0.2:
                stance_text = "我开始质疑这个正面叙事，担心其中存在不透明或造神问题。"
            else:
                stance_text = "我暂时保持观望，等待更多可核验证据。"
            content = f"我看到了关于{topic}的新消息，{stance_text}"
            return self._decision("看到系统新闻且尚未表达过该主题观点，发布一条包含自身立场的简短帖子。", "send_post", {
                "topic": topic,
                "content": content,
                "opinion_index": opinion_index,
            })
        if posts:
            post_id = posts[0].get("id")
            return self._decision("浏览到帖子，进行轻度互动。", "like_post", {"post_id": post_id})
        return self._decision("没有可见帖子，本轮不进行社交动作。", "", {})

    def _conversation_action(self, user_text: str) -> str:
        """生成符合新版 speak schema 的确定性线下对话回复。"""

        target_match = re.search(r"agent_[A-Za-z0-9_]+", user_text)
        target_id = target_match.group(0) if target_match else "agent_1"
        return self._decision(
            "mock 按新版线下对话 schema 给出温和回应。",
            "speak",
            {
                "ID": target_id,
                "content": "谢谢你的消息，我会认真考虑，也愿意继续交流。",
                "response_to": "",
                "intent": "social_bonding",
                "social_valence": 0.4,
                "topic": "",
                "topic_stance": None,
            },
        )

    def _opinion_assessment(self, user_text: str) -> str:
        data = self._first_json_object(user_text)
        current = self._float(data.get("current_opinion"), OPINION_NEUTRAL)
        seen_posts = ((data.get("context") or {}).get("seen_posts") or [])
        if seen_posts:
            avg = sum(self._float(post.get("opinion_index"), 0.0) for post in seen_posts) / len(seen_posts)
            score = clamp_opinion(current * 0.75 + avg * 0.25)
            evidence = [f"看到帖子 opinion_index={post.get('opinion_index')}" for post in seen_posts[:3]]
        else:
            score = current
            evidence = ["无新增主题帖子"]
        return json.dumps(
            {
                "score": score,
                "confidence": 0.7,
                "reason": "mock 根据本周期看到的系统主题帖子立场调整 opinion。",
                "evidence": evidence,
            },
            ensure_ascii=False,
        )

    def _post_opinion_score(self, user_text: str) -> str:
        data = self._first_json_object(user_text)
        content = str(data.get("content") or "")
        score = 0.0
        if any(word in content for word in ["支持", "同情", "励志", "相信"]):
            score += 0.35
        if any(word in content for word in ["质疑", "反对", "造神", "违规"]):
            score -= 0.35
        return json.dumps({"score": clamp_opinion(score), "reason": "mock 关键词立场评分"}, ensure_ascii=False)

    def _psychology_assessment(self, user_text: str) -> str:
        data = self._first_json_object(user_text)
        need_key = str(data.get("need_key") or "unknown")
        pressure = self._float(((data.get("assessment_window") or {}).get("effective_pressure") or {}).get("max"), 0.0)
        # 心理评测器只接受理论卡中声明过的 mediator key，因此 mock 也必须从理论卡读取精确键名。
        theory_card = data.get("theory_card") if isinstance(data.get("theory_card"), dict) else {}
        mediator_keys = [
            str(item.get("key"))
            for item in theory_card.get("mediators", [])
            if isinstance(item, dict) and item.get("key")
        ]
        if not mediator_keys:
            mediator_keys = ["stress"]
        return json.dumps(
            {
                "mediators": {
                    key: max(0.0, min(1.0, pressure))
                    for key in mediator_keys[:2]
                },
                "role_card_delta": {
                    "summary": f"{need_key} 压力触发的轻量 mock 心理角色卡",
                    "emotion_tone": "谨慎、略紧张",
                    "cognition": ["更关注与当前需求有关的信息。"],
                    "behavior": ["优先处理当前缺口对应的行动。"],
                    "social_expression": ["表达更简短直接。"],
                    "online_behavior": ["倾向于围绕当前关注主题表达低强度看法。"],
                    "decision_bias": ["优先选择能缓解当前压力的动作。"],
                    "constraints": ["不得编造未发生的经历。"],
                },
                "reason": "mock 根据有效压力生成心理中介。",
                "evidence": [f"{need_key} pressure={pressure:.3f}"],
                "confidence": 0.65,
            },
            ensure_ascii=False,
        )

    def _person_profile_summary(self) -> str:
        return json.dumps(
            {
                "actions_impression": "mock 规则摘要：近期行为正常。",
                "opinion_impression": "mock 规则摘要：观念印象暂不明确。",
                "relationship_impression": "mock 规则摘要：关系印象稳定。",
                "confidence": 0.5,
            },
            ensure_ascii=False,
        )

    def _decision(self, think: str, tool: str, args: dict[str, Any]) -> str:
        action = {"tool": tool, "args": args} if tool else {}
        return json.dumps({"think": think, "action": action}, ensure_ascii=False)

    def _parse_world_state(self, user_text: str) -> dict[str, Any]:
        position_match = re.search(r"位置：\((\-?\d+),\s*(\-?\d+)\).*?时间步：t=(\d+)", user_text)
        position = [0, 0]
        tick = 0
        if position_match:
            position = [int(position_match.group(1)), int(position_match.group(2))]
            tick = int(position_match.group(3))
        inside_match = re.search(r"inside_building_id:\s*([^\n]+)", user_text)
        return {
            "position": position,
            "tick": tick,
            "inside_building_id": (inside_match.group(1).strip() if inside_match else "none"),
        }

    def _parse_need_lines(self, text: str) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        pattern = re.compile(r"([A-Za-z_]+): satisfaction=([\d.\-]+).*?threshold=([\d.\-]+)", re.DOTALL)
        for key, satisfaction, threshold in pattern.findall(text):
            out[key] = {
                "satisfaction": self._float(satisfaction, 0.0),
                "threshold": self._float(threshold, 0.0),
            }
        return out

    def _task_need_from_text(self, user_text: str) -> str:
        task_match = re.search(r"- 任务：([^\n]+)", user_text)
        task_text = task_match.group(1) if task_match else ""
        if any(word in task_text for word in ["belonging", "交流", "归属"]):
            return "belonging"
        if any(word in task_text for word in ["esteem", "认可", "尊重"]):
            return "esteem"
        if any(word in task_text for word in ["self_actualization", "探索", "实现"]):
            return "self_actualization"
        if any(word in task_text for word in ["money", "赚钱", "打工"]):
            return "money"
        if any(word in task_text for word in ["relax", "休息", "放松"]):
            return "relax"
        if any(word in task_text for word in ["satiety", "食物", "饱腹", "吃"]):
            return "satiety"
        needs = self._parse_need_lines(user_text)
        if not needs:
            return "satiety"
        return min(needs, key=lambda key: needs[key]["satisfaction"] - needs[key]["threshold"])

    def _nearest_object(self, objects: list[dict], kinds: set[str], position: list[int]) -> dict | None:
        filtered = [
            obj for obj in objects
            if isinstance(obj, dict) and obj.get("kind") in kinds and isinstance(obj.get("position"), list)
        ]
        if not filtered:
            return None
        return min(filtered, key=lambda obj: abs(obj["position"][0] - position[0]) + abs(obj["position"][1] - position[1]))

    def _adjacent(self, a: list[int], b: list[int]) -> bool:
        return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 <= 2

    def _xy_args(self, position: list[int]) -> dict[str, int]:
        return {"x": int(position[0]), "y": int(position[1])}

    def _extract_after_label(self, text: str, label: str) -> str:
        index = text.find(label)
        if index < 0:
            return ""
        return text[index + len(label):].strip()

    def _extract_json_block_after_label(self, text: str, label: str) -> dict:
        section = self._extract_after_label(text, label)
        return self._first_json_object(section)

    def _extract_section(self, text: str, start_label: str, end_label: str) -> str:
        """提取两个标题之间的 prompt 片段，供 mock 决策读取上下文。"""

        start = text.find(start_label)
        if start < 0:
            return ""
        start += len(start_label)
        end = text.find(end_label, start)
        if end < 0:
            end = len(text)
        return text[start:end].strip()

    def _first_json_object(self, text: str) -> dict:
        start = text.find("{")
        if start < 0:
            return {}
        depth = 0
        in_string = False
        escape = False
        for offset, char in enumerate(text[start:], start=start):
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start: offset + 1])
                    except json.JSONDecodeError:
                        return {}
                    return parsed if isinstance(parsed, dict) else {}
        return {}

    def _float(self, value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
