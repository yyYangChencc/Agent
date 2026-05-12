from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from persona.agents.agent import Agent

# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class BasePromptBuilder:
    """All concrete builders return a (system_prompt, user_prompt) tuple."""

    # -- shared block helpers ------------------------------------------------

    def _state_block(self, agent: "Agent") -> str:
        return (
            f"- 位置：({agent.position[0]}, {agent.position[1]}) "
            f"| 时间步：t={agent.world.time}"
        )

    def _demand_block(self, agent: "Agent") -> str:
        d = agent.demand
        n = agent.need
        t = agent.demand_threshold
        lines = ["- 需求状态（当前属性=客观拥有量 0→1，急迫度=主观急迫度 1→0，当前属性超过阈值表示已满足）："]
        for k, urgency in d.items():
            need_val = n.get(k, 0.0)
            th = t.get(k, "?")
            status = "已满足" if isinstance(th, float) and need_val > th else "未满足"
            lines.append(
                f"  {k}：当前属性 {need_val:.2f} | 急迫度 {urgency:.2f}"
                f"（阈值 {th:.2f}，{status}）"
            )
        return "\n".join(lines)

    def _history_block(self, agent: "Agent", max_n: int = 10) -> str:
        recent = agent.history[-max_n:]
        if not recent:
            return "（无历史记录）"
        return "\n".join(f"  {i + 1}. {h}" for i, h in enumerate(recent))

    def _memory_block(self, mem_info: list | str | None) -> str:
        if not mem_info:
            return "（无相关记忆）"
        if isinstance(mem_info, list):
            return "\n".join(f"  - {m}" for m in mem_info) if mem_info else "（无相关记忆）"
        return mem_info

    def _opinion_block(self, agent: "Agent") -> str:
        return f"- 当前观念倾向：{agent.opinion:.3f}（0=保守，1=激进）"

    def _focus_block(self, agent: "Agent") -> str:
        if not agent.current_focus:
            return ""
        return f"- 当前焦点：{agent.current_focus}\n"

    def build(self, agent: "Agent", observation: str, mem_info) -> tuple[str, str]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# World interaction (replaces PromptBuilder + MemPromptBuilder)
# ---------------------------------------------------------------------------

class WorldPromptBuilder(BasePromptBuilder):
    """Standard world interaction prompt for move / eat / speak / social_step."""

    def build(self, agent: "Agent", observation: str, mem_info) -> tuple[str, str]:
        persona_lines = []
        if agent.role:
            persona_lines.append(f"- 角色：{agent.role}")
        if agent.speaking_style:
            persona_lines.append(f"- 说话风格：{agent.speaking_style}")
        persona_lines.append(f"- 当前情绪：{agent.emotion}")
        persona_block = "\n".join(persona_lines)

        system = (
            f"你是自主智能体 {agent.id}，运行在一个 2D 网格世界中。\n"
            f"{persona_block}\n"
            "你的目标是根据当前状态、观测和记忆，选择最合理的单一动作。\n\n"
            "行为原则：\n"
            "- 需求值越高表示越急迫，低于阈值表示该需求已暂时满足\n"
            "- 优先执行当前任务，通过行动降低对应需求值，若行动可顺便解决其他需求，可在不耽误当前主任务的前提下进行\n"
            "- 若任务为 none，根据需求自由决策\n"
            "- 观测外的物体无法直接交互，但可通过记忆、向附近智能体 speak 或使用 social_step 发帖来获取其位置信息\n"
            "- 若任务长时间无进展，优先依据当前焦点行动\n"
            "- 每轮只能执行一个工具调用\n\n"
            "输出格式（必须严格遵守，必须输出完整成对的<Think></Think>与<Action></Action>标签）：\n"
            "<Think>\n"
            "[分析：当前任务状态 → 最紧迫需求 → 可行动作列表 → 前提检查 → 选择理由]\n"
            "</Think>\n"
            "<Action>\n"
            '{"tool": "<tool_name>", "args": {"<key>": <value>}}\n'
            "</Action>\n"
            "若本轮无可执行动作：\n"
            "<Action>\n"
            "{}\n"
            "</Action>\n\n"
            "示例：\n"
            "<Think>\n"
            "任务 eat something，satiety=0.85 未满足。观测到 food_1 在 (7,5)，当前位置 (3,5)，需移动靠近。move 工具满足前提。\n"
            "</Think>\n"
            "<Action>\n"
            '{"tool": "move", "args": {"x": 7, "y": 5}}\n'
            "</Action>"
        )

        user = (
            "## 当前状态\n"
            f"{self._state_block(agent)}\n"
            f"- 任务：{agent.task}\n"
            f"{self._demand_block(agent)}\n"
            f"{self._focus_block(agent)}"
            f"{self._opinion_block(agent)}\n\n"
            "## 观测（半径5格）\n"
            f"{observation}\n\n"
            "## 近期历史\n"
            f"{self._history_block(agent)}\n\n"
            "## 相关记忆\n"
            f"{self._memory_block(mem_info)}\n\n"
            "## 可调用工具\n"
            f"{agent.world.tools_prompt}"
        )
        return system, user


# ---------------------------------------------------------------------------
# Social platform
# ---------------------------------------------------------------------------

class SocialPromptBuilder(BasePromptBuilder):
    """Social platform interaction prompt."""

    def build(self, agent: "Agent", posts_info: str, mem_info) -> tuple[str, str]:
        persona_lines = []
        if agent.role:
            persona_lines.append(f"- 角色：{agent.role}")
        if agent.speaking_style:
            persona_lines.append(f"- 说话风格：{agent.speaking_style}")
        persona_lines.append(f"- 当前情绪：{agent.emotion}")
        persona_block = "\n".join(persona_lines)

        system = (
            f"你是社交平台智能体 {agent.id}。\n"
            f"{persona_block}\n"
            "你的目标是通过社交平台获取有用信息、表达观点、建立社交联系；\n"
            "当物理世界中遇到困难（如找不到食物），也可发帖向他人寻求帮助。\n"
            "每轮只能执行一个社交动作。\n\n"
            "行为原则：\n"
            "- 发帖内容应与你的记忆和当前话题相关\n"
            "- 不允许重复发布完全相同的内容\n"
            "- 仅对真实存在的帖子执行 comment / like / dislike\n\n"
            "输出格式（必须严格遵守，必须输出完整成对的<Think></Think>与<Action></Action>标签）：\n"
            "<Think>\n"
            "[分析：当前帖子对我意味着什么 → 我想传达什么 → 最合适的社交动作]\n"
            "</Think>\n"
            "<Action>\n"
            '{"tool": "<tool_name>", "args": {"<key>": <value>}}\n'
            "</Action>\n"
            "若本轮无可执行动作：\n"
            "<Action>\n"
            "{}\n"
            "</Action>\n\n"
            "示例：\n"
            "<Think>\n"
            "帖子讨论的是食物话题，与我的记忆相关。我想分享自己的见解，适合发帖。\n"
            "</Think>\n"
            "<Action>\n"
            '{"tool": "send_post", "args": {"content": "今天发现了一种新食材，味道很不错！"}}\n'
            "</Action>"
        )

        user = (
            "## 当前状态\n"
            f"- 时间步：t={agent.world.time}\n"
            f"- 任务：{agent.task}\n"
            f"{self._demand_block(agent)}\n"
            f"{self._focus_block(agent)}"
            f"{self._opinion_block(agent)}\n\n"
            "## 你的发帖历史\n"
            f"{agent.get_post_history() or '（暂无发帖记录）'}\n\n"
            "## 当前浏览的帖子\n"
            f"{posts_info}\n\n"
            "## 相关记忆\n"
            f"{self._memory_block(mem_info)}\n\n"
            "## 可调用工具\n"
            f"{agent.platform.tools_prompt}"
        )
        return system, user


# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------

class ConversationPromptBuilder(BasePromptBuilder):
    """Multi-round conversation prompt."""

    def build(self, agent: "Agent", observation: str, mem_info) -> tuple[str, str]:
        persona_lines = []
        if agent.role:
            persona_lines.append(f"- 角色：{agent.role}")
        if agent.speaking_style:
            persona_lines.append(f"- 说话风格：{agent.speaking_style}")
        persona_lines.append(f"- 当前情绪：{agent.emotion}")
        persona_block = "\n".join(persona_lines)

        system = (
            f"你是智能体 {agent.id}，当前正在进行对话。\n"
            f"{persona_block}\n"
            "你需要根据收到的消息和上下文，决定是回复还是保持沉默（沉默将终止对话）。\n\n"
            "行为原则：\n"
            "- 若剩余轮数为 0，必须主动收尾或保持沉默\n"
            "- 对话目的达成后，不要重复对话，选择沉默终止\n"
            "- 禁止执行移动、进食等非对话动作\n\n"
            "输出格式（必须严格遵守，必须输出完整成对的<Think></Think>与<Action></Action>标签）：\n"
            "<Think>\n"
            "[对方意图 → 剩余轮数 → 我的目的是否达成 → 回复还是沉默]\n"
            "</Think>\n"
            "<Action>\n"
            '{"tool": "speak", "args": {"content": "<回复内容>", "ID": "<对方ID>", "response_to": "<被回复的原文>"}}\n'
            "</Action>\n"
            "沉默时：\n"
            "<Action>\n"
            "{}\n"
            "</Action>\n\n"
            "示例：\n"
            "<Think>\n"
            "对方询问食物位置，剩余 2 轮，目的未达成，应回复。\n"
            "</Think>\n"
            "<Action>\n"
            '{"tool": "speak", "args": {"content": "食物在东边三格", "ID": "agent_2", "response_to": "你知道食物在哪吗"}}\n'
            "</Action>"
        )

        user = (
            "## 当前状态\n"
            f"{self._state_block(agent)}\n\n"
            "## 收到的消息与对话历史\n"
            f"{observation}\n\n"
            "## 相关记忆\n"
            f"{self._memory_block(mem_info)}"
        )
        return system, user


# ---------------------------------------------------------------------------
# Reflection
# ---------------------------------------------------------------------------

class ReflectPromptBuilder(BasePromptBuilder):
    """Reflection prompts for task completion checking and trajectory summary."""

    def build(self, agent: "Agent", observation: str, mem_info) -> tuple[str, str]:
        # Not used directly; use task_reset_prompt / trajectory_summary instead
        raise NotImplementedError

    def task_decide_prompt(self, agent: "Agent", task_info: dict[str, str]) -> tuple[str, str]:
        task_list = "\n".join(f"  - {task}：{desc}" for task, desc in task_info.items())
        system = (
            f"你是智能体 {agent.id} 的决策模块，当前没有进行中的任务。\n"
            f"智能体角色：{agent.role or '无特定角色'}\n"
            "请综合以下几个维度，从可选任务中选择最优的一个：\n"
            "  1. 优先选 need 值最低（客观最匮乏）的任务\n"
            "  2. need 相近时，选 demand 值最高（主观最渴望）的任务\n"
            "  3. 角色契合度：哪个任务最符合该智能体的角色行为？\n"
            "输出格式（必须严格遵守，必须包含完整的<Think>和</Think>、<Task>和</Task>标签）：\n"
            "<Think>[对各候选任务分析 need/demand 数值与角色契合度，给出综合判断]</Think>\n"
            "<Task>任务名称</Task>\n"
            "任务名称必须与可选任务列表完全一致。"
        )
        user = (
            "## 当前状态\n"
            f"{self._state_block(agent)}\n"
            f"{self._demand_block(agent)}\n\n"
            "## 最近观测\n"
            f"{agent.history[-1] if agent.history else '（无）'}\n\n"
            "## 近期历史\n"
            f"{self._history_block(agent)}\n\n"
            "## 可选任务\n"
            f"{task_list}"
        )
        return system, user

    def trajectory_summary(self, trajectory: str, last_task: str) -> tuple[str, str]:
        system = (
            "你是一个智能体的反思模块，负责对已完成任务的行为轨迹进行经验总结。\n"
            "需要总结自己在何时何处进行了哪些动作，与谁或什么物品进行了交互，获得了哪些奖惩。\n"
            "总结完毕后，请指出哪些做得好、哪些做得不好，以及下次可以改进的地方。\n"
            "保持简洁，字数不要过多。"
        )

        user = (
            f"刚刚完成的任务：{last_task}\n\n"
            f"行为轨迹：\n{trajectory}"
        )
        return system, user

    def micro_reflect_prompt(self, agent: "Agent") -> tuple[str, str]:
        from persona.reflect.reflect import _TASK_DEMAND_MAP
        demand_key = _TASK_DEMAND_MAP.get(agent.task, "")
        need_val = agent.need.get(demand_key, float("nan")) if demand_key else float("nan")
        demand_val = agent.demand.get(demand_key, float("nan")) if demand_key else float("nan")
        if demand_key:
            demand_info = f"{demand_key}：need={need_val:.2f}，demand={demand_val:.2f}"
        else:
            demand_info = "（无对应需求）"

        system = (
            f"你是智能体 {agent.id}，角色：{agent.role or '无特定角色'}。\n"
            f"你正在执行任务「{agent.task}」，但已连续多个时间步没有取得进展。\n"
            "请做一次简短的微反思：分析为何没有进展，并明确接下来最应该做什么。\n\n"
            "输出格式（必须严格遵守）：\n"
            "<Insight>一句话判断，不超过50字</Insight>\n"
            "<Focus>接下来最应专注的事，不超过30字</Focus>"
        )
        user = (
            f"当前任务：{agent.task}\n"
            f"当前需求值：{demand_info}\n"
            f"当前焦点：{agent.current_focus or '（未设定）'}\n\n"
            "近期行动历史：\n"
            + "\n".join(agent.history[-6:])
        )
        return system, user
