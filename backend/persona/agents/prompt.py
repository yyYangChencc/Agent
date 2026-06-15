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

    def _urgency_block(self, agent: "Agent") -> str:
        d = agent.urgency
        n = agent.satisfaction
        t = agent.satisfaction_threshold
        lines = ["- 需求状态（satisfaction低=匮乏，urgency高=主观急迫；satisfaction>阈值=已满足）："]
        for k, urgency in d.items():
            satisfaction_val = n.get(k, 0.0)
            threshold = t.get(k)
            status = "✓" if isinstance(threshold, (int, float)) and satisfaction_val > threshold else "✗"
            threshold_text = f"{threshold:.1f}" if isinstance(threshold, (int, float)) else "未配置"
            lines.append(
                f"  {k}: satisfaction={satisfaction_val:.1f} urgency={urgency:.2f} threshold={threshold_text} {status}"
            )
        return "\n".join(lines)

    def _persona_block(self, agent: "Agent") -> str:
        lines = []
        if agent.role:
            lines.append(f"- 角色：{agent.role}")
        if agent.speaking_style:
            lines.append(f"- 说话风格：{agent.speaking_style}")
        lines.append(f"- 当前情绪：{agent.emotion}")
        return "\n".join(lines)

    def _action_format_block(self, action_schema: str, no_action_label: str) -> str:
        return (
            "输出格式（必须严格遵守，必须输出完整成对的<Think></Think>与<Action></Action>标签）：\n"
            "<Think>\n"
            "[分析过程]\n"
            "</Think>\n"
            "<Action>\n"
            f"{action_schema}\n"
            "</Action>\n"
            f"{no_action_label}\n"
            "<Action>\n"
            "{}\n"
            "</Action>"
        )

    def _history_block(self, agent: "Agent", max_n: int = 5) -> str:
        recent = agent.history[-max_n:]
        if not recent:
            return "（无历史记录）"
        return "\n".join(f"  {i + 1}. {h}" for i, h in enumerate(recent))

    def _memory_block(self, mem_info: list | str | None) -> str:
        if not mem_info:
            return "（无相关记忆）"
        if isinstance(mem_info, list):
            if not mem_info:
                return "（无相关记忆）"
            lines = ["可用记忆会影响本轮行动，应优先用于确定位置、对象、已验证流程和失败教训："]
            for memory in mem_info:
                lines.append(f"  - {self._memory_action_hint(memory)}")
            return "\n".join(lines)
        return mem_info

    def _memory_action_hint(self, memory: str) -> str:
        if "[semantic" in memory:
            return f"[地图/规则] {memory}。若其中含坐标或对象ID，可直接用于 move / enter_building / eat / sleep。"
        if "[procedural" in memory:
            return f"[行动流程] {memory}。若当前任务匹配，应优先复用该流程。"
        if "[episodic" in memory:
            return f"[过往经验] {memory}。参考其结果，避免重复低收益行动。"
        if "[reflective" in memory:
            return f"[反思] {memory}。若与当前卡住原因相同，应按其中焦点调整行动。"
        if "[social" in memory:
            return f"[社交记忆] {memory}。用于决定是否发帖、评论或联系相关智能体。"
        return memory

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
        persona_block = self._persona_block(agent)

        system = (
            f"你是自主智能体 {agent.id}，运行在一个 2D 网格世界中。\n"
            f"{persona_block}\n"
            "你的目标是根据当前状态、观测和记忆，选择最合理的单一动作。\n\n"
            "行为原则：\n"
            "- satisfaction越低表示越匮乏，urgency越高表示主观越急迫；satisfaction大于阈值才表示该需求已满足\n"
            "- 优先执行当前任务，通过行动提高对应satisfaction，若行动可顺便解决其他需求，可在不耽误当前主任务的前提下进行\n"
            "- 若任务为 none，根据需求自由决策\n"
            "- 若任务长时间无进展，优先依据当前焦点行动\n"
            "- 每轮只能执行一个工具调用\n"
            "- 建筑交互规则：建筑效果只通过 enter_building 后的自动 interact 获得\n"
            "- enter_building 成功后会立即自动 interact；之后只要 inside_building_id 不是 none，每个时间步都会自动 interact，直到执行 exit_building 或 move 离开建筑\n"
            "- 如果 inside_building_id 不是 none，思考时必须先判断当前需求和任务是否还需要留在建筑内；若已满足或继续停留会造成浪费，应调用 exit_building 或 move 离开\n"
            "- 根据当前情况选择最合适的动作：\n"
            "  · 有任务目标且知道目标位置时 → 使用 move 前往，配合 eat/sleep/enter_building/exit_building 完成任务\n"
            "  · 想分享经历、表达观点、了解他人动态时 → 使用 social_step 浏览或发布内容\n"
            "  · 确实不知道某信息（如食物/建筑位置）且记忆中也没有时 → 用 speak 询问附近智能体，或用 social_step 发帖求助\n"
            "  · 禁止：发帖询问你已知的信息（记忆中或观测到的建筑、食物、商店位置等）\n"
            "  · 任务为 none 且所有需求已满足时，可自由选择任意动作\n"
            "- 记忆中的位置信息是可靠的，直接使用，不需要反复确认\n\n"
            f"{self._action_format_block('{\"tool\": \"<tool_name>\", \"args\": {\"<key>\": <value>}}', '若本轮无可执行动作：')}\n\n"
            "示例：\n"
            "<Think>\n"
            "任务 eat something，satiety=10 未满足。记忆中 food_1 位于 (7,5)，当前位置 (3,5)，使用 move 前往。move 工具满足前提。\n"
            "</Think>\n"
            "<Action>\n"
            '{"tool": "move", "args": {"x": 7, "y": 5}}\n'
            "</Action>"
        )

        user = (
            "## 当前状态\n"
            f"{self._state_block(agent)}\n"
            f"- inside_building_id: {agent.inside_building_id or 'none'}\n"
            f"- 任务：{agent.task}\n"
            f"{self._urgency_block(agent)}\n"
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
        persona_block = self._persona_block(agent)

        system = (
            f"你是社交平台智能体 {agent.id}。\n"
            f"{persona_block}\n"
            "你的目标是通过社交平台获取有用信息、表达观点、建立社交联系。\n"
            "每轮只能执行一个社交动作。\n\n"
            "行为原则：\n"
            "- 适合发帖的时机：分享最近的行动经历或发现、对他人帖子发表看法、表达你的观点和立场\n"
            "- 发帖内容应与你的记忆和当前话题相关\n"
            "- 不允许重复发布完全相同的内容\n"
            "- 禁止发帖询问你已知的信息（如记忆中已有的建筑、食物位置等）\n"
            "- 仅对当前浏览内容里明确列出的帖子执行 comment_post / like_post / dislike_post\n"
            "- 调用 comment_post / like_post / dislike_post 时，post_id 必须逐字复制“当前可互动帖子ID列表”或“帖子ID”字段中的整数\n"
            "- 禁止把发布时间、评论数量、列表顺序、作者编号、历史记忆中的帖子编号当成 post_id\n"
            "- 浏览时积极参与互动（点赞/评论），而非只看不发\n\n"
            f"{self._action_format_block('{\"tool\": \"<tool_name>\", \"args\": {\"<key>\": <value>}}', '若本轮无可执行动作：')}\n\n"
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
            f"{self._urgency_block(agent)}\n"
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
        persona_block = self._persona_block(agent)

        system = (
            f"你是智能体 {agent.id}，当前正在进行对话。\n"
            f"{persona_block}\n"
            "你需要根据收到的消息和上下文，决定是回复还是保持沉默（沉默将终止对话）。\n\n"
            "行为原则：\n"
            "- 若剩余轮数为 0，必须主动收尾或保持沉默\n"
            "- 对话目的达成后，不要重复对话，选择沉默终止\n"
            "- 对话消息中若包含 session 或 intent，应优先围绕该会话线程和意图回复，避免混淆多个话题\n"
            "- 回答事实问题时，优先依据观察和记忆；不知道时直接说明不知道，不要编造坐标、对象ID或他人状态\n"
            "- 禁止执行移动、进食等非对话动作\n\n"
            f"{self._action_format_block('{\"tool\": \"speak\", \"args\": {\"content\": \"<回复内容>\", \"ID\": \"<对方ID>\", \"response_to\": \"<被回复的原文>\"}}', '沉默时：')}\n\n"
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

    def task_decide_prompt(self, agent: "Agent") -> tuple[str, str]:
        system = (
            f"你是智能体 {agent.id} 的决策模块，当前没有进行中的任务。\n"
            f"智能体角色：{agent.role or '无特定角色'}\n"
            "请根据当前需求状态，自由决定一个最合适的任务名称，并指定该任务所针对的需求键。\n\n"
            "决策原则：\n"
            "  1. 优先针对 satisfaction 值最低（客观最匮乏）的需求\n"
            "  2. satisfaction 相近时，选 urgency 值最高（主观最渴望）的需求\n"
            "  3. 任务名称应简洁描述智能体接下来要做的事（例如：'寻找食物'、'前往休息'、'赚钱打工'）\n"
            "  4. 需求键必须是以下之一：satiety（饱腹度）、relax（放松度）、money（金钱）\n\n"
            "输出格式（必须严格遵守）：\n"
            "<Think>[分析各需求的 satisfaction/urgency 数值与紧迫程度，给出综合判断]</Think>\n"
            "<Task>任务名称</Task>\n"
            "<UrgencyKey>需求键</UrgencyKey>"
        )
        user = (
            "## 当前状态\n"
            f"{self._state_block(agent)}\n"
            f"{self._urgency_block(agent)}\n\n"
            "## 最近观测\n"
            f"{agent.history[-1] if agent.history else '（无）'}\n\n"
            "## 近期历史\n"
            f"{self._history_block(agent)}"
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
        urgency_key = agent.task_urgency_key
        satisfaction_val = agent.satisfaction.get(urgency_key, float("nan")) if urgency_key else float("nan")
        urgency_val = agent.urgency.get(urgency_key, float("nan")) if urgency_key else float("nan")
        if urgency_key:
            urgency_info = f"{urgency_key}：satisfaction={satisfaction_val:.2f}，urgency={urgency_val:.2f}"
        else:
            urgency_info = "（无对应需求）"

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
            f"当前需求值：{urgency_info}\n"
            f"当前焦点：{agent.current_focus or '（未设定）'}\n\n"
            "近期行动历史：\n"
            + "\n".join(agent.history[-6:])
        )
        return system, user
