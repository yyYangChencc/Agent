from __future__ import annotations
from typing import TYPE_CHECKING
from persona.opinion.scale import get_opinion_topic_definition

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
                f"  {k}: satisfaction={satisfaction_val:.1f} urgency={urgency:.2f} "
                f"threshold={threshold_text} gap={agent.need_gap.get(k, 0.0):.2f} "
                f"load={agent.load_saturation.get(k, 0.0):.2f} "
                f"pressure={agent.effective_pressure.get(k, 0.0):.2f} {status}"
            )
        return "\n".join(lines)

    def _persona_block(self, agent: "Agent") -> str:
        lines = []
        if agent.speaking_style:
            lines.append(f"- 说话风格：{agent.speaking_style}")
        lines.append(f"- 当前情绪：{agent.emotion}")
        return "\n".join(lines)

    def _role_card_instruction(self) -> str:
        return (
            "- 必须按照“动态心理角色卡”调整本轮认知、表达、社交和行动倾向；"
            "若角色卡与工具规则、地图观测或真实需求冲突，以工具规则、地图观测和真实需求为准\n"
        )

    def _psychological_role_card_block(self, agent: "Agent") -> str:
        if not getattr(agent.config, "dynamic_role_card_enabled", True):
            return "（动态心理角色卡已在当前实验版本中关闭）"
        assessment = agent.last_psychological_assessment
        if not isinstance(assessment, dict):
            return "（暂无动态心理角色卡）"
        role_card = assessment.get("role_card_delta")
        if not isinstance(role_card, dict) or not role_card:
            return "（暂无动态心理角色卡）"

        lines = []
        tick = assessment.get("tick")
        if tick is not None:
            lines.append(f"- 生成时间步：t={tick}")
        status = assessment.get("status")
        if status:
            lines.append(f"- 评测状态：{status}")
        activated_needs = role_card.get("activated_needs") or assessment.get("activated_needs")
        if activated_needs:
            lines.append(f"- 激活需求：{', '.join(str(x) for x in activated_needs)}")
        dominant_need = role_card.get("dominant_need")
        if dominant_need:
            lines.append(f"- 主导需求：{dominant_need}")
        summary = role_card.get("summary")
        if summary:
            lines.append(f"- 心理摘要：{summary}")
        emotion_tone = role_card.get("emotion_tone")
        if emotion_tone:
            lines.append(f"- 情绪基调：{emotion_tone}")

        field_labels = [
            ("cognition", "认知偏置"),
            ("behavior", "行为倾向"),
            ("social_expression", "表达风格"),
            ("online_behavior", "线上行为倾向"),
            ("decision_bias", "决策偏置"),
            ("constraints", "约束"),
        ]
        for field, label in field_labels:
            values = role_card.get(field)
            if isinstance(values, str):
                values = [values]
            if isinstance(values, list) and values:
                lines.append(f"- {label}：")
                for value in values[:4]:
                    lines.append(f"  · {value}")

        mediator_focus = role_card.get("mediator_focus")
        if isinstance(mediator_focus, list) and mediator_focus:
            focus_text = []
            for item in mediator_focus[:4]:
                if isinstance(item, dict):
                    key = item.get("key")
                    value = item.get("value")
                    need_key = item.get("need_key")
                    if key is not None and value is not None:
                        prefix = f"{need_key}." if need_key else ""
                        focus_text.append(f"{prefix}{key}={value}")
            if focus_text:
                lines.append(f"- 主要心理中介：{'; '.join(focus_text)}")

        if not lines:
            return "（暂无动态心理角色卡）"
        return "\n".join(lines)

    def _action_format_block(self, action_schema: str, no_action_label: str) -> str:
        return (
            "输出格式（必须严格遵守）：\n"
            "- 只输出一个合法 JSON 对象字符串\n"
            "- 不要输出 Markdown、额外解释、<Think> 或 <Action> 标签\n"
            "- 顶层必须包含 think 与 action 字段\n"
            "- think 必须说明本次选择动作的思考过程和理由\n"
            "- 可执行动作格式：\n"
            f'{{"think": "<思考过程和选择理由>", "action": {action_schema}}}\n'
            f"- {no_action_label}\n"
            '{"think": "<不行动的思考过程和理由>", "action": {}}\n'
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
        # 观念分数解释来自主题注册表，避免多新闻主题时 prompt 仍沿用旧语义。
        topic = getattr(agent.config, "default_opinion_topic", "")
        topic_definition = get_opinion_topic_definition(topic)
        return (
            f"- 当前观念主题：{topic_definition.topic}\n"
            f"- 当前观念倾向：{agent.opinion:.3f}（{topic_definition.direction_prompt}）"
        )

    def _focus_block(self, agent: "Agent") -> str:
        if not agent.current_focus:
            return ""
        return f"- 当前焦点：{agent.current_focus}\n"

    def build(self, agent: "Agent", observation: str, mem_info) -> tuple[str, str]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Memory planning
# ---------------------------------------------------------------------------

class MemoryPlannerPromptBuilder(BasePromptBuilder):
    """为行动前的记忆查询计划生成 prompt。"""

    def build(self, agent: "Agent", observation: str, context: str = "world") -> tuple[str, str]:
        system = (
            f"你是自主智能体 {agent.id} 的记忆查询规划器，只负责决定本轮要查询哪些记忆。\n"
            "你不是行动决策器，禁止输出 move、eat、speak、social_step、sleep、enter_building、exit_building 等世界动作。\n"
            "你不能输出 SQL、数据库语句、表扫描请求或任何未列出的查询类型。\n"
            "当前 observe 或当前浏览内容已经包含的信息不要重复查询：\n"
            "- 当前看见的物品/建筑已经有最新位置，不要为这些可见物品查询 entity_state。\n"
            "- 当前看见的人可以查询 person_profile，因为 observe 只提供位置，不提供印象和历史。\n"
            "- 当前输入的 social.notifications 若包含 [系统新闻]，说明系统刚投放高优先级新闻；应优先考虑查询 social_post、semantic 或 reflective 记忆来理解新闻背景。\n"
            "- 当前看不见但任务或需求需要的目标，可以查询 entity_state。\n"
            "- 需要过去经验、任务流程、反思、地图背景时，使用 semantic 查询 Chroma 长期记忆。\n"
            "只输出一个合法 JSON 对象字符串，不要输出 Markdown、额外解释或标签。\n"
            "输出 schema：\n"
            "{\n"
            '  "think": "为什么本轮需要查询这些记忆",\n'
            f'  "context": "{context}",\n'
            '  "queries": [\n'
            '    {"type": "person_profile", "intent": "understand_person", "target_agent_ids": ["agent_2"], "limit": 2},\n'
            '    {"type": "entity_state", "intent": "find_unseen_need_target", "entity_type": "object", "kinds": ["food"], "exclude_visible": true, "limit": 5},\n'
            '    {"type": "event_history", "intent": "avoid_failed_actions", "entity_ids": ["food_1"], "source_types": ["action_result"], "limit": 3},\n'
            '    {"type": "social_post", "intent": "inspect_known_posts", "post_ids": [1], "limit": 3},\n'
            '    {"type": "derived_memory", "intent": "reuse_task_summary", "memory_types": ["episodic", "procedural"], "limit": 3},\n'
            '    {"type": "semantic", "intent": "recall_similar_experience", "query": "自然语言检索文本", "memory_types": ["episodic", "procedural", "reflective", "semantic"], "limit": 3}\n'
            "  ]\n"
            "}\n"
            "queries 最多 5 条，每条 limit 最大 5。"
        )
        user = (
            "## 当前状态\n"
            f"{self._state_block(agent)}\n"
            f"- inside_building_id: {agent.inside_building_id or 'none'}\n"
            f"- 任务：{agent.task}\n"
            f"{self._urgency_block(agent)}\n"
            f"{self._focus_block(agent)}"
            f"{self._opinion_block(agent)}\n\n"
            "## 动态心理角色卡\n"
            f"{self._psychological_role_card_block(agent)}\n\n"
            f"## 当前上下文类型\n{context}\n\n"
            "## 当前输入\n"
            f"{observation}\n\n"
            "## 近期历史\n"
            f"{self._history_block(agent)}"
        )
        return system, user


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
            "- 移动规则：move 不再有单次移动距离上限，但每移动一格都会消耗 relax；若 relax 降为 0，本次移动会停止，之后不能继续 move\n"
            "- 恢复规则：relax 不再每个时间步自动衰减；如果本轮没有 move 且没有工作，relax 会缓慢恢复\n"
            "- 工作规则：公司工作会获得 money 但消耗 relax；relax 很低时应避免长时间工作，除非 money 需求更紧急\n"
            "- 根据当前情况选择最合适的动作：\n"
            "  · 有任务目标且知道目标位置时 → 使用 move 前往，配合 eat/sleep/enter_building/exit_building 完成任务\n"
            "  · 观测 social.notifications 中若出现 [系统新闻]，这是系统投放的高优先级新闻；可优先使用 social_step 查看平台详情、表达观点或据此调整行动，但不强制压过更紧急的生理/安全需求\n"
            "  · 想分享经历、表达观点、了解他人动态时 → 使用 social_step 浏览或发布内容\n"
            "  · 确实不知道某信息（如食物/建筑位置）且记忆中也没有时 → 用 speak 询问附近智能体，或用 social_step 发帖求助\n"
            "  · 禁止：发帖询问你已知的信息（记忆中或观测到的建筑、食物、商店位置等）\n"
            "  · 任务为 none 且所有需求已满足时，可自由选择任意动作\n"
            "- 记忆中的位置信息是可靠的，直接使用，不需要反复确认\n\n"
            "- 必须按照“动态心理角色卡”调整本轮认知、表达、社交和行动倾向；"
            "若角色卡与工具规则、地图观测或真实需求冲突，以工具规则、地图观测和真实需求为准\n"
            f"{self._action_format_block('{\"tool\": \"<tool_name>\", \"args\": {\"<key>\": <value>}}', '若本轮无可执行动作：')}\n\n"
            "示例：\n"
            '{"think": "任务 eat something，satiety=10 未满足。记忆中 food_1 位于 (7,5)，当前位置 (3,5)，使用 move 前往。move 工具满足前提。", '
            '"action": {"tool": "move", "args": {"x": 7, "y": 5}}}'
        )

        user = (
            "## 当前状态\n"
            f"{self._state_block(agent)}\n"
            f"- inside_building_id: {agent.inside_building_id or 'none'}\n"
            f"- 任务：{agent.task}\n"
            f"{self._urgency_block(agent)}\n"
            f"{self._focus_block(agent)}"
            f"{self._opinion_block(agent)}\n\n"
            "## 动态心理角色卡\n"
            f"{self._psychological_role_card_block(agent)}\n\n"
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
            "- 适合发帖的时机：你确实有自己的当前看法、真实想法、亲身经历、行动发现，或有明确求助需求\n"
            "- 发帖前必须判断：这条内容是否提供了你的新增观点、个人理由、真实状态或求助信息；如果没有，就不要调用 send_post\n"
            "- 发帖不是复述、改写或模仿当前浏览帖子和其他智能体刚说过的话，也不是简单重复新闻摘要\n"
            "- 如果只是泛泛赞同、泛泛反对、没有新增理由、没有真实求助，优先选择 like_post / dislike_post / comment_post 或本轮不行动\n"
            "- 发帖内容应与你的记忆、当前话题、需求状态或动态心理角色卡相关，并体现“我为什么这样想”或“我现在需要什么帮助”\n"
            "- 使用 send_post 时必须填写 topic、content、opinion_index；如果内容与当前观念主题相关，topic 必须使用当前观念主题，否则为普通帖子自行定义简短主题\n"
            "- opinion_index 表示你在发布这条帖子时对 topic 的立场快照，必须是 -1 到 1 的数值：-1=强烈反对/负向，0=中立/不关心/不确定，1=强烈支持/正向\n"
            "- 不允许重复发布与你历史发帖相同或只是换一种说法的内容\n"
            "- 禁止发帖询问你已知的信息（如记忆中已有的建筑、食物位置等）\n"
            "- 仅对当前浏览内容里明确列出的帖子执行 comment_post / like_post / dislike_post\n"
            "- 调用 comment_post 时必须填写 post_id、content、agreement_to_post；agreement_to_post 只表示你有多认同被评论帖子本身，必须是 -1 到 1 的数值\n"
            "- agreement_to_post 不是你对帖主是否友善、是否尊重对方，也不是你的最终观念分数；观点认同、社交友善、尊重表达是三件不同的事\n"
            "- 调用 comment_post / like_post / dislike_post 时，post_id 必须逐字复制“当前可互动帖子ID列表”或“帖子ID”字段中的整数\n"
            "- 禁止把发布时间、评论数量、列表顺序、作者编号、历史记忆中的帖子编号当成 post_id\n"
            "- 浏览时可以参与互动（点赞/评论），但不要为了参与而发布无新增内容的帖子\n\n"
            f"{self._role_card_instruction()}"
            f"{self._action_format_block('{\"tool\": \"<tool_name>\", \"args\": {\"<key>\": <value>}}', '若本轮无可执行动作：')}\n\n"
            "示例：\n"
            '{"think": "我对姜萍事件形成了谨慎支持的看法，并且这不是复述当前帖子。", '
            '"action": {"tool": "send_post", "args": {"topic": "姜萍事件", "content": "我现在更倾向于先保护当事人，不要在证据不足时把质疑变成人身攻击。", "opinion_index": 0.45}}}\n'
            '{"think": "我部分认同这条帖子对证据链的重视，但不赞成它把所有支持者都说成盲信。", '
            '"action": {"tool": "comment_post", "args": {"post_id": 12, "content": "我同意需要证据链，但不应该把所有支持者都说成盲信。", "agreement_to_post": 0.2}}}\n'
            '{"think": "当前帖子没有让我形成新的看法，我也没有真实求助需求；为了避免重复别人说法，本轮不发帖。", '
            '"action": {}}'
        )

        user = (
            "## 当前状态\n"
            f"- 时间步：t={agent.world.time}\n"
            f"- 任务：{agent.task}\n"
            f"{self._urgency_block(agent)}\n"
            f"{self._focus_block(agent)}"
            f"{self._opinion_block(agent)}\n\n"
            "## 动态心理角色卡\n"
            f"{self._psychological_role_card_block(agent)}\n\n"
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
            f"{self._role_card_instruction()}"
            f"{self._action_format_block('{\"tool\": \"speak\", \"args\": {\"content\": \"<回复内容>\", \"ID\": \"<对方ID>\", \"response_to\": \"<被回复的原文>\"}}', '沉默时：')}\n\n"
            "示例：\n"
            '{"think": "对方询问食物位置，剩余 2 轮，目的未达成，应回复。", '
            '"action": {"tool": "speak", "args": {"content": "食物在东边三格", "ID": "agent_2", "response_to": "你知道食物在哪吗"}}}'
        )

        system += (
            "\n\n线下对话结构化要求：\n"
            "- 回复时只能使用 speak；沉默时输出空 action。\n"
            "- speak.args 必须包含 ID、content、response_to、intent、social_valence、topic、topic_stance。\n"
            "- intent 表示这句话的社会功能，只能取：ask_info、answer_info、ask_help、offer_help、social_bonding、self_disclosure、emotional_support、coordination、persuasion、disagreement、conflict、thanks、apology、reaction、notification、end。\n"
            "- social_valence 表示你对接收者的社会态度，范围 [-1,1]：负值表示敌意或贬损，0 表示中性，正值表示支持、尊重或亲近。\n"
            "- topic 只在谈论系统新闻主题时填写；否则填写空字符串。\n"
            "- topic_stance 只在 topic 等于系统新闻主题时填写 [-1,1]；否则填写 null。\n"
            "- topic_stance 是对新闻主题的立场，social_valence 是对人的态度，二者必须分开。\n"
            "- 示例：{\"think\":\"对方表达压力，我给出支持性回应。\",\"action\":{\"tool\":\"speak\",\"args\":{\"ID\":\"agent_2\",\"content\":\"我理解你现在有点撑不住，可以先去休息一下。\",\"response_to\":\"我有点撑不住了\",\"intent\":\"emotional_support\",\"social_valence\":0.8,\"topic\":\"\",\"topic_stance\":null}}}\n"
        )
        user = (
            "## 当前状态\n"
            f"{self._state_block(agent)}\n\n"
            "## 动态心理角色卡\n"
            f"{self._psychological_role_card_block(agent)}\n\n"
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
        valid_need_keys = "、".join(agent.satisfaction.keys())
        system = (
            f"你是智能体 {agent.id} 的决策模块，当前没有进行中的任务。\n"
            "请根据当前需求状态，自由决定一个最合适的任务名称，并指定该任务所针对的需求键。\n\n"
            "决策原则：\n"
            "  1. 优先针对 satisfaction 值最低（客观最匮乏）的需求\n"
            "  2. satisfaction 相近时，选 urgency 值最高（主观最渴望）的需求\n"
            "  3. 任务名称应简洁描述智能体接下来要做的事（例如：'寻找食物'、'前往休息'、'赚钱打工'）\n"
            "  4. 需求键必须是以下之一：satiety（饱腹度）、relax（放松度）、money（金钱）\n"
            "  5. 若存在动态心理角色卡，应按照其中的认知偏置、行为倾向和约束调整任务选择\n\n"
            "输出格式（必须严格遵守）：\n"
            "<Think>[分析各需求的 satisfaction/urgency 数值与紧迫程度，给出综合判断]</Think>\n"
            "<Task>任务名称</Task>\n"
            "<UrgencyKey>需求键</UrgencyKey>"
        )
        system += f"\n\n注意：当前真实可选需求键为：{valid_need_keys}。必须以此列表为准。"
        user = (
            "## 当前状态\n"
            f"{self._state_block(agent)}\n"
            f"{self._urgency_block(agent)}\n\n"
            "## 动态心理角色卡\n"
            f"{self._psychological_role_card_block(agent)}\n\n"
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
            f"你是智能体 {agent.id}。\n"
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
            "动态心理角色卡：\n"
            f"{self._psychological_role_card_block(agent)}\n\n"
            "近期行动历史：\n"
            + "\n".join(agent.history[-6:])
        )
        return system, user
