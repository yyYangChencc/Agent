from __future__ import annotations
import json
from typing import TYPE_CHECKING
from persona.opinion.scale import get_opinion_topic_definition

if TYPE_CHECKING:
    from persona.agents.agent import Agent


# 所有行为决策共用同一份世界规则，避免不同决策入口产生相互冲突的物品认知。
GENERAL_AGENT_BEHAVIOR_RULES = """## 智能体通用扮演与行为规则
以下规则在所有行为决策中持续有效；当前子模块的工具范围和输出格式仍以本次提示词为准。

### 身份与信息边界
- 你是 2D 网格世界中的自主智能体，只能依据当前状态、当前观测、相关记忆、近期历史和工具反馈决策。
- 不得编造对象 ID、智能体 ID、帖子 ID、坐标、库存、价格、收益、距离、人物状态或已经发生的事件。
- 所有标识符必须逐字使用输入中出现的原值；不得改写其大小写、前后缀、编号、格式或结构。
- 当前观测是当前时间步的直接证据；记忆用于补充当前未观测到的信息。二者冲突时采用当前观测，并根据最新工具反馈修正后续行动。

### 坐标、对象与建筑种类
- 世界坐标统一写作 [row, col]；调用 move 时，x 必须填写 row，y 必须填写 col。
- kind 是精确种类名。可交互物品种类为 food、bed；建筑种类为 company、food_shop、playground，building 是通用建筑基类。entity_type=object 是记忆查询分类，不是物品 kind。
- food：地图食物。只有目标 kind=food 且位于欧氏距离 sqrt(2) 内时才能调用 eat；成功后增加 satiety、减少该食物库存，库存归零后对象消失。
- bed：床。只有目标 kind=bed、存在空闲床位且位于欧氏距离 sqrt(2) 内时才能调用 sleep；目标带有 owner_agent_id 时，只有 ID 与 owner_agent_id 完全相同的智能体可以使用。进入睡眠后按时间步恢复 relax，睡眠结束前不能执行其他行动。处于建筑内部时必须先离开建筑才能睡觉。
- company：公司建筑。调用 enter_building 后立即工作一次；停留期间每个时间步继续自动工作。每次工作增加 money、消耗 relax，并产生工作对应的 esteem 变化；具体数值只采用观测或记忆中的 salary、relax_cost 和工具反馈。
- food_shop：食品店建筑。调用 enter_building 后立即尝试购买一次食物；停留期间每个时间步继续自动购买。库存充足且 money 不低于 price 时，每次扣除 price、增加 provide 对应的 satiety；余额不足或售罄时不会成功购买。
- playground：游乐场建筑。调用 enter_building 后立即娱乐一次；停留期间每个时间步继续自动娱乐。money 不低于 price 时，每次扣除 price、增加 provide 对应的 relax；余额不足时不会产生恢复效果。
- building：通用建筑。可以进入和离开，但只有观测、记忆或工具说明明确给出的具体效果才可作为决策依据。

### 世界交互规则
- 每轮只选择一个当前子模块允许的动作；不得把多个工具调用塞入一次 action。
- food 和 bed 是物品，分别使用 eat 和 sleep，不得对它们调用 enter_building。
- company、food_shop、playground 和明确标为 building 的对象使用 enter_building，不得对它们调用 eat 或 sleep。
- 建筑占据的坐标不可作为普通落脚点。已有 entrance 时优先 move 到 entrance；没有 entrance 时可 move 到建筑 position，寻路会停在最近可达位置，然后下一轮再调用 enter_building。
- enter_building 仅在目标建筑位于欧氏距离 sqrt(2) 内时调用。处于一个建筑内时，进入其他建筑前必须先调用 exit_building 或 move 离开当前建筑。
- inside_building_id 不是 none 时，必须检查继续停留的自动效果。任务已满足、余额不足、库存售罄或继续停留会造成不必要消耗时，立即离开建筑。
- move 在 relax 大于 0 时每移动一格都会消耗 relax，本次移动可能在途中因 relax 耗尽而停止；relax 为 0 时仍可低速移动，每次最多 5 格且 relax 保持为 0。
- speak 只用于距离 5 格内的线下交流；social_step 用于浏览或操作社交平台。两者不能替代已经可以直接执行的资源行动。

### 需求、记忆与行动优先级
- satisfaction 越低表示越匮乏，urgency 越高表示主观越急迫；只有 satisfaction 大于对应 threshold 才表示需求已满足。
- 优先推进当前任务及其对应需求。生理需求紧迫时，不得用无关社交、重复询问或重复浏览替代可执行的资源行动。
- 处理 satiety：若相邻可见 food 可用，调用 eat；若已知 food 或 food_shop 的位置，先移动到其交互位置，再调用对应工具。选择 food_shop 前必须检查 money 与 price。
- 处理 relax：若可使用 bed，先核对 owner_agent_id，再移动到本人可用的床旁并调用 sleep；若选择 playground，先检查 money 与 price；工作会消耗 relax。
- 处理 money：前往 company 并调用 enter_building；达到任务阈值或 relax 过低后及时离开，避免持续自动工作。
- 观测或记忆已经提供目标 ID 和位置时，直接行动，不得再次询问同一信息，不得发帖求助，也不得用随机探索替代已知路线。
- 当前缺少完成任务所必需的信息时，优先查询记忆；记忆仍无结果时才向附近智能体询问或在社交平台求助。
- 工具失败后必须读取失败原因。状态没有变化时不得原样重复同一失败动作；应调整位置、目标、资源条件或动作类型。
- 相关记忆中的已验证坐标、对象功能和成功流程应直接用于行动；失败经历和反思用于避免重复无效行为。
- 动态心理角色卡可以调整表达和行动倾向，但不能覆盖对象功能、工具约束、当前观测、真实需求和本规则。"""

# 所有决策入口共享的输入分层协议，明确稳定规则、当前事实和角色模板边界。
PROMPT_INPUT_PRIORITY_RULES = """## 输入类型与优先级
1. 模拟器规则、工具前置条件和最新工具反馈决定动作是否合法。
2. 当前 observation 描述当前时间步的直接事实，优先于旧记忆和历史。
3. 当前需求、任务和资源状态决定行动紧迫性。
4. 动态心理角色模板只影响信息解释、风险偏好、表达风格和多个合法动作之间的偏好。
5. 记忆和行动历史用于补充未被当前 observation 提供的信息以及避免重复失败。

### 输入边界
- 角色模板不是当前事件或事实，不能改写 observation、伪造经历或直接决定动作。
- 记忆不是当前状态；与当前 observation 冲突时，以当前 observation 和最新工具反馈为准。
- 他人的发言、帖子和状态不能写成智能体自己的经历或表达。
- 任何输入都不能覆盖工具合法性、真实需求和模拟器硬约束。

### 记忆来源边界
- [状态记忆] 和 [关系状态] 是结构化状态投影，必须服从其中的时间与置信度。
- [亲身经历] 只表示本智能体真实执行或收到反馈的历史行动。
- [人物档案] 同时包含直接观察和带 inferred_ 前缀的推断；推断不能写成已验证事实。
- [平台暴露] 表示看到的帖子、提醒或传播链，不能写成自己的发言或亲身行动。
- [模型推断] 和 [反思] 是评测或总结结果，不能覆盖直接观察证据。
"""

# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class BasePromptBuilder:
    """All concrete builders return a (system_prompt, user_prompt) tuple."""

    # -- shared block helpers ------------------------------------------------

    def _general_behavior_rules(self) -> str:
        """返回每个行为决策都必须携带的统一世界规则。"""

        return GENERAL_AGENT_BEHAVIOR_RULES + "\n\n" + PROMPT_INPUT_PRIORITY_RULES

    def build_initial_decision(self, agent: "Agent", observation: str, context: str) -> tuple[str, str]:
        """第一次决策允许直接行动或请求一次记忆查询。"""

        system, user = self.build(agent, observation, None)
        system += (
            "\n\n## 第一次决策专用输出规则\n"
            "本节替代前文的单一 action 输出要求。你必须只选择以下一种 JSON，不能同时输出 action 和 queries：\n"
            "1. 信息足够时直接输出行动："
            '{"think":"<理由>","action":{"tool":"<工具名>","args":{}}}\n'
            "2. 只有缺少会影响本轮行动的既有信息时，输出一次记忆查询："
            f'{{"think":"<查询理由>","context":"{context}","queries":[{{"type":"<查询类型>","intent":"<查询目的>","limit":3}}]}}\n'
            "查询类型只能是 person_profile、entity_state、event_history、social_post、derived_memory、semantic。\n"
            "查询对象字段必须沿用以下受控格式：\n"
            '{"type":"person_profile","intent":"understand_person","target_agent_ids":["agent_2"],"limit":2}\n'
            '{"type":"entity_state","intent":"find_unseen_need_target","entity_type":"object","kinds":["food"],"exclude_visible":true,"limit":5}\n'
            '{"type":"event_history","intent":"avoid_failed_actions","entity_ids":["food_1"],"source_types":["action_result"],"limit":3}\n'
            '{"type":"social_post","intent":"inspect_known_posts","post_ids":[1],"limit":3}\n'
            '{"type":"derived_memory","intent":"reuse_task_summary","memory_types":["episodic","procedural"],"limit":3}\n'
            '{"type":"semantic","intent":"recall_similar_experience","query":"自然语言检索文本","memory_types":["episodic","procedural","reflective","semantic"],"limit":3}\n'
            "queries 必须非空，最多 5 条，每条 limit 最大为 5；不得输出 SQL。"
        )
        user += "\n\n## 决策阶段\n这是第一次决策，当前尚未提供召回记忆。请直接行动，或输出一次受控记忆查询。"
        return system, user

    def build_after_memory_decision(
        self,
        agent: "Agent",
        observation: str,
        mem_info,
    ) -> tuple[str, str]:
        """第二次决策只允许根据查询结果输出行动。"""

        system, user = self.build(agent, observation, mem_info)
        system += (
            "\n\n## 查询后第二次决策专用输出规则\n"
            "记忆查询机会已经使用完毕。本次只能输出包含 think 和 action 的行动 JSON。\n"
            "禁止输出 context、queries 或任何新的记忆查询；信息仍不足时输出空 action。"
        )
        user += "\n\n## 决策阶段\n这是记忆查询后的第二次决策。只能根据当前输入和相关记忆输出行动。"
        return system, user

    def _state_block(self, agent: "Agent") -> str:
        lines = [
            f"- 位置：({agent.position[0]}, {agent.position[1]}) "
            f"| 时间步：t={agent.world.time}"
        ]
        personal_bed_id = getattr(agent, "personal_bed_id", None)
        if personal_bed_id is not None:
            lines.append(
                f"- 专属床铺：bed_id={personal_bed_id} | "
                f"position={agent.personal_bed_position} | entrance={agent.personal_bed_entrance} | "
                f"owner_agent_id={agent.id}；只能在此床调用 sleep"
            )
        return "\n".join(lines)

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
        profile = getattr(agent, "dataset_user_profile", {})
        if isinstance(profile, dict) and profile:
            argument_style = str(profile.get("argument_style") or "").strip()
            social_style = str(profile.get("social_style") or "").strip()
            if argument_style:
                lines.append(f"- 论证方式：{argument_style}")
            if social_style:
                lines.append(f"- 社交表达：{social_style}")
            raw_author = profile.get("raw_author")
            if isinstance(raw_author, dict):
                source_fields = [
                    f"{key}={value}"
                    for key, value in raw_author.items()
                    if value not in {None, ""}
                ]
                if source_fields:
                    lines.append(f"- 数据集原始资料：{'，'.join(source_fields)}")
            notes = profile.get("initialization_notes")
            if isinstance(notes, list):
                clean_notes = [str(note).strip() for note in notes if str(note).strip()]
                if clean_notes:
                    lines.append(f"- 初始化约束：{'；'.join(clean_notes)}")
        lines.append(f"- 当前情绪：{agent.emotion}")
        return "[ROLE_TEMPLATE_ONLY]\n" + "\n".join(lines) + "\n[/ROLE_TEMPLATE_ONLY]"

    def _role_card_instruction(self, agent: "Agent | None" = None) -> str:
        if agent is not None and (
            not getattr(agent.config, "dynamic_role_card_enabled", True)
            or not getattr(agent.config, "dynamic_role_card_behavior_enabled", True)
        ):
            return (
                "- 当前实验版本关闭动态心理角色卡；不得假设、补造或引用角色卡影响，"
                "只能依据当前观测、真实需求、记忆和工具规则决定动作。\n"
            )
        return (
            "- 必须按照“动态心理角色卡”调整本轮认知、表达、社交和行动倾向；"
            "若角色卡与工具规则、地图观测或真实需求冲突，以工具规则、地图观测和真实需求为准\n"
        )

    def _role_card_system_block(self, agent: "Agent") -> str:
        """将动态角色卡作为 system 层的稳定行为约束和当前内容。"""

        if (
            not getattr(agent.config, "dynamic_role_card_enabled", True)
            or not getattr(agent.config, "dynamic_role_card_behavior_enabled", True)
        ):
            return (
                "## 动态心理角色卡\n"
                "当前实验版本关闭行为角色卡；不得假设、补造或引用角色卡影响。"
            )
        return (
            "## 动态心理角色卡（稳定行为约束）\n"
            "角色卡只影响信息解释、风险偏好、表达风格以及多个合法动作之间的偏好；"
            "它不是当前事实，不能改写 observation、伪造经历、覆盖真实需求或工具规则。\n"
            f"{self._psychological_role_card_block(agent)}"
        )

    def _psychological_role_card_block(self, agent: "Agent") -> str:
        if not getattr(agent.config, "dynamic_role_card_enabled", True):
            return "（动态心理角色卡已在当前实验版本中关闭）"
        if not getattr(agent.config, "dynamic_role_card_behavior_enabled", True):
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
        """兼容非行动 Prompt，使用最近完整 tick 的结构化投影。"""

        entries = agent.short_term_memory.recent_entries(agent.config.short_term_memory_hot_ticks)
        if not entries:
            return "（无历史记录）"
        values = [entry.render() for entry in entries]
        return "\n".join(f"  {i + 1}. {value}" for i, value in enumerate(values[-max_n:]))

    def _short_term_memory_block(self, agent: "Agent") -> str:
        """渲染滚动总结和最近完整时间步。"""

        parts = []
        summaries = agent.short_term_memory.summaries()
        if summaries:
            summary = summaries[-1]
            parts.append(
                "### 较早经历的滚动总结\n"
                + json.dumps(summary.content, ensure_ascii=False, sort_keys=True, default=str)
            )
        else:
            parts.append("### 较早经历的滚动总结\n（暂无）")

        recent = agent.short_term_memory.recent_entries(agent.config.short_term_memory_hot_ticks)
        if not recent:
            parts.append("### 最近完整时间步\n（暂无）")
            return "\n\n".join(parts)
        grouped: dict[int, list[str]] = {}
        for entry in recent:
            # 当前 observation 已在行动 Prompt 中单独注入，避免同一事实重复占用上下文。
            if entry.record_type == "observation" and entry.world_time == int(agent.world.time):
                continue
            payload = entry.summary_payload()
            grouped.setdefault(entry.world_time, []).append(
                f"- {entry.record_type}: "
                + json.dumps(payload["content"], ensure_ascii=False, sort_keys=True, default=str)
            )
        if not grouped:
            parts.append("### 最近完整时间步\n（暂无额外历史记录）")
            return "\n\n".join(parts)
        tick_blocks = []
        for tick in sorted(grouped):
            tick_blocks.append(f"t={tick}\n" + "\n".join(grouped[tick]))
        parts.append("### 最近完整时间步\n" + "\n\n".join(tick_blocks))
        return "\n\n".join(parts)

    def _task_working_memory_block(self, agent: "Agent") -> str:
        progress = agent.task_working_memory()
        if not progress:
            return "（当前任务尚无轨迹）"
        return json.dumps(progress, ensure_ascii=False, sort_keys=True, default=str)

    def _memory_block(self, agent: "Agent", mem_info: list | str | None) -> str:
        if not mem_info:
            return "（无相关记忆）"
        per_item_limit = max(1, int(agent.config.memory_prompt_max_chars_per_item))
        total_limit = max(1, int(agent.config.memory_prompt_max_total_chars))
        if isinstance(mem_info, list):
            if not mem_info:
                return "（无相关记忆）"
            lines = ["可用记忆会影响本轮行动，应优先用于确定位置、对象、已验证流程和失败教训："]
            for memory in mem_info:
                # 单条记忆先截断，再限制全部记忆的总长度。
                hint = self._memory_action_hint(str(memory)[:per_item_limit])
                next_line = f"  - {hint}"
                if len("\n".join(lines + [next_line])) > total_limit:
                    break
                lines.append(next_line)
            return "\n".join(lines)[:total_limit]
        return str(mem_info)[:total_limit]

    def _memory_action_hint(self, memory: str) -> str:
        # 先按来源标记分流，避免把 observe 或平台暴露误写成亲身经历。
        direct_observation = (
            "source=observe" in memory
            or "source=direct_observation" in memory
            or "provenance=direct_observation" in memory
        )
        platform_exposure = (
            "source=social_browse" in memory
            or "source=social_notification" in memory
            or "provenance=platform_exposure" in memory
            or "provenance=platform_event" in memory
        )
        personal_social = (
            "source=social_feedback" in memory
            or "source=conversation" in memory
            or "provenance=self_social_action" in memory
        )
        personal_action = (
            "source=action_result" in memory
            or "source=need_event" in memory
            or "provenance=self_action" in memory
        )
        if platform_exposure:
            return f"[平台暴露] {memory}。保留作者、传播来源和提醒目标，不得写成自己的表达。"
        if personal_social:
            return f"[亲身社交经历] {memory}。这是本智能体真实执行或参与的社交历史。"
        if direct_observation:
            return f"[状态记忆] {memory}。这是直接观察到的事实投影，不是本智能体执行的经历。"
        if personal_action:
            return f"[亲身经历] {memory}。参考真实结果和需求变化，避免重复低收益行动。"
        if "[state" in memory:
            return f"[状态记忆] {memory}。这是带时间和置信度的状态投影；与当前观察冲突时采用当前观察。"
        if "[person_profile" in memory:
            return f"[人物档案] {memory}。observed_ 字段是观察记录，inferred_ 字段只是人物印象。"
        if "[relation" in memory:
            return f"[关系状态] {memory}。只用于理解当前关注、信任或互动关系。"
        if "[semantic" in memory:
            return f"[语义知识] {memory}。若其中含已验证坐标或对象ID，可用于合法行动。"
        if "[procedural" in memory:
            return f"[行动流程] {memory}。若当前任务匹配，应优先复用该流程。"
        if "[episodic" in memory:
            return f"[亲身经历] {memory}。参考真实结果和需求变化，避免重复低收益行动。"
        if "[reflective" in memory:
            return f"[模型推断/反思] {memory}。只能作为解释和策略参考，不能覆盖直接事实。"
        if "[social" in memory and (
            "provenance=self_social_action" in memory
            or "source=social_feedback" in memory
            or "source=conversation" in memory
        ):
            return f"[亲身社交经历] {memory}。这是本智能体真实执行或参与的社交历史。"
        if "[social" in memory:
            return f"[平台暴露] {memory}。保留作者、传播来源和提醒目标，不得写成自己的表达。"
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

    def build(
        self,
        agent: "Agent",
        observation: str,
        context: str = "world",
        recalled_memories: list[str] | None = None,
    ) -> tuple[str, str]:
        system = (
            f"你是自主智能体 {agent.id} 的记忆查询规划器，只负责决定本轮要查询哪些记忆。\n"
            "## 记忆规划规则\n"
            "只判断完成当前任务所必需的信息是否缺失，并生成受控查询；不执行行动，不改写 observation，不根据心理角色模板决定是否召回。\n\n"
            "你不是行动决策器，禁止输出 move、eat、speak、social_step、sleep、enter_building、exit_building 等世界动作。\n"
            "你不能输出 SQL、数据库语句、表扫描请求或任何未列出的查询类型。\n"
            "当前 observe、当前浏览内容或基础召回已经包含的信息不要重复查询：\n"
            "- 当前看见的物品/建筑已经有最新位置，不要为这些可见物品查询 entity_state。\n"
            "- 当前看见的人可以查询 person_profile，因为 observe 只提供位置，不提供印象和历史。\n"
            "- 当前输入的 social.notifications 若包含 [系统新闻]，说明系统刚投放高优先级新闻；应优先考虑查询 social_post、semantic 或 reflective 记忆来理解新闻背景。\n"
            "- 当前看不见但任务或需求需要的目标，可以查询 entity_state。\n"
            "- 需要过去经验、任务流程、反思、地图背景时，使用 semantic 查询 Chroma 长期记忆。\n"
            "- semantic 查询每轮最多 1 条；query 必须是一句聚焦的信息需求，不得复制 observation、帖子或评论原文，且最多 500 个字符。\n"
            "- 结构化查询中的 ID、类型或 kinds 列表每个最多 10 项，只保留完成当前任务必需的值。\n"
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
            "没有仍然缺失的信息时 queries 必须为空数组。queries 最多 5 条，每条 limit 最大 5。"
        )
        user = (
            "## 当前状态\n"
            f"{self._state_block(agent)}\n"
            f"- inside_building_id: {agent.inside_building_id or 'none'}\n"
            f"- 任务：{agent.task}\n"
            f"{self._urgency_block(agent)}\n"
            f"{self._focus_block(agent)}"
            f"{self._opinion_block(agent)}\n\n"
            f"## 当前上下文类型\n{context}\n\n"
            "## 当前输入\n"
            f"{observation}\n\n"
            "## 已完成的基础召回\n"
            f"{self._planner_memory_block(agent, recalled_memories or [])}\n\n"
            "## 近期历史\n"
            f"{self._planner_history_block(agent)}"
        )
        return system, user

    def _planner_history_block(self, agent: "Agent", max_n: int = 5) -> str:
        """排除原始观察，只给 planner 少量有界行动与查询历史。"""

        values = []
        entries = agent.short_term_memory.recent_entries(agent.config.short_term_memory_hot_ticks)
        for entry in reversed(entries):
            if entry.record_type in {"observation", "conversation"}:
                continue
            text = entry.render()
            values.append(text[:300])
            if len(values) >= max_n:
                break
        values.reverse()
        if not values:
            return "（无历史记录）"
        return "\n".join(f"  {index + 1}. {item}" for index, item in enumerate(values))[:1200]

    def _planner_memory_block(self, agent: "Agent", memories: list[str]) -> str:
        """只向规划器展示已召回内容，不附加行动建议。"""

        if not memories:
            return "（基础召回无结果）"
        per_item_limit = max(1, int(agent.config.memory_prompt_max_chars_per_item))
        total_limit = max(1, int(agent.config.memory_prompt_max_total_chars))
        lines = []
        for memory in memories:
            next_line = f"- {str(memory)[:per_item_limit]}"
            if len("\n".join(lines + [next_line])) > total_limit:
                break
            lines.append(next_line)
        return "\n".join(lines)[:total_limit] or "（基础召回无结果）"


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
            f"{self._general_behavior_rules()}\n\n"
            f"{self._role_card_system_block(agent)}\n\n"
            "你的目标是根据当前状态、观测和记忆，选择最合理的单一动作。\n\n"
            "行为原则：\n"
            "- 优先执行当前任务，通过行动提高对应satisfaction，若行动可顺便解决其他需求，可在不耽误当前主任务的前提下进行\n"
            "- 若任务为 none，根据需求自由决策\n"
            "- 若任务长时间无进展，优先依据当前焦点行动\n"
            "- 恢复规则：relax 不再每个时间步自动衰减；如果本轮没有 move 且没有工作，relax 会缓慢恢复\n"
            "- 根据当前情况选择最合适的动作：\n"
            "  · 有任务目标且知道目标位置时 → 使用 move 前往，配合 eat/sleep/enter_building/exit_building 完成任务\n"
            "  · 观测 social.notifications 中若出现 [系统新闻]，这是系统投放的高优先级新闻；可优先使用 social_step 查看平台详情、表达观点或据此调整行动，但不强制压过更紧急的生理/安全需求\n"
            "  · 想分享经历、表达观点、了解他人动态时 → 使用 social_step 浏览或发布内容\n"
            "  · 任务为 none 且所有需求已满足时，可自由选择任意动作\n"
            "- 记忆中的位置信息可直接用于行动；若当前观测给出更新信息，以当前观测为准\n\n"
            f"{self._role_card_instruction(agent)}"
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
            "## 观测（半径5格）\n"
            f"{observation}\n\n"
            "## 短期记忆\n"
            f"{self._short_term_memory_block(agent)}\n\n"
            "## 当前任务进展\n"
            f"{self._task_working_memory_block(agent)}\n\n"
            "## 相关记忆\n"
            f"{self._memory_block(agent, mem_info)}\n\n"
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
            f"{self._general_behavior_rules()}\n\n"
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
            "- comment_post / like_post / dislike_post / reply_comment / repost_post / quote_post 的 post_id 只能逐字复制本轮浏览 payload 的 visible_post_ids 中的整数\n"
            "- 调用 comment_post 时必须填写 post_id、content、agreement_to_post；agreement_to_post 只表示你有多认同被评论帖子本身，必须是 -1 到 1 的数值\n"
            "- agreement_to_post 不是你对帖主是否友善、是否尊重对方，也不是你的最终观念分数；观点认同、社交友善、尊重表达是三件不同的事\n"
            "- 调用 reply_comment 时，comment_id 必须逐字复制本轮浏览 payload 中目标帖子的 posts[].comments[].id；不得使用未在本轮展示的评论 ID\n"
            "- 调用 follow_author 时，author_id 必须逐字复制本轮浏览 payload 的 account_ids 中的字符串\n"
            "- 调用 unfollow_author 时，author_id 必须逐字复制本轮浏览 payload 的 following_ids；该字段与下方“当前关注列表”一致\n"
            "- 调用 repost_post / quote_post 时必须显式填写 opinion_index，表示你转发时对帖子 topic 的立场快照，必须是 -1 到 1 的数值\n"
            "- 禁止把发布时间、评论数量、列表顺序、作者编号、历史记忆中的帖子编号当成 post_id\n"
            "- 浏览时可以参与互动（点赞/评论），但不要为了参与而发布无新增内容的帖子\n\n"
            f"{self._role_card_instruction(agent)}"
            f"{self._action_format_block('{\"tool\": \"<tool_name>\", \"args\": {\"<key>\": <value>}}', '若本轮无可执行动作：')}\n\n"
            "示例：\n"
            '{"think": "我对姜萍事件形成了谨慎支持的看法，并且这不是复述当前帖子。", '
            '"action": {"tool": "send_post", "args": {"topic": "姜萍事件", "content": "我现在更倾向于先保护当事人，不要在证据不足时把质疑变成人身攻击。", "opinion_index": 0.45}}}\n'
            '{"think": "我部分认同这条帖子对证据链的重视，但不赞成它把所有支持者都说成盲信。", '
            '"action": {"tool": "comment_post", "args": {"post_id": 12, "content": "我同意需要证据链，但不应该把所有支持者都说成盲信。", "agreement_to_post": 0.2}}}\n'
            '{"think": "当前帖子没有让我形成新的看法，我也没有真实求助需求；为了避免重复别人说法，本轮不发帖。", '
            '"action": {}}'
        )

        system += "\n\n" + self._role_card_system_block(agent)

        user = (
            "## 当前状态\n"
            f"- 时间步：t={agent.world.time}\n"
            f"- 任务：{agent.task}\n"
            f"{self._urgency_block(agent)}\n"
            f"{self._focus_block(agent)}"
            f"{self._opinion_block(agent)}\n\n"
            "## 你的发帖历史\n"
            f"{agent.get_post_history() or '（暂无发帖记录）'}\n\n"
            "## 当前关注列表\n"
            f"{json.dumps(list(getattr(agent, 'followers', []) or []), ensure_ascii=False)}\n\n"
            "## 当前浏览的帖子\n"
            f"{posts_info}\n\n"
            "## 相关记忆\n"
            f"{self._memory_block(agent, mem_info)}\n\n"
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
            f"{self._general_behavior_rules()}\n\n"
            "你需要根据收到的消息和上下文，决定是回复还是保持沉默（沉默将终止对话）。\n\n"
            "行为原则：\n"
            "- 若剩余轮数为 0，必须主动收尾或保持沉默\n"
            "- 对话目的达成后，不要重复对话，选择沉默终止\n"
            "- 对话消息中若包含 session 或 intent，应优先围绕该会话线程和意图回复，避免混淆多个话题\n"
            "- 回答事实问题时，优先依据观察和记忆；不知道时直接说明不知道，不要编造坐标、对象ID或他人状态\n"
            "- 禁止执行移动、进食等非对话动作\n\n"
            f"{self._role_card_instruction(agent)}"
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
        system += "\n\n" + self._role_card_system_block(agent)
        user = (
            "## 当前状态\n"
            f"{self._state_block(agent)}\n\n"
            "## 收到的消息与对话历史\n"
            f"{observation}\n\n"
            "## 相关记忆\n"
            f"{self._memory_block(agent, mem_info)}"
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
            f"{self._general_behavior_rules()}\n\n"
            "请根据当前需求状态，自由决定一个最合适的任务名称，并指定该任务所针对的需求键。\n\n"
            "决策原则：\n"
            "  1. 优先针对 satisfaction 值最低（客观最匮乏）的需求\n"
            "  2. satisfaction 相近时，选 urgency 值最高（主观最渴望）的需求\n"
            "  3. 任务名称应简洁描述智能体接下来要做的事（例如：'寻找食物'、'前往休息'、'赚钱打工'）\n"
            "  4. 需求键必须是以下之一：satiety（饱腹度）、relax（放松度）、money（金钱）\n"
            "  5. 若存在动态心理角色卡，应按照其中的认知偏置、行为倾向和约束调整任务选择；"
            "角色卡关闭时不得假设或补造角色卡影响\n\n"
            "输出格式（必须严格遵守）：\n"
            "<Think>[分析各需求的 satisfaction/urgency 数值与紧迫程度，给出综合判断]</Think>\n"
            "<Task>任务名称</Task>\n"
            "<UrgencyKey>需求键</UrgencyKey>"
        )
        system += f"\n\n注意：当前真实可选需求键为：{valid_need_keys}。必须以此列表为准。"
        system += "\n\n" + self._role_card_system_block(agent)
        user = (
            "## 当前状态\n"
            f"{self._state_block(agent)}\n"
            f"{self._urgency_block(agent)}\n\n"
            "## 最近观测\n"
            f"{self._latest_observation_block(agent)}\n\n"
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

    def short_term_memory_summary(
        self,
        entries: list[dict],
        *,
        start_tick: int,
        end_tick: int,
        max_chars: int,
    ) -> tuple[str, str]:
        """构造只允许压缩既有事实的短期记忆总结提示词。"""

        system = (
            "你是短期记忆压缩器，只能压缩输入 JSON 中已经出现的事实。"
            "禁止补造实体 ID、位置、动作、反馈、奖励、任务结果或因果关系。"
            "历史位置和资源状态必须写成过去信息，不能描述为当前状态。"
            "必须保留成功动作、失败动作及其原始原因，并指出尚未完成的目标。"
            "referenced_entity_ids 只能逐字复制输入中已经出现的标识符。"
            "只输出一个合法 JSON 对象，不要输出 Markdown 或额外解释。"
        )
        user = json.dumps(
            {
                "start_tick": int(start_tick),
                "end_tick": int(end_tick),
                "max_output_chars": int(max_chars),
                "entries": entries,
                "output_schema": {
                    "start_tick": int(start_tick),
                    "end_tick": int(end_tick),
                    "chronology": "按时间顺序压缩的经历",
                    "task_progress": "任务进展",
                    "successful_actions": ["成功动作及结果"],
                    "failed_actions": ["失败动作及原始原因"],
                    "unresolved_goals": ["尚未完成的目标"],
                    "referenced_entity_ids": ["输入中原样出现的实体 ID"],
                },
            },
            ensure_ascii=False,
            sort_keys=True,
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
            f"{self._general_behavior_rules()}\n\n"
            "请做一次简短的微反思：分析为何没有进展，并明确接下来最应该做什么。\n\n"
            "输出格式（必须严格遵守）：\n"
            "<Insight>一句话判断，不超过50字</Insight>\n"
            "<Focus>接下来最应专注的事，不超过30字</Focus>"
        )
        system += "\n\n" + self._role_card_system_block(agent)
        user = (
            f"当前任务：{agent.task}\n"
            f"当前需求值：{urgency_info}\n"
            f"当前焦点：{agent.current_focus or '（未设定）'}\n\n"
            "当前任务工作记忆：\n"
            + self._task_working_memory_block(agent)
        )
        return system, user

    def _latest_observation_block(self, agent: "Agent") -> str:
        entry = agent.short_term_memory.latest({"observation"})
        return entry.render() if entry is not None else "（无）"
