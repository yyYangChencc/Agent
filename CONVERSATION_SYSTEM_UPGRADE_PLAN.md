# 对话系统升级计划

本文档记录当前项目对话行为的升级计划。当前阶段只记录设计，不修改业务代码。

计划参考 AutoGen 式多智能体对话流程中的几个核心思想：用会话对象管理上下文、用 group chat 管理发言顺序、用 selector 决定下一位发言者、用 termination conditions 明确结束条件、用消息流记录完整过程。本文档不会直接引入 AutoGen 框架，而是将这些机制转化为适合当前 2D 世界模拟的轻量实现。

## 当前对话机制

当前对话相关代码入口：

- `backend/world/world.py`
  - `World._conversation_phase()`
  - `World.execute()`
- `backend/persona/agents/agent.py`
  - `Agent.receive_message()`
  - `Agent.conversation_step()`
  - `Agent.inbox`
  - `Agent.conversation_opted_out`
- `backend/persona/agents/prompt.py`
  - `ConversationPromptBuilder`
- `backend/tools/operator_tools.py`
  - `Operator.speak()`

当前流程：

1. 主行动阶段中，智能体可以调用 `speak`。
2. `World.execute()` 将 `speak` 产生的消息投递给可感知范围内的目标智能体。
3. 被投递消息进入目标智能体的 `inbox`。
4. 每个 tick 的行动、反思、观点更新后，`World._conversation_phase()` 被触发。
5. `_conversation_phase()` 委托 `ConversationManager.run_phase()` 推进会话。
6. `ConversationManager` 从本 tick 收到的 `inbox` 消息中创建或复用 `ConversationSession`。
7. 每个 active session 每轮最多选择一个 speaker。
8. 被选中的智能体调用 `conversation_step()`。
9. `conversation_step()` 将 session 历史与 `inbox` 拼成 observation，读取 `context="conversation"` 的记忆，再调用 `ConversationPromptBuilder`。
10. LLM 返回 `speak` JSON 或空行动。
11. 返回 `speak` 时，manager 再次调用 `World.execute()` 投递回复，并将回复写入当前 session。
12. 达到终止条件或 `conversation_max_rounds` 后，对话阶段结束。

## 当前 `speak` 运行链条

本节记录当前代码中的 `speak` 全流程。`speak` 不是由单个函数完成的，而是分为“主行动阶段发起发言”和“对话阶段继续多轮回复”两段。`Operator.speak()` 本身只负责生成工具反馈和日志，不负责把消息送到其他智能体；实际投递发生在 `World.execute()`。

### 1. 运行时装配

1. `backend/persona/runtime.py` 中 `SimulationRuntime.build()` 创建三套策略：
   - `policy = LLMPolicy(llm, WorldPromptBuilder(), ActionParser())`
   - `social_policy = LLMPolicy(llm, SocialPromptBuilder(), ActionParser())`
   - `conv_policy = LLMPolicy(llm, ConversationPromptBuilder(), ActionParser())`
2. `world.conversation_policy = conv_policy`。
3. `world.conversation_max_rounds = conversation_max_rounds`。
4. 因此默认运行时，只要 `world.conversation_policy` 存在，每个 tick 都会在主行动、反思、观点更新后进入对话阶段。

### 2. 主行动阶段产生 `speak`

1. `World.astep()` 每个 tick 先执行所有智能体的主行动阶段。
2. 单个智能体的链条是：`observe(agent, agent.config.observation_radius)` -> `agent.astep(obs)` -> `world.execute(agent, action)`。
3. `Agent.astep()` 内部执行：`add_history("observation", observation)` -> `arecall(observation)` -> `policy.adecide(...)` -> `add_history("action", action)`。
4. `WorldPromptBuilder` 允许普通世界行动中使用 `speak`。它要求 LLM 在 `<Action>...</Action>` 中输出工具 JSON。
5. `LLMPolicy.adecide()` 调用 LLM 后，用 `ActionParser.parse_action_with_error()` 提取 `<Action>` 标签里的 JSON。解析失败时最多重试 `MAX_RETRIES = 2` 次。

主行动阶段的 `speak` action 形态仍是工具调用 JSON：

```json
{"tool": "speak", "args": {"content": "你知道食物在哪里吗", "ID": "agent_2"}}
```

### 3. `World.execute()` 执行工具并投递消息

1. `World.execute(agent, action_str)` 解析 action JSON。
2. 根据 `data["tool"]` 从 `self.tools` 中取得工具。
3. 将 `args["operator_ID"] = agent.id` 注入参数。
4. 调用 `tool.run(**args)`。
5. 当工具是 `speak` 时，实际执行的是 `Operator.speak(operator_ID, content, ID, response_to=None)`。
6. `Operator.speak()` 只返回反馈文本并写日志，不负责消息投递。
7. `World.execute()` 用工具反馈创建 `Event`，其中 `type=data["tool"]`，`actor=agent.id`，`info=feedback`，`time=self.time`，`position=agent.position`。
8. `World.execute()` 遍历所有其他智能体：
   - 先判断 `other.can_perceive(event)`。
   - 如果 `data["tool"] == "speak"`，再读取 `target_str = args.get("ID", "")`。
   - 当 `target_str == "<all>"` 或 `other.id in target_str.split()` 时，调用 `other.receive_message(...)`。
9. `Agent.receive_message()` 将消息写入 `self.inbox`，字段包括 `sender`、`content`、`response_to`、`time`、`session_id`、`intent`、`target`。

因此，`Operator.speak()` 返回成功只表示发言动作被执行；目标是否实际收到，取决于 `World.execute()` 中的感知范围和 `ID` 过滤。

### 4. 对话阶段接管后续回复

1. `World.astep()` 在主行动、反思、观点更新之后检查 `self.conversation_policy`。
2. 如果存在，就调用 `World._conversation_phase(agents)`。
3. `World._conversation_phase()` 只委托给 `self.conversation_manager.run_phase(agents, self.conversation_policy, self.conversation_max_rounds)`。
4. `ConversationManager.run_phase()` 每次开始时先将所有 agent 的 `conversation_opted_out` 重置为 `False`。
5. `_seed_sessions_from_inboxes()` 从各智能体 `inbox` 中读取主行动阶段收到的 `speak` 消息，创建或复用 `ConversationSession`。
6. 创建 session 时会生成 `session_id`、`participants`、`initiator`、`topic`、`intent`、`status`、`round`、`max_rounds`、`created_at`、`last_updated`。
7. `intent` 由 `infer_conversation_intent(content, response_to)` 根据内容和 `response_to` 推断。
8. `_seed_sessions_from_inboxes()` 会把 `session_id` 和 `intent` 回写到原始 inbox 消息中。
9. `_resolve_mutual_speaks()` 会处理同一主行动阶段互相说话导致的双向循环风险。

### 5. 选择发言人并生成回复

1. `ConversationManager.run_phase()` 在 `1..max_rounds` 范围内推进轮次。
2. 每一轮先调用 `_select_speakers(active_sessions, agents)`。
3. `_select_speakers()` 对每个 active session 最多选择一个 speaker，并确保同一轮同一个 speaker 不会被多个 session 重复选择。
4. `_select_speaker_for_session()` 的优先级：
   - 如果最后一条消息的 `target` 不是 `<all>`，优先选择该 target。
   - 然后选择 session participants 中除最后发送者以外的智能体。
   - 被选中的 agent 必须未 `conversation_opted_out`，`inbox` 非空，并且 `inbox` 中有该 `session_id` 的消息。
5. 被选中的 agent 调用 `Agent.conversation_step(policy, round_n, max_rounds, conv_history)`。
6. `Agent.conversation_step()` 将 session 历史和当前 `inbox` 组装为 observation。
7. `Agent.conversation_step()` 清空 `self.inbox`。
8. `Agent.conversation_step()` 调用 `recall(observation, context="conversation")` 读取对话相关记忆。
9. 随后调用 `policy.decide(...)`，即 `ConversationPromptBuilder` 对应的 LLM 策略。
10. `ConversationPromptBuilder` 要求 LLM 只能回复 `speak` JSON，或输出 `{}` 表示沉默。

对话阶段的回复 action 形态是：

```json
{"tool": "speak", "args": {"content": "回复内容", "ID": "agent_2", "response_to": "被回复的原文"}}
```

### 6. 回复消息再次进入投递链条

1. 如果 `conversation_step()` 返回空 action，agent 会被标记为 `conversation_opted_out = True`。
2. 如果返回非空 action，`ConversationManager.run_phase()` 解析 JSON。
3. 只有 `data.get("tool") == "speak"` 的 action 会继续处理。
4. manager 调用 `self.world.execute(agent, action)`。
5. 因此对话回复会再次走 `World.execute()` 的工具执行和消息投递逻辑。
6. manager 随后用 `_new_message()` 创建 `ConversationMessage`，并写入当前 `ConversationSession`。
7. session 的 `messages`、`participants`、`open_questions`、`known_facts` 会通过 `ConversationSession.add_message()` 更新。

### 7. 终止条件

当前已实现的终止路径：

1. `no_respondents`：没有可选择的发言人。
2. `speaker_opted_out`：被选中的 agent 沉默，且没有该 session 的待处理 inbox。
3. `explicit_end`：最后一条消息的 `intent` 是 `ConversationIntent.END`。
4. `resolved_intent`：session intent 是 `ConversationIntent.ASK_INFO`，且 session 中已经出现 `known_facts`。
5. `max_rounds`：达到 `conversation_max_rounds` 后 session 仍未结束。

### 8. 当前限制

1. `speak` 的工具反馈不等于实际送达结果；实际送达取决于 `other.can_perceive(event)` 和 `ID` 过滤。
2. 多目标 `ID` 依赖空格分隔，判断逻辑是 `other.id in target_str.split()`。
3. `<all>` 只会发送给能感知到该 `Event` 的其他智能体。
4. `Agent.conversation_step()` 当前会清空整个 `self.inbox`。如果同一个 agent 同时收到多个 session 的消息，存在把其他 session 消息一起清掉的风险。
5. 当前 `ConversationPromptBuilder` 仍要求直接输出 `speak` 工具 JSON，尚未升级为独立的对话决策 JSON。

## 当前问题

1. 对话已经有 `ConversationSession`，但 `Agent.conversation_step()` 仍直接消费整个 `inbox`，多 session 同时进入同一智能体时存在消息被一起清空的风险。
2. 对话缺少明确目标，问路、闲聊、说服、协调、回应通知都走同一个 prompt。
3. 多人对话已有 session 线程，但 `<all>` 广播后的发言顺序仍是基础规则选择，尚未结合任务紧急度、关系和记忆相关度。
4. 发言人选择已从并行响应改为每个 active session 每轮一个 speaker，但仍缺少更细的优先级评分。
5. 终止条件已覆盖 `no_respondents`、`speaker_opted_out`、`explicit_end`、`resolved_intent`、`max_rounds`，但还缺少重复内容、紧急任务、不可达、拒绝继续等规则。
6. 对话结束后没有统一总结，也没有稳定写入 `social` / `episodic` / `semantic` 记忆。
7. 关系、信任、观点差异没有充分进入对话决策。
8. `speak` 工具反馈只表示发言动作发生，不直接说明哪些智能体实际收到。
9. 缺少对话调试信息，例如会话 ID、话题、发言顺序、结束原因、写入记忆。

## AutoGen 式机制映射

本项目不需要完整接入 AutoGen，但可以借鉴它的流程结构。

| AutoGen 式机制 | 本项目映射 |
| --- | --- |
| Team / GroupChat | `ConversationSession` 管理一组参与者 |
| RoundRobinGroupChat | 双人对话或固定顺序多人对话 |
| SelectorGroupChat | 根据目标、话题、关系、最近发言选择下一位发言者 |
| TerminationCondition | 用明确规则结束会话 |
| Message stream | 统一记录 `ConversationMessage` |
| Agent description / role | 使用当前 `role`、`speaking_style`、`opinion`、trust 信息 |
| Shared context | 会话级 topic、intent、history、known_facts、open_questions |

核心思想是把“是否回复、谁回复、何时结束、是否写入记忆”从单个 LLM prompt 中拆出来，由明确的数据结构和管理器控制。

## 目标架构

### 新增核心对象

#### ConversationMessage

建议字段：

```python
{
    "message_id": str,
    "session_id": str,
    "round": int,
    "sender": str,
    "target": str,
    "content": str,
    "intent": str,
    "response_to": str | None,
    "time": int,
}
```

#### ConversationSession

建议字段：

```python
{
    "session_id": str,
    "participants": list[str],
    "initiator": str,
    "topic": str,
    "intent": str,
    "status": "active|resolved|expired|cancelled",
    "round": int,
    "max_rounds": int,
    "messages": list[ConversationMessage],
    "open_questions": list[str],
    "known_facts": list[str],
    "termination_reason": str | None,
    "created_at": int,
    "last_updated": int,
}
```

#### ConversationManager

建议职责：

1. 将 `speak` 消息归入已有 session 或创建新 session。
2. 维护每个 session 的参与者、话题、轮次和状态。
3. 选择下一位发言者。
4. 组装 `ConversationPromptBuilder` 需要的上下文。
5. 执行终止条件。
6. 对结束的 session 做总结和记忆写入。
7. 记录调试日志。

建议放置位置：

- 初期可放入 `backend/world/world.py` 附近，减少迁移成本。
- 稳定后可拆分为 `backend/persona/conversation/manager.py`、`backend/persona/conversation/session.py`。

## 对话意图分类

建议先引入固定枚举，减少 prompt 模糊度。

```text
ask_info       询问信息，例如位置、路线、对象状态
answer_info    回答信息
social_bonding 闲聊、维系关系
persuasion     观点影响或争论
coordination   协调行动，例如一起去某处
reaction       对刚收到的消息作出回应
notification   回应系统新闻或社交通知
end            主动结束对话
```

意图来源：

1. 主行动阶段 `speak` 可以先由 prompt 输出 intent。
2. 若旧格式暂时不变，可由规则从 content 中推断基础 intent。
3. 后续可加入轻量 LLM 分类器，但第一阶段不建议增加额外 LLM 调用。

## 发言人选择策略

参考 AutoGen 的 selector 思路，但第一版使用规则选择，不增加额外模型调用。

### 双人对话

规则：

1. 被点名者优先。
2. 上一条消息接收者优先。
3. 同一轮内已经发言者不重复发言。
4. 连续空行动后标记为退出。
5. 达到最大轮数后结束。

### 多人对话

规则：

1. `target` 明确指向某人时，该智能体优先。
2. `target="<all>"` 时，从可感知且未退出的参与者中选择。
3. 被直接提问的智能体优先。
4. 最近发言次数少的智能体优先。
5. 与话题相关记忆更多的智能体优先。
6. 当前任务紧迫度过高的智能体降低优先级。
7. 已 `conversation_opted_out` 的智能体跳过。

后续可加入 LLM selector：

```text
输入：session 摘要、参与者状态、最近消息、未解决问题
输出：下一位 speaker_id 或 END
```

## 终止条件

参考 AutoGen 的 termination conditions，建议显式实现以下结束规则：

1. `MaxRoundsTermination`：达到 `conversation_max_rounds`。
2. `NoResponderTermination`：没有智能体需要回复。
3. `ResolvedIntentTermination`：`ask_info` 已被回答，或 `coordination` 已达成一致。
4. `RepeatedContentTermination`：连续重复表达相同意思。
5. `OptOutTermination`：所有参与者主动沉默或退出。
6. `UrgencyTermination`：智能体当前任务需求过急，不继续闲聊。
7. `UnavailableTermination`：目标睡眠、离开、建筑内忙碌或不在感知范围。
8. `RefusalTermination`：对方明确拒绝继续交流。

结束时写入 `termination_reason`，方便日志和前端展示。

## Prompt 升级计划

当前 `ConversationPromptBuilder` 只要求智能体回复或沉默。升级后应让 prompt 接收更完整的 session 上下文。

### Prompt 输入新增

建议加入：

- `session_id`
- 当前 `topic`
- 当前 `intent`
- 本轮发言目标
- 未解决问题 `open_questions`
- 已知事实 `known_facts`
- 最近消息摘要
- 完整短历史
- 双方 `offline_trust`
- 双方 `online_trust`
- 双方 `opinion`
- 当前任务与需求状态
- 是否接近最大轮数
- 可用记忆

### Prompt 输出升级

建议输出结构：

```json
{
  "decision": "reply|end|ask_clarification",
  "target": "agent_2",
  "content": "回复内容",
  "intent": "answer_info",
  "conversation_done": false,
  "memory_worthy": true
}
```

执行层再将 `decision="reply"` 转换为现有 `speak` 工具调用。这样可以保持工具层稳定，同时让对话策略更清晰。

### Prompt 行为规则

新增规则：

1. 回答事实问题时，优先使用观察、记忆和已知事实。
2. 不知道时直接说不知道，不编造位置、对象 ID、帖子 ID。
3. 如果当前任务紧迫，简短回复后结束。
4. 如果对方的问题已经回答，不重复解释。
5. 如果对方表达感谢或确认，主动结束。
6. 如果话题涉及观点，说话风格要受 `role`、`speaking_style`、`opinion` 影响。
7. 如果 trust 较低，避免过度承诺和过度配合。
8. 如果是协调行动，必须明确地点、对象和下一步。

## 记忆接入计划

对话系统应与当前记忆系统衔接。

### 读取

当前已有：

- `conversation_step()` 使用 `context="conversation"`。
- `CONTEXT_MEMORY_TYPES["conversation"]` 包含 `social`、`episodic`、`reflective`、`semantic`。

后续增强：

1. query 中加入 `session.topic`、`intent`、对方 agent_id。
2. 优先召回与对方相关的 `social` 记忆。
3. 若是 `ask_info`，优先召回 `semantic` 记忆。
4. 若是重复冲突，优先召回 `reflective` 记忆。

### 写入

对话结束时根据内容写入不同类型：

```text
social     关系、信任、互动历史、观点交流
semantic   对话中获得的稳定事实
episodic   一次完整对话经历
reflective 对话失败、误解、重复追问的反思
```

写入示例：

```python
agent.remember(
    summary,
    memory_type="social",
    task=agent.task,
    object_id=other_agent_id,
    importance=0.6,
    confidence=0.7,
)
```

## 信任与观点影响

当前项目已有：

- `agent.online_trust`
- `agent.offline_trust`
- `agent.opinion`
- `OpinionUpdater`

对话升级后建议加入轻量规则：

1. 成功回答问题：提高接收者对回答者的 `offline_trust`。
2. 重复无效追问：降低对方 trust。
3. 观点相近且表达友好：小幅增加 trust。
4. 观点冲突且表达激烈：降低 trust。
5. 高 trust 对象的观点更容易影响 `opinion`。
6. 对话影响值应小于正式社交传播，避免一次聊天导致过大变化。

第一阶段先使用规则分数，不增加额外 LLM 评价器。

## 调试与前端展示

建议新增日志字段：

```text
tick
session_id
round
speaker
target
intent
decision
termination_reason
memory_written
trust_delta
opinion_delta
```

前端可展示：

1. 地图上的对话气泡。
2. 当前活跃 session 列表。
3. 最近对话历史。
4. 每次对话结束原因。
5. 对话写入的记忆摘要。

## 分阶段实施计划

### 阶段 1：会话数据结构

状态：已完成。

目标：

- 新增 `ConversationMessage`。
- 新增 `ConversationSession`。
- 保留当前 `inbox`，但将 `inbox` 消息转换为 session。
- 为每个 session 生成 `session_id`。
- 新增 `ConversationIntent` 与基础意图推断函数。

验收标准：

- 当前对话行为保持不退化。
- 日志中能看到 `session_id`、参与者、轮次、消息。

当前落地：

- 新增 `backend/persona/conversation/session.py`。
- 新增 `ConversationIntent`，包含 `ask_info`、`answer_info`、`social_bonding`、`persuasion`、`coordination`、`reaction`、`notification`、`end`。
- 新增 `ConversationMessage` 与 `ConversationSession`。
- `Agent.receive_message()` 支持携带 `session_id` 和 `intent`。
- `Agent.conversation_step()` 会在对话历史和当前消息中展示 `session` 与 `intent`。

### 阶段 2：ConversationManager

状态：已完成。

目标：

- 将 `_conversation_phase()` 中的会话推进逻辑迁移到 manager。
- manager 负责创建 session、推进轮次、结束 session。
- `World._conversation_phase()` 只调用 manager。

验收标准：

- `world.py` 中对话控制逻辑明显减少。
- session 可以跨多个轮次保存状态。

当前落地：

- 新增 `backend/persona/conversation/manager.py`。
- `ConversationManager.run_phase()` 兼容原有 `inbox + speak JSON` 流程。
- `World.__init__()` 初始化 `conversation_manager`。
- `World._conversation_phase()` 已改为委托 `conversation_manager.run_phase()`。
- manager 会从主行动阶段产生的 `inbox` 中创建 session，并为回复消息继续归入同一 session。
- 原并行 respondents 机制已在阶段 3 中替换为规则式发言人选择。

### 阶段 3：规则式发言人选择

状态：已完成。

目标：

- 实现双人轮流回复。
- 实现多人对话中的规则 selector。
- 避免所有待回复智能体无序并发回复。

验收标准：

- 同一 session 中每轮最多选择一个主要发言者。
- `<all>` 消息可以触发多人参与，但发言顺序可解释。

当前落地：

- `ConversationManager` 已从“所有 inbox agent 并发回复”改为“每个 active session 每轮选择一个 speaker”。
- 双人会话优先选择上一条消息的接收者。
- `<all>` 广播会话会合并为一个 session，参与者包含可接收广播的多个智能体。
- 同一轮内同一个 speaker 不会被多个 session 重复选择。
- 当前仍是规则式 selector，没有增加额外 LLM 调用。

### 阶段 4：终止条件

状态：部分完成。

目标：

- 实现明确 termination rules。
- 每次结束 session 时写入 `termination_reason`。

验收标准：

- 重复对话能被提前截断。
- 已回答问题不会持续追问。
- 达到最大轮数时有明确结束原因。

当前落地：

- 已实现 `no_respondents`。
- 已实现 `max_rounds`。
- 已实现 `speaker_opted_out`。
- 已实现 `explicit_end`。
- 已实现 `resolved_intent`：`ask_info` 会话出现回答后结束。

未完成：

- `RepeatedContentTermination`。
- `UrgencyTermination`。
- `UnavailableTermination`。
- `RefusalTermination`。

### 阶段 5：对话 Prompt 升级

状态：未开始。

目标：

- `ConversationPromptBuilder` 接收 session 上下文。
- prompt 输出结构从直接 `speak` 升级为对话决策 JSON。
- 执行层将 `reply` 决策转换为 `speak`。

验收标准：

- LLM 能明确输出回复、结束、追问澄清。
- 回复中能体现话题、关系、任务紧迫度。

### 阶段 6：对话记忆写入

状态：未开始。

目标：

- session 结束时生成摘要。
- 按内容写入 `social`、`semantic`、`episodic`、`reflective`。

验收标准：

- 询问信息后，稳定事实可进入 `semantic` 记忆。
- 重要互动可进入 `social` 记忆。
- 后续对话能检索到相关记忆。

### 阶段 7：信任与观点影响

状态：未开始。

目标：

- 根据对话结果调整 trust。
- 根据观点交流小幅影响 opinion。
- 写入对应日志。

验收标准：

- 帮助性回答会提高 trust。
- 重复无效对话会降低 trust。
- 观点影响可解释且幅度受限。

### 阶段 8：可视化与调试

状态：未开始。

目标：

- 后端暴露最近 session 信息。
- 前端展示对话气泡和对话日志。
- 日志展示 session 推进和结束原因。

验收标准：

- 调试时能看到一次对话从创建到结束的完整链路。
- 能确认某条记忆是否来自对话。

### 阶段 9：测试与回归

状态：未开始。

目标：

- 增加对话单元测试和端到端测试。

测试场景：

1. A 问 B 食物位置，B 知道则回答。
2. B 不知道时明确说不知道，不编造坐标。
3. A 得到答案后不继续重复追问。
4. 多人 `<all>` 对话中发言顺序稳定。
5. 达到最大轮数后 session 结束。
6. 对话结束后写入 social 或 semantic 记忆。
7. 高 urgency 智能体简短回复后退出。

## 推荐实施顺序

1. 阶段 1：会话数据结构。
2. 阶段 2：ConversationManager。
3. 阶段 4：终止条件。
4. 阶段 3：规则式发言人选择。
5. 阶段 5：对话 Prompt 升级。
6. 阶段 6：对话记忆写入。
7. 阶段 7：信任与观点影响。
8. 阶段 9：测试与回归。
9. 阶段 8：可视化与调试。

终止条件应尽早实现，因为它直接限制无意义多轮对话和重复回复；发言人选择和 prompt 升级可以在有 session 状态后逐步推进。

## 当前阶段结论

当前项目已经有对话阶段、消息投递、轮次限制和 conversation 记忆读取基础，但还不是完整的会话管理系统。下一步应先建立 `ConversationSession` 和 `ConversationManager`，把对话从“多个智能体 inbox 的临时处理”升级为“可追踪、可终止、可总结、可写入记忆”的会话流程。
