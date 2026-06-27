# 记忆系统重构实施方案

本文档记录当前记忆系统的完整设计和落地计划。核心结论是：Memory 不是 RAG。RAG 负责静态知识召回，Memory 负责动态、个性化、可更新、可遗忘、可冲突调解的状态管理。

## 目标

当前系统从单层 Chroma 文本召回升级为混合记忆系统：

- SQLite 保存结构化事实、事件、状态、关系和证据链。
- Chroma 继续保存适合 LLM 阅读的语义摘要。
- Memory Controller 统一负责写入闸门、结构化写入、双写、查询计划和访问记录。
- Prompt 暂时仍消费 `list[str]`，避免一次性重写决策提示词。

当前代码落点：

- `backend/persona/agent_memory/mem.py`：兼容入口，保留 Chroma 接口并挂接结构化记忆。
- `backend/persona/agent_memory/structured_store.py`：SQLite 结构化存储。
- `backend/persona/agent_memory/controller.py`：Memory Controller。
- `backend/persona/agents/agent.py`：observe、social、conversation、trajectory 写入和 routed recall。
- `backend/world/world.py`：action result 写入。
- `backend/world/observer.py`：结构化 observe 来源。
- `backend/social_sys/platform/platform.py`：结构化 social browse 和 feedback 来源。
- `backend/persona/opinion/assessment.py`：opinion evidence 写入和专用查询 context。
- `backend/server.py`：记忆 API 返回结构化分组。

## 分层模型

### Working Memory

只服务当前 tick/session，不直接作为长期记忆写入。

包括当前 observe JSON、当前 social browse、`agent.history` 近期窗口、`trajectory_buffer`、`inbox`、conversation history、当前心理角色卡、需求状态和 opinion assessment 上下文。

### Episodic Memory

保存具体经历：任务轨迹、观察到的动作、对话事件、行动结果、社交互动和观点变化经历。

### Semantic Memory

保存稳定事实和抽象结论：地图、建筑、物品功能、长期偏好、稳定关系、topic 立场摘要。

### Procedural Memory

保存可复用流程：如何满足某个需求、如何完成某类任务、失败后应避免的动作、社交平台互动规则。

### Reflective Memory

保存反思和策略调整：micro reflection、失败原因、卡住原因、心理角色卡相关解释。

### Social Memory

保存社交平台状态：帖子、作者、互动数、评论、点赞、踩、自己和他人的社交反馈、在线信任变化证据。

## 存储设计

### SQLite 结构化层

新增 `StructuredMemoryStore`，默认数据库路径为 `./chroma_agents/structured_memory.sqlite3`。

表结构：

- `memory_events`：原始事件流，保存 observe、action_result、conversation、social feedback、opinion assessment。
- `entity_states`：人物、物品、建筑的最新状态。
- `entity_relations`：人物关系、动作关系、对话关系、帖子作者关系、社交互动关系。
- `social_posts`：以 `agent_id + post_id` 保存帖子快照。
- `scene_snapshots`：以 `agent_id + world_time` 保存场景观察。
- `derived_memories`：保存长期归纳后的 episodic、semantic、procedural、reflective 记忆。
- `memory_conflicts`：保存冲突事实、调解结果和证据链。
- `memory_access_log`：保存检索访问时间、访问次数和命中场景。

每条结构化记忆都带有所属智能体、类型、来源、时间、置信度、重要性、有效状态和原始 payload。

### Chroma 语义层

Chroma 只保存语义摘要，继续服务自然语言召回。复杂 JSON 不放入 Chroma metadata，完整 payload 保存在 SQLite。

Chroma metadata 保留标量字段：

- `agent_id`
- `saved_at`
- `last_accessed_at`
- `access_count`
- `importance`
- `confidence`
- `memory_type`
- `task`
- `need_key`
- `object_id`
- `post_id`
- `entity_id`

## 写入机制

长期写入由 Memory Controller 统一处理，业务模块只提交结构化 payload。

当前已接入的写入时机：

- observe 后写人物、物品建筑、动作、场景、通知。
- `World.execute()` 后写 `think/action/tool/args/feedback/reward`。
- social browse 后写可见帖子快照。
- social feedback 后写发帖、点赞、踩、评论结果。
- conversation 后写收到的消息、回复和会话上下文。
- task 完成后保留现有 episodic summary，并写结构化 task summary action result。
- micro reflection 继续通过 `remember()` 写 Chroma，并同步写入 `derived_memories`。
- opinion assessment 后写 topic、score、reason、evidence。

写入闸门原则：

- 新人物、新物品、新建筑、位置变化、新动作、新对话、新帖子、新社交互动、任务完成、卡住反思、观点评估证据直接写入。
- 重复看到同一静态实体只更新时间和状态。
- 无信息量重复 observation、完全一致事实、无法解析来源/时间/主体的碎片文本不提升为长期语义摘要。

## 更新、冲突与遗忘

当前已实现的基础规则：

- 时间变化事实进入 `entity_states` 最新状态。
- 旧状态保留在 `memory_events` 和 `memory_conflicts` 中。
- `derived_memories` 维护 `valid/access_count/last_accessed_at`。
- 结构化检索命中后写入 `memory_access_log`。

后续增强：

- 稳定事实冲突进入 reconciliation。
- 规则无法解决时由 LLM 生成调解结论。
- 周期性 consolidation 合并重复 episodic，提升 semantic/procedural。
- 低重要性、低置信度、长期未访问 raw event 降权或归档。
- 失效事实不参与默认检索，但保留证据链。

## 查询机制

`Agent.recall()` 已改为优先调用 `retrieve_context()`。没有新接口的测试 stub 会回退到原来的 `smart_retrieve()`。

### world 查询

查询顺序：

1. 从当前 observe 中提取人物、物品、动作、区域。
2. 查 `entity_states` 最新状态。
3. 查 observe/action/conversation 相关 `memory_events`。
4. 查 task 相关 `derived_memories`。
5. 补充 Chroma 语义召回。

### social 查询

查询顺序：

1. 按当前 `visible_post_ids` 查 `social_posts`。
2. 查 social browse、social feedback、opinion assessment 事件。
3. 查近期社交帖子。
4. 补充 Chroma 语义召回。

### conversation 查询

查询顺序：

1. 按 sender/提到的实体查人物状态。
2. 查人物关系和历史对话事件。
3. 补充 Chroma 语义召回。

### opinion assessment 查询

查询顺序：

1. 查 opinion assessment、social feedback、social browse、conversation 事件。
2. 查近期社交帖子。
3. 补充 Chroma 语义召回。

## API

`/api/agents/{agent_id}/memories` 保留旧字段：

- `agent_id`
- `count`
- `memories`

新增字段：

- `structured`

`structured` 内包含：

- `events`
- `entity_states`
- `entity_relations`
- `social_posts`
- `scene_snapshots`
- `derived_memories`
- `memory_conflicts`
- `memory_access_log`

## 评测体系

新增和扩展 `backend/test_memory_system.py`，覆盖：

- SQLite schema 初始化。
- observe/social/action payload 写入。
- current state 更新。
- social post 以 `post_id` 精确查询。
- routed recall 先返回结构化状态，再补充 Chroma 语义结果。
- Chroma metadata 标量化和旧接口兼容。

后续评测应继续补：

- 写入准确率。
- 写入精度。
- 检索准确率。
- 时间一致性。
- 冲突处理。
- 遗忘正确性。
- 长期一致性。
- 行为收益。

最小场景：

- 同一物品位置变化。
- 同一人物多次对话。
- 同一帖子多次互动。
- 稳定事实冲突。
- 长时间重复观察静态场景。
- agent 卡住后反思并改变行动。
- opinion 因新闻、评论、对话证据变化。

## 当前实施状态

已完成：

1. 新增 SQLite 结构化存储骨架。
2. 新增 Memory Controller。
3. `MultiAgentMemoryManager` 保留旧接口，并增加结构化写入、查询和 list 接口。
4. observe、social、conversation、trajectory、action result、opinion assessment 接入结构化写入。
5. `Agent.recall()` 切换到查询计划，返回值仍为 `list[str]`。
6. 记忆 API 增加结构化分组。
7. 记忆系统测试扩展到结构化写入和 routed recall。

未完成：

1. LLM reconciliation 冲突调解。
2. 周期性 memory consolidation。
3. 遗忘/归档/预算清理策略。
4. 前端分类展示 UI。
5. 更完整的长期运行 eval 和行为收益评测。

## 后续实施顺序

1. 完成冲突调解：区分时间变化事实和稳定事实。
2. 增加 consolidation：从 raw event 提升 semantic/procedural。
3. 增加遗忘和预算：按 importance、confidence、access_count、last_accessed_at 降权或归档。
4. 增加前端分类查看：按人物、场景、社交、反思、背景分组。
5. 增加长期运行评测：70/100 tick 后检查检索质量和 prompt 噪声。
