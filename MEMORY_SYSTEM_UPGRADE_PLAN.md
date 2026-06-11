# 记忆系统升级计划

本文档记录当前记忆模块的升级方案、已完成阶段和后续阶段。内容以当前代码为准，核心入口位于 `backend/persona/agent_memory/mem.py`、`backend/persona/agents/agent.py`、`backend/persona/agents/prompt.py`、`backend/persona/reflect/reflect.py`。

## 目标

当前目标不是替换整个记忆后端，而是在保留 ChromaDB 向量存储的前提下，先解决三个问题：

1. 缩短每个时间步的记忆读取时间。
2. 让读出的记忆更贴合当前任务、需求和交互场景。
3. 让记忆以更明确的行动提示形式提供给 LLM，而不是只把原始文本塞进 prompt。

长期目标是把记忆系统从“简单向量召回”升级为“分层、可解释、可压缩、可评估”的智能体长期记忆模块。

## 当前读写链路

### 读取链路

1. `Agent.step()`、`Agent.social_step()`、`Agent.conversation_step()` 触发记忆读取。
2. `Agent.recall()` 根据场景传入 `context`：
   - `world`
   - `social`
   - `conversation`
3. `MultiAgentMemoryManager.smart_retrieve()` 基于 observation、当前 task、urgency、satisfaction_threshold 构造确定性检索 query。
4. `retrieve_agent_memories()` 从当前智能体的 Chroma collection 中按 `agent_id` 召回多条记忆。
5. 本地按 `context` 过滤 `memory_type`，再按相似度、重要性、任务匹配、需求匹配、置信度、时间因素重排。
6. `_format_memory()` 将记忆格式化为带元数据前缀的文本。
7. `PromptBuilder._memory_block()` 将记忆转换为行动提示，注入 `## 相关记忆` 区块。
8. `LLMPolicy.decide()` 将包含记忆的 system/user prompt 发送给 LLM。

### 写入链路

1. `Agent.remember()` / `Agent.aremember()` 是通用写入入口。
2. `store_agent_memory()` / `astore_agent_memory()` 负责规范化 metadata、生成 memory_id、获取 embedding、写入 Chroma。
3. 任务完成时，`Reflect.step()` 保存完成前的 `task` 和 `task_urgency_key`，再调用 `flush_trajectory()` 将轨迹总结存为 `episodic` 记忆。
4. 智能体卡住时，`_micro_reflect()` / `_amicro_reflect()` 将 `<Insight>` 存为 `reflective` 记忆。
5. 默认场景初始化时，`_seed_default_memories()` 将地图信息存为 `semantic` 记忆。

## 已完成阶段

### 阶段 1：读取性能优化

状态：已完成。

完成内容：

- 移除 `smart_retrieve()` 中的 LLM 关键词提取步骤。
- 改为 `_build_retrieval_query()` 直接基于 observation、task、urgency 构造检索 query。
- 新增 embedding 缓存，避免同一文本重复请求 embedding。
- 检索时先扩大召回数量，再在本地重排。

当前收益：

- 每次记忆读取减少一次 LLM 文本生成调用。
- 对同一 observation 或 query 的重复 embedding 调用会被缓存命中。
- 在不改变 Chroma 后端的情况下，降低单步响应时间。

对应代码：

- `backend/persona/agent_memory/mem.py`
  - `smart_retrieve()`
  - `_build_retrieval_query()`
  - `_get_embedding()`
  - `_aget_embedding()`

### 阶段 2：结构化记忆元数据

状态：已完成。

完成内容：

- 新增默认记忆类型 `DEFAULT_MEMORY_TYPE = "episodic"`。
- 新增兼容映射：
  - `trajectory` -> `episodic`
  - `micro_reflection` -> `reflective`
  - `system` -> `semantic`
- 统一写入 metadata：
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
  - 其他调用方传入字段
- `_sanitize_metadata()` 会移除 `None`，并将 list/tuple 转为逗号字符串，避免 Chroma metadata 类型不兼容。

当前收益：

- 后续可以基于 `memory_type`、`task`、`need_key` 进行检索筛选和重排。
- 旧写法 `type="trajectory"`、`type="micro_reflection"`、`memory_type="system"` 仍可被兼容。
- 记忆记录更容易解释和调试。

对应代码：

- `backend/persona/agent_memory/mem.py`
  - `_normalize_metadata()`
  - `_sanitize_metadata()`
  - `MEMORY_TYPE_ALIASES`

### 阶段 3：上下文感知检索

状态：已完成。

完成内容：

- 新增 `CONTEXT_MEMORY_TYPES`：
  - `world`: `semantic`, `procedural`, `episodic`, `reflective`
  - `social`: `social`, `semantic`, `episodic`, `reflective`
  - `conversation`: `social`, `episodic`, `reflective`, `semantic`
- `Agent.recall()` 支持传入 `context`。
- `social_step()` 使用 `context="social"`。
- `conversation_step()` 使用 `context="conversation"`。
- `retrieve_agent_memories()` 在本地按 context 过滤 `memory_type`。

当前收益：

- 普通地图行动、社交互动、对话不会使用完全相同的记忆集合。
- 不依赖 Chroma 的复杂 `$in` 查询能力，避免向量库版本差异导致检索失败。

对应代码：

- `backend/persona/agents/agent.py`
  - `recall()`
  - `arecall()`
  - `social_step()`
  - `conversation_step()`
- `backend/persona/agent_memory/mem.py`
  - `CONTEXT_MEMORY_TYPES`
  - `_memory_types_for_context()`
  - `_format_ranked_results()`

### 阶段 4：任务与需求感知重排

状态：已完成。

完成内容：

- `_memory_score()` 综合以下因素：
  - 向量相似度
  - `importance`
  - `task` 是否匹配
  - `saved_at` 带来的时间因素
  - `confidence`
  - `need_key` 与当前 urgency 的匹配
  - 简单重复惩罚
- `Agent._memory_top_k_for()` 根据智能体状态动态决定读取数量。
- 智能体卡住或存在 `current_focus` 时，读取更多记忆。
- 社交和对话场景降低读取数量，减少 prompt 噪声。

当前收益：

- 不是只按向量距离取前几条，而是把任务、需求和重要性纳入排序。
- 卡住时能更主动调出反思和过往经验。

对应代码：

- `backend/persona/agent_memory/mem.py`
  - `_memory_score()`
  - `_format_ranked_results()`
- `backend/persona/agents/agent.py`
  - `_memory_top_k_for()`

### 阶段 5：写入来源升级

状态：已完成。

完成内容：

- 任务完成轨迹总结写入为 `episodic`。
- 微反思写入为 `reflective`。
- 默认地图信息写入为 `semantic`。
- 任务完成时会先保存 `completed_need_key`，避免 `set_task("none")` 后丢失需求键。

当前收益：

- 轨迹、反思、地图信息在检索阶段可被区分。
- 后续可以针对不同记忆类型设计不同压缩和淘汰策略。

对应代码：

- `backend/persona/agents/agent.py`
  - `flush_trajectory()`
  - `aflush_trajectory()`
- `backend/persona/reflect/reflect.py`
  - `step()`
  - `astep()`
  - `_micro_reflect()`
  - `_amicro_reflect()`
- `backend/default_scenario.py`
  - `_seed_default_memories()`

### 阶段 6：Prompt 侧行动化记忆

状态：已完成。

完成内容：

- `_memory_block()` 不再只输出原始记忆列表。
- 根据记忆前缀增加行动提示：
  - `[semantic...]` -> `[地图/规则]`
  - `[procedural...]` -> `[行动流程]`
  - `[episodic...]` -> `[过往经验]`
  - `[reflective...]` -> `[反思]`
  - `[social...]` -> `[社交记忆]`
- 记忆统一注入到 `## 相关记忆` 区块。

当前收益：

- LLM 更容易理解某条记忆应该怎样影响本轮动作。
- 地图信息、失败经验、反思焦点不再混为普通文本。

对应代码：

- `backend/persona/agents/prompt.py`
  - `_memory_block()`
  - `_memory_action_hint()`
  - `WorldPromptBuilder`
  - `SocialPromptBuilder`
  - `ConversationPromptBuilder`

### 阶段 7：基础测试

状态：已完成。

完成内容：

- 新增 `backend/test_memory_system.py`。
- 覆盖：
  - embedding 缓存
  - metadata 规范化
  - Chroma where 形状
  - context 过滤和重排
  - `smart_retrieve()` 不调用 LLM 关键词提取
  - prompt 记忆行动提示

已验证命令：

```powershell
C:\Users\chen\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m py_compile backend\persona\agent_memory\mem.py backend\persona\agents\agent.py backend\persona\agents\prompt.py backend\persona\reflect\reflect.py backend\persona\config.py backend\default_scenario.py backend\test_memory_system.py
C:\Users\chen\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m unittest backend.test_memory_system
```

验证结果：

```text
Ran 6 tests in 0.001s
OK
```

## 后续阶段

### 阶段 8：写入筛选与去重

状态：未开始。

问题：

当前系统主要是“有总结就写入”，缺少写入前筛选。长期运行后会产生大量低价值、重复或过时记忆。

需要完成：

- 增加写入前评分：
  - 是否包含新信息
  - 是否与已有记忆重复
  - 是否和任务结果有关
  - 是否包含稳定地图、对象、社交关系或失败教训
- 增加相似记忆去重：
  - 相同 `memory_text` 已由 `memory_id` 间接覆盖。
  - 仍需处理语义重复但文本不同的情况。
- 增加低价值记忆跳过逻辑。

建议实现位置：

- `backend/persona/agent_memory/mem.py`
  - `store_agent_memory()`
  - `astore_agent_memory()`
  - 新增 `_should_store_memory()`
  - 新增 `_find_similar_memories()`

### 阶段 9：访问统计回写

状态：未开始。

问题：

当前 metadata 中已有 `last_accessed_at` 和 `access_count`，但检索命中后没有回写更新。

需要完成：

- 检索命中后更新：
  - `last_accessed_at`
  - `access_count`
- 将访问频率纳入 `_memory_score()`。
- 防止高频但低价值记忆长期垄断 prompt。

建议实现位置：

- `backend/persona/agent_memory/mem.py`
  - `_format_ranked_results()`
  - 新增 `_update_access_stats()`

### 阶段 10：记忆压缩与层级化

状态：未开始。

问题：

当前长期记忆仍然偏“平铺”。轨迹总结、反思、地图知识、社交关系都存在同一个 collection 中，只靠 `memory_type` 区分。

需要完成：

- 将短期轨迹压缩为 episodic summary。
- 将多条 episodic summary 周期性归纳为 reflective / procedural 记忆。
- 将稳定事实抽取为 semantic 记忆。
- 对社交互动抽取为 social 记忆。

推荐层级：

- `raw_event`：原始事件，可短期保留。
- `episodic`：一次任务或一段经历的总结。
- `reflective`：失败原因、策略调整、当前焦点。
- `procedural`：可复用行动流程。
- `semantic`：地图、规则、对象位置等稳定事实。
- `social`：关系、信任、互动历史、帖子相关经验。

建议实现位置：

- `backend/persona/reflect/reflect.py`
- `backend/persona/agent_memory/mem.py`
- 可新增独立压缩器，但需先确认是否放在现有 `mem.py` 内还是拆分模块。

### 阶段 11：任务结果与奖励驱动的记忆权重

状态：未开始。

问题：

当前 `importance` 和 `confidence` 多数是固定值，未充分利用 reward、任务完成情况和 satisfaction 增量。

需要完成：

- 根据任务结果动态计算 `importance`：
  - 成功完成任务的轨迹更高。
  - 卡住后的有效反思更高。
  - 没有带来收益的重复行为更低。
- 将 satisfaction 变化写入 metadata：
  - `satisfaction_before`
  - `satisfaction_after`
  - `satisfaction_delta`
  - `reward_total`
- 在 `_memory_score()` 中加入结果质量。

建议实现位置：

- `backend/persona/agents/agent.py`
  - `append_trajectory()`
  - `flush_trajectory()`
- `backend/persona/reflect/reflect.py`
- `backend/persona/agent_memory/mem.py`

### 阶段 12：关系记忆与社交记忆增强

状态：未开始。

问题：

当前 social context 可以读取 `social` 类型，但实际写入 social 记忆的路径还不完整。

需要完成：

- 评论、点赞、点踩、私信后写入 social 记忆。
- 记录：
  - 互动对象
  - 帖子 ID
  - 互动类型
  - 观点倾向
  - trust 变化
  - 对方回应
- 检索社交记忆时优先匹配当前帖子作者、当前话题和当前任务。

建议实现位置：

- `backend/tools/operator_tools.py`
  - `SocialOperator.comment_post()`
  - `SocialOperator.like_post()`
  - `SocialOperator.dislike_post()`
- `backend/persona/agents/agent.py`
- `backend/persona/agent_memory/mem.py`

### 阶段 13：地图语义记忆自动维护

状态：未开始。

问题：

当前默认地图记忆是在场景初始化时写入，后续对象变化、区域变化、道路变化没有自动更新为稳定语义记忆。

需要完成：

- 当地图对象创建、消失、位置变化时更新 semantic 记忆。
- 将区域、道路、建筑、物品归纳为可检索地图知识。
- 支持按区域和对象类型检索：
  - 某区域有哪些建筑
  - 某需求对应去哪里
  - 某道路如何到达目标

建议实现位置：

- `backend/default_scenario.py`
- `backend/world/map.py`
- `backend/world/world.py`
- `backend/persona/agent_memory/mem.py`

### 阶段 14：记忆预算与淘汰策略

状态：未开始。

问题：

长期运行后，每个智能体 collection 会持续增长。当前没有每个智能体的记忆预算、淘汰策略或归档策略。

需要完成：

- 增加每个智能体的最大活跃记忆数量。
- 对低分记忆进行归档或删除。
- 优先保留：
  - 高 importance
  - 高 confidence
  - 高频访问
  - 近期有效
  - 稳定地图事实
  - 成功任务流程
- 对旧 episodic 记忆做周期性压缩。

建议实现位置：

- `backend/persona/config.py`
  - 新增 memory budget 配置。
- `backend/persona/agent_memory/mem.py`
  - 新增 cleanup / compact 方法。

### 阶段 15：可观测性与调试面板

状态：未开始。

问题：

当前日志能看到部分检索结果，但缺少完整的记忆解释信息。调试时难以判断某条记忆为什么被召回、为什么排在前面、为什么进入 prompt。

需要完成：

- 日志记录每次检索：
  - query
  - context
  - fetch_n
  - returned_n
  - 每条记忆的 score 组成
  - 被过滤掉的 memory_type
- 前端或调试接口展示：
  - 当前智能体的记忆列表
  - 本轮召回记忆
  - prompt 中实际注入的记忆
  - 记忆来源和 metadata

建议实现位置：

- `backend/persona/agent_memory/mem.py`
- `backend/server.py`
- `frontend/src/components`

### 阶段 16：集成测试与回归场景

状态：未开始。

问题：

当前新增测试主要覆盖纯逻辑，没有覆盖真实仿真中的端到端效果。

需要完成：

- 增加端到端测试场景：
  - 智能体知道食物位置后不再反复询问。
  - 智能体卡住后能读取 reflective 记忆并改变行动。
  - 智能体完成任务后能在后续同类任务中复用 episodic 经验。
  - 社交平台只使用当前可见帖子 ID。
  - 地图 semantic 记忆能引导 move / enter_building。
- 增加性能基线：
  - 单步平均耗时。
  - 记忆检索耗时。
  - embedding 缓存命中率。
  - prompt 中记忆 token 数。

建议实现位置：

- `backend/test_memory_system.py`
- 新增端到端测试文件。
- 可扩展 `backend/test_spatial.py` 的脚本化验证方式。

## 推荐实施顺序

后续建议按以下顺序推进：

1. 阶段 9：访问统计回写。
2. 阶段 8：写入筛选与去重。
3. 阶段 11：任务结果与奖励驱动的记忆权重。
4. 阶段 12：关系记忆与社交记忆增强。
5. 阶段 13：地图语义记忆自动维护。
6. 阶段 10：记忆压缩与层级化。
7. 阶段 14：记忆预算与淘汰策略。
8. 阶段 15：可观测性与调试面板。
9. 阶段 16：集成测试与回归场景。

优先做访问统计和写入筛选，是因为这两项可以直接减少记忆污染，并为后续压缩、淘汰、可视化提供基础字段。

## 当前限制

1. 当前没有真实 Chroma 读写的集成测试，只验证了纯逻辑。
2. `last_accessed_at` 和 `access_count` 已写入 metadata，但还没有在读取后回写。
3. `social` 类型记忆已有检索支持，但写入路径仍不完整。
4. `procedural` 类型已有检索支持，但还没有稳定生成路径。
5. 记忆淘汰、归档和周期性压缩尚未实现。
6. 当前 prompt 侧只按格式化字符串前缀判断记忆类型，后续可以改为结构化 prompt 数据。

