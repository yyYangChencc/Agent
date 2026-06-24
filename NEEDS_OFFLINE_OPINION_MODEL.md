# 需求缺口、时间累积与线下观念变化模型

本文档整理“需求缺口随时间累积，并与线下社交传播共同驱动智能体观念变化”的一套可实现建模方案。目标是避免把瞬时需求状态直接映射为观点，而是在中间加入时间记忆与心理中介层。

## 1. 时间化需求缺口

对智能体 `i` 的每个需求维度 `m`，先定义当前 `satisfaction`：

```text
s_im(t) in [0, 1]
```

设该需求的舒适阈值为 `theta_m`，则瞬时需求缺口为：

```text
g_im(t) = max(0, theta_m - s_im(t))
```

含义：

- `s_im(t)` 越低，说明当前满足程度越差
- `g_im(t)` 只在低于舒适阈值时才为正
- 该值是“瞬时缺口”，还没有反映持续时间

## 2. 持续压力状态

为了表示“刚出现缺口时影响不大，但持续越久越严重”，为每个需求引入一个持续压力状态 `h_im(t)`：

```text
if g_im(t) > 0:
    h_im(t + dt) = h_im(t) + g_im(t) * dt
else:
    h_im(t + dt) = (1 - rho_m * dt) * h_im(t)
```

其中：

- `dt` 是时间步长
- `rho_m` 是该需求恢复期的衰减速度
- `h_im(t)` 同时记住了缺口强度和持续时长

直观上：

- 需求刚出现缺口时，`h_im(t)` 还很小
- 如果缺口持续存在，`h_im(t)` 会逐步累积
- 需求恢复后，`h_im(t)` 不会瞬间清零，而是缓慢回落

## 3. 有效需求压力

不直接将持续压力 `h_im(t)` 映射为心理中介，而是先构造有效需求压力 `p_im(t)`。原因是 `h_im(t)` 更适合作为“慢变量的历史负荷记忆”，而不是“当前正在作用的需求驱动力”。

如果直接使用 `h_im(t) -> 心理中介`，会混淆两类信息：

- 当前是否仍然存在需求缺口
- 过去一段时间内是否长期处于缺口状态

例如，智能体刚恢复进食后，`g_im(t) = 0`，但 `h_im(t)` 仍可能较高。如果直接将 `h_im(t)` 映射为匮乏感，就会把“曾经长期匮乏”误当成“当前仍在强烈匮乏”。更稳妥的做法是让 `h_im(t)` 作为放大器或残余脆弱性的来源，而不是完全替代当前缺口 `g_im(t)`。

因此，将瞬时缺口 `g_im(t)` 和持续压力 `h_im(t)` 组合成有效需求压力 `p_im(t)`：

```text
s_im(t) = 1 - exp(-h_im(t) / kappa_m)

p_im(t) = alpha_m * g_im(t)
        + gamma_m * s_im(t)
        + delta_m * g_im(t) * s_im(t)
```

其中：

- `alpha_m`：当前缺口的即时作用强度
- `gamma_m`：长期负荷留下的残余脆弱性强度
- `delta_m`：长期负荷对当前缺口的放大强度
- `kappa_m`：持续压力转化为放大/残余效应的时间尺度

解释：

- `alpha_m * g_im(t)` 表示刚出现缺口时就会产生即时影响
- `gamma_m * s_im(t)` 表示即使当前缺口减弱，长期负荷仍可能留下残余心理影响
- `delta_m * g_im(t) * s_im(t)` 表示相同大小的当前缺口，在长期累积后会被放大
- `s_im(t) = 1 - exp(-h_im(t) / kappa_m)` 是一个单调有界的饱和项，用来避免负荷效应无限增长

如果需要更保守的第一版实现，可令 `gamma_m = 0`，退化为“当前缺口 + 负荷放大”的简化模型：

```text
p_im(t) = alpha_m * g_im(t) + delta_m * g_im(t) * s_im(t)
```

这时 `h_im(t)` 只作为当前缺口的放大器，不单独产生残余影响。

## 4. 从需求压力到心理中介

在本模型中，`p_im(t)` 才是更适合映射到心理中介的量，而不是直接使用 `h_im(t)`。判据如下：

- `g_im(t)` 表示当前需求偏离程度
- `h_im(t)` 表示历史负荷与持续时长
- `p_im(t)` 将两者组合为“当前心理上真正感受到的有效压力”

这与相关文献中的共同思路一致：当前反应通常同时受到“当前刺激”和“既往累积负荷”的影响，而不是由其中任一者单独决定。

不直接让 `p_im(t)` 决定观念，而是先映射到心理中介变量。可定义如下：

```text
scarcity_i = a1 * p_i,phys + a2 * p_i,safety
threat_i = b1 * p_i,safety + b2 * scarcity_i
affiliation_i = c1 * p_i,belonging + c2 * threat_i
status_defense_i = d1 * p_i,esteem + d2 * scarcity_i
growth_block_i = e1 * p_i,self_actualization + e2 * p_i,esteem
```

推荐心理含义：

- `scarcity_i`：匮乏感
- `threat_i`：威胁敏感
- `affiliation_i`：依附与从众倾向
- `status_defense_i`：面子与地位防御
- `growth_block_i`：成长受阻感

这一步的作用是把“需求缺口”转化为“认知和情绪偏置”，使后续观念变化更像真实社会心理过程。

## 4.1 理论依据与公式来源

本节中的变量拆分和组合方式有明确的理论来源，但 `p_im(t)` 的具体函数形式不是直接摘自某一篇论文，而是基于以下几类文献做的工程化整合：

- `g_im(t)` 的思想来自稳态调节、驱力理论和 homeostatic reinforcement learning：行为应当响应当前内部状态与目标设定点之间的偏差。
- `h_im(t)` 的思想来自 allostasis 和 allostatic load：长期、重复、持续的压力会形成累积负荷，并改变后续对当前刺激的反应方式。
- `g_im(t)` 与 `h_im(t)` 的组合，体现的是“当前缺口 + 累积负荷 + 负荷调节当前反应”的结构。
- `1 - exp(-h_im(t) / kappa_m)` 属于常见的有界饱和非线性，用于表达“前期增长快、后期趋于饱和”的负荷效应。这是工程上常用的稳定化处理，而非特定心理学文献中的标准唯一公式。

因此，本文档中的 `p_im(t)` 应视为“有理论依据的建模选择”，而不是“文献原样公式”。

## 4.2 参考文献

与当前结构直接相关的参考文献包括：

- Keramati, M., & Gutkin, B. (2014). Homeostatic reinforcement learning for integrating reward collection and physiological stability. 该文强调内部状态偏离设定点对行为驱动的重要性。<https://elifesciences.org/articles/04811>
- McEwen, B. S. (1998). Protective and damaging effects of stress mediators. 该文系统阐述了 allostasis 与 allostatic load。<https://pubmed.ncbi.nlm.nih.gov/9629234/>
- Ganzel, B. L., Morris, P. A., & Wethington, E. (2010). Allostasis and the human brain. 该文明确讨论了既往负荷如何调节当前压力反应。<https://pmc.ncbi.nlm.nih.gov/articles/PMC2808193/>
- French, K. A., et al. (2024). Stress pile-up. 该文强调当前压力通常是近期一系列压力源累积的结果。<https://pmc.ncbi.nlm.nih.gov/articles/PMC12353331/>

如果要进一步提高理论贴合度，建议在实现中保留 `gamma_m` 这一残余项，而不是只保留 `g_im(t) * s_im(t)` 的放大型结构。

## 5. 观念表征评测

不再使用线下邻居均值或线上帖子均值的公式化观念增量。观念状态由评测模块在每个 tick 末根据上下文记录：

```text
current topic
+ recent experiences
+ recent posts / comments
+ memory
+ psychological role card
+ current opinion
-> opinion assessment
-> topic-specific opinion score
```

这里的重点不是手工计算一个 `opinion` 增量，而是让心理中介影响智能体的信息选择、解释、表达和记忆检索，再由观念评测器记录该主题下的观念表征。

## 6. 需求驱动表征

需求驱动表征由心理中介构成，可作为观念评测和行为解释的输入：

```text
need_drive_profile_i
= scarcity_i
+ threat_i
+ affiliation_i
+ status_defense_i
+ growth_block_i
```

解释：

- 匮乏、威胁、依附、防御越强，越可能改变信息解释、表达强度和主题关注
- `growth_block_i` 表示成长受阻，可影响开放性、长期导向和自我效能叙事

## 7. 心理状态对传播敏感性的调节

传播敏感性不再用于公式化观念增量，而是作为行为和评测上下文中的解释变量：

```text
susceptibility_profile_i
= threat_i
+ affiliation_i
+ status_defense_i
+ autonomy_i
```

其中：

- `susceptibility_profile_i` 表示智能体对社交线索、群体态度和可信来源的易感方式
- `autonomy_i` 表示自主性或内在稳定度

含义：

- 越焦虑、越孤独、越防御，越容易被周围人带动
- 越自主、越稳定，越不容易被短期线下意见同化

## 8. 总体链路

完整的观念形成链路如下：

```text
原子需求 satisfaction
-> 瞬时需求缺口 g
-> 持续压力 h
-> 饱和负荷项 s(h)
-> 有效需求压力 p
-> 心理中介 M
-> 角色卡与行为/认知模板
-> 信息选择、解释、表达与观念评测结果
```

相比“需求缺口直接决定观点”的简单方案，这种做法有三个优势：

- 能反映时间累积效应
- 能区分不同需求缺口的心理后果
- 能更自然地与社会传播过程耦合

## 8.1 线下到线上的心理中介层

如果研究方向转向：

```text
An Online-Offline Social Simulation Sandbox for Studying Opinion Dynamics with LLM Agents
```

则心理中介模型不应被删除，而应被重新定位为：

```text
online-offline coupling mechanism
```

也就是说，心理中介模型不是为了完整复现人类心理，而是为了把线下生活状态转译成线上传播行为和观点接受方式。它回答的核心问题是：

```text
线下经历和线下需求状态如何影响线上的信息选择、信息解释、表达行为和观点更新？
```

推荐链路如下：

```text
线下经历 / 线下需求状态
-> 内部心理中介
-> 线上信息选择、解释、表达与传播
-> 线上观点动态
```

更具体地说：

```text
offline experience
+ need pressure
+ social contact
-> psychological mediators
-> online behavior modulation
-> opinion update
```

此时心理中介层可以被命名为：

```text
Offline-to-online psychological mediation layer
线下到线上的心理中介层
```

其作用是：

```text
将线下生活经历、需求压力和社会接触转化为线上信息选择、信息解释、表达行为和观点更新的调制变量。
```

### 8.1.1 线下影响线上的五条路径

第一，信息接触路径。线下状态会影响智能体是否上网、看什么信息。

例如：

```text
疲劳高 -> 减少复杂线下行动，增加刷帖概率
资源压力高 -> 更关注资源、价格、福利相关帖子
归属压力高 -> 更关注熟人动态和群体态度
威胁感高 -> 更关注风险、冲突、煽动信息
```

对应线上变量包括：

```text
social_step probability
feed attention weight
topic preference
news exposure probability
```

第二，信息解释路径。线下状态会影响智能体如何理解同一条信息。

例如：

```text
高资源压力看到“物价上涨”
-> 更容易解释为自身生存威胁

高归属压力看到群体立场
-> 更容易解释为自己是否被群体接纳的信号

高威胁感看到争议新闻
-> 更容易解释为外部群体威胁
```

对应变量包括：

```text
appraisal_bias
perceived_relevance
perceived_threat
group_identity_relevance
```

第三，接受/抵抗路径。线下状态会影响观点更新强度。

例如：

```text
高归属压力 -> 更容易向信任群体靠拢
高自主性受挫 -> 可能更抵抗外部说服
高疲劳 -> 认知控制下降，更依赖简单线索
高威胁 -> 更依赖可信来源或内群体来源
```

第四，表达路径。线下状态会影响发帖、评论、点赞、点踩的方式。

例如：

```text
资源压力高 -> 更频繁表达不满或资源诉求
归属压力高 -> 更容易点赞群体一致观点
威胁感高 -> 更容易点踩反方观点
愤怒高 -> 评论更激烈
疲劳高 -> 发言更短、更情绪化
```

对应指标包括：

```text
post_frequency
like_rate
dislike_rate
comment_sentiment
incivility_score
stance_strength
```

第五，社交反馈回流路径。线上反馈会反过来影响后续心理状态。

例如：

```text
帖子被点赞 -> 归属感恢复，自尊提高
帖子被点踩 -> 归属压力上升，防御增强
无人回应 -> 孤独感上升
争论失败 -> 胜任感下降或愤怒上升
```

因此，线上线下耦合不是单向链路，而是闭环：

```text
线下状态 -> 心理中介 -> 线上行为 / 观点
线上反馈 -> 心理中介 -> 后续线下与线上行为
```

### 8.1.2 推荐收缩的心理中介变量

如果主线是线上线下观点动态，心理变量不宜过多。第一版建议收缩为：

```text
scarcity_pressure
fatigue_load
social_belonging_threat
threat_sensitivity
cognitive_control
```

它们与线上传播行为之间的对应关系如下：

```text
scarcity_pressure -> 资源议题相关性、福利/物价观点
fatigue_load -> 低努力加工、情绪化表达
social_belonging_threat -> 从众、群体一致性、点赞内群体
threat_sensitivity -> 风险信息敏感、外群体敌意
cognitive_control -> 是否深思熟虑、是否被煽动内容带动
```

### 8.1.3 线上线下观念表征评测

不再手工规定线上或线下观念增量公式。推荐采用评测式链路：

```text
offline state
+ online exposure
+ social interaction
+ memory
+ psychological mediators
-> behavior and expression
-> opinion assessment
-> topic-specific opinion score
```

含义：

- `M_i`：由线下经历和需求压力生成的心理中介状态
- 线上帖子、线下接触和社交反馈作为上下文证据进入评测器
- 观念变化由评测器根据上下文记录，而不是由他人观点或帖子观点与自身观点之间的差值直接计算

这种写法使心理中介自然嵌入线上线下沙盒，同时避免把 LLM 智能体的观念变化压缩成固定加权平均。

### 8.1.4 论文中的建议表述

英文表述：

```text
We introduce an offline-to-online psychological mediation layer that links offline need pressure and lived experience to online exposure, interpretation, expression, and opinion updating.
```

中文表述：

```text
本文提出线下到线上的心理中介层，将线下需求压力与生活经历连接到线上信息接触、信息解释、表达行为和观点更新过程。
```

这种定位比单独声称“模拟心理状态”更明确，因为它回答了一个具体机制问题：

```text
线下生活为什么会改变线上舆论动态？
```

## 9. 时间尺度建议

不同需求的时间累积参数应不同：

- 生理需求：快，约 `2-6` 小时出现明显恶化
- 安全需求：中等，约 `6-24` 小时
- 归属需求：较慢，约 `1-3` 天
- 尊重需求：较慢，约 `2-7` 天
- 自我实现需求：最慢，约 `7-30` 天

这样会让系统更贴近人的经验：

- 饥饿、疲劳通常较快影响判断
- 孤独、被忽视、缺乏认可更像慢性累积
- 自我实现受阻通常是长期性的心理阴影

## 10. 实现建议

如果要落地到项目中，建议按以下方式组织：

- 在需求层维护 `satisfaction -> gap -> pressure_memory -> load_saturation -> effective_pressure`
- 在心理层维护若干中介变量，而不是直接维护单一观点驱动力
- 在观念评测阶段，将需求压力、心理中介、近期经历、社交文本和记忆共同作为评测上下文
- 将不同需求类型的时间参数独立配置，避免所有需求共享同一响应速度

## 11. 心理中介到行为与认知的模板化映射

一种可行的实现方式是：为不同心理中介预设提示词模板，再根据中介变量强度动态组合这些模板，用于控制智能体的行为与认知输出。

该方案可行，但不建议采用“中介 1 对应 `s1/s2/s3` 整段模板，中介 2 对应 `a1/a2/a3` 整段模板，然后直接拼接”的粗粒度做法。直接拼接容易产生三个问题：

- 模板冲突，例如 `threat` 倾向谨慎回避，而 `affiliation` 倾向主动接近
- 组合爆炸，中介越多，模板排列越难维护
- 提示词漂移，长段人格描述会削弱控制精度

更稳妥的做法是将模板设计为“槽位化微模板”，而不是完整人格段落。

## 11.1 槽位化微模板

建议将每个心理中介拆分到多个功能槽位中：

- `attention`：优先注意什么信息
- `memory`：优先检索什么记忆
- `planning`：倾向选择何种目标和策略
- `social`：如何对待他人和社交关系
- `expression`：如何表达观点、控制语气和措辞

例如：

- `threat_vigilance`
  - `attention`：优先关注风险、损失和异常信号
  - `planning`：倾向低风险、高确定性的动作
  - `expression`：语气更谨慎、更防御
- `affiliation_drive`
  - `social`：优先维系关系、回应熟人、避免被排斥
  - `expression`：更寻求认同，更顾及群体氛围
- `status_defensiveness`
  - `memory`：更容易提取维护自我形象的记忆
  - `expression`：更容易辩护、反驳、保全面子

这样组合时，系统拼接的不是若干整段模板，而是不同槽位中的短指令。

## 11.2 按强度分桶选择模板

建议按心理中介强度分桶，而不是为每个中介固定对应一个模板：

- `0`：不激活
- `1`：低强度模板
- `2`：中强度模板
- `3`：高强度模板

例如：

```text
if threat < 0.2 -> no template
if 0.2 <= threat < 0.5 -> threat_1
if 0.5 <= threat < 0.8 -> threat_2
if threat >= 0.8 -> threat_3
```

这样可以避免模板切换过于跳变，也便于后续调参。

## 11.3 组合规则与冲突裁决

模板组合不能只做拼接，还需要显式裁决：

1. 先根据当前状态选出被激活的心理中介
2. 每个槽位只保留前 `k` 条模板，通常 `k = 1` 或 `2`
3. 低层需求相关中介通常优先级更高
4. 出现冲突时，不同时保留相反指令，而是做裁决或合成

例如，当 `threat` 与 `affiliation` 同时较高时：

- 若 `threat > affiliation + margin`，优先输出回避与防御模板
- 若 `affiliation > threat + margin`，优先输出接近与维系关系模板
- 若二者接近，则输出“谨慎接近”这类合成模板，而不是同时给出“回避”和“主动接近”

## 11.4 认知模板与行为模板分离

为了提高可控性，建议将模板分成两类，而不是混在同一段提示词中：

- `认知模板`
  - 控制注意什么信息
  - 控制如何解释事件
  - 控制优先检索哪些记忆
  - 控制更信任谁、是否接受新信息
- `行为模板`
  - 控制当前优先任务
  - 控制社交动作倾向
  - 控制发言方式
  - 控制遇到冲突时的策略

这样可以形成两步链路：

```text
心理中介 -> 认知模板组合 -> 内部判断
内部判断 + 行为模板组合 -> 动作与表达
```

该结构比将所有控制指令直接混入一个总 prompt 更清晰，也更利于调试。

## 11.5 模板化控制提示示例

如果当前状态为：

- `threat_vigilance = 0.8`
- `affiliation_drive = 0.6`
- `growth_orientation = 0.2`

则可选择如下模板：

- `threat_3.attention`：优先关注风险、损失和异常信号
- `threat_3.planning`：优先选择低风险和高确定性的动作
- `affiliation_2.social`：在安全前提下维系关系并优先回应熟人
- `growth_1.expression`：暂不主动探索陌生观点

最终组合成短控制提示：

```text
当前认知偏置：
- 优先关注风险、损失和异常信号
- 判断方案时优先低风险和高确定性
- 在确保安全的前提下维系关系并回应熟人
- 暂不主动探索陌生观点
```

该类短控制提示通常比长段人格描述更稳定，也更容易让大模型遵循。

这套模型的核心思想是：

需求的影响不是即时、线性的，而是经过“持续时间积累”与“心理中介层”之后，才进入观念变化过程。
