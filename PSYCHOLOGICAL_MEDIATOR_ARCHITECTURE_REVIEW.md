# 文献约束心理中介架构评审与改进建议

本文档整理当前模型架构的审稿人视角评估、潜在质疑点与推荐改进方向。该方案的核心是：将需求缺口与压力状态输入由心理学文献蒸馏而来的理论专家模块，由专家评估智能体心理中介状态，并生成行为认知模板；当多个专家同时激活时，通过裁判模块整合输出，得到最终动态角色卡，再进入智能体决策与规划链路。

## 1. 当前架构概述

当前模型主要包含两条链路。

### 1.1 心理中介链路

```text
需求缺口 / 压力状态
-> 文献蒸馏理论专家
-> 心理中介变量
-> 行为认知模板
-> 裁判整合
-> 最终动态角色卡
```

这一链路的目标不是直接决定智能体行为，而是将需求压力转化为可解释的心理状态、认知偏置、社交倾向和表达风格。

### 1.2 决策规划链路

```text
需求缺口 / 当前状态 / 心理因素 / 环境信息 / 记忆
-> 智能体决策与规划
-> 后续行动、对话、发帖或观念反应
```

这一链路负责根据智能体的现实需求和环境状态决定具体行动。

### 1.3 为什么需要显式心理状态层

在论文表述中，不宜直接写：

```text
LLM 本身不会产生抑郁、从众、反社会等心理表征。
```

该说法过于绝对，也很难被实证证明。更稳妥的论证方式是：

```text
现有 LLM 可以生成类似人类心理状态的语言与行为模式，但不应被视为自发具备人类式、持续演化、具身化、可验证的心理机制。因此，若研究目标是模拟需求受挫、压力累积、抑郁样状态、从众、攻击性或反社会倾向的形成过程，需要在 LLM 外部显式建模心理状态变量与更新机制。
```

这一弱版本表述更容易被文献支持。核心论据如下。

首先，LLM 的语言能力不等于具备人类式心理机制。Bender & Koller 在 ACL 2020 讨论了语言形式与意义之间的区别，指出只从语言形式训练的系统缺少获得 grounded meaning 的充分依据。这可以支持如下判断：LLM 可以说出“我很抑郁”或“我想报复”，但这不等于其内部真的经历了需求缺口、身体疲劳、社会排斥和情绪调节过程。

其次，人类情绪和心理障碍高度依赖身体状态、内感受和稳态调节。Barrett & Simmons 在 *Nature Reviews Neuroscience* 中将情绪与心理状态同 interoception、body prediction、homeostasis 联系起来。这支持本项目的建模理由：饥饿、疲劳、压力负荷这类变量并不是语言模型天然拥有的内在生理过程，因此需要外置 `need_gap`、`pressure_memory` 和 `effective_pressure` 等状态。

再次，LLM 智能体要产生可信人类行为，通常需要外置记忆、反思、规划等结构。Generative Agents 并不是让 LLM 直接裸跑，而是加入 memory stream、reflection 和 planning，并通过消融说明这些结构对 believable behavior 的重要性。该思路可以支持本项目：如果日常行为模拟都需要显式状态机制，那么心理状态演化更不应完全依赖 LLM 的即时生成。

此外，LLM 不能可靠替代真实人类参与者或群体心理。相关研究指出，用 LLM 替代人类参与者可能错误描绘并压平身份群体差异。这支持如下判断：裸 LLM 输出不是稳定的人类心理样本，也不是可靠的社会心理过程模拟器。

最后，LLM 会表现出偏见、迎合和刻板印象，但这不等于它具有人类心理机制本身。Hofmann 等在 *Nature* 中发现，语言模型会基于方言产生隐性种族偏见，并影响就业、犯罪判断等决策。这说明 LLM 可能产生类似歧视、攻击或反社会判断的输出，但其来源更可能是训练数据和对齐机制中的偏差，而不是由需求受挫、压力累积、愤怒和报复动机驱动的心理过程。

因此，更准确的总结是：

```text
LLM 可以表达抑郁、从众、攻击、偏见；
LLM 可以在 prompt 下模仿这些心理表现；
LLM 也可能因为训练偏差产生某些类似人类偏见或迎合的行为；
但 LLM 没有人类式的生理需求、社会经历、长期压力负荷和临床症状连续性；
因此，若研究目标是模拟心理现象如何产生和演化，需要外部心理状态机制。
```

论文中可以使用如下英文表述：

```text
Although LLMs can generate language that resembles human psychological states, prior work suggests that language-only models should not be assumed to possess grounded, embodied, and temporally persistent human psychological mechanisms. Therefore, we introduce an explicit theory-guided psychological state layer to model the formation and evolution of stress-related mediators from need deficits and lived experiences.
```

对应中文表述为：

```text
尽管 LLM 能生成类似人类心理状态的语言表现，但现有研究并不支持将其直接视为具备具身化、持续性和可验证的人类心理机制。因此，本文引入显式的理论约束心理状态层，用于模拟需求缺口和经历如何累积为压力，并进一步影响心理中介变量与行为倾向。
```

可引用文献包括：

- Bender, E. M., & Koller, A. (2020). Climbing towards NLU: On meaning, form, and understanding in the age of data. ACL 2020. <https://aclanthology.org/2020.acl-main.463/>
- Barrett, L. F., & Simmons, W. K. (2015). Interoceptive predictions in the brain. *Nature Reviews Neuroscience*. <https://www.nature.com/articles/nrn3950>
- Park, J. S., et al. (2023). Generative Agents: Interactive Simulacra of Human Behavior. UIST 2023. <https://arxiv.org/abs/2304.03442>
- Wang, A., Morgenstern, J., & Dickerson, J. P. (2024). Large language models that replace human participants can harmfully misportray and flatten identity groups. <https://arxiv.org/abs/2402.01908>
- Hofmann, V., et al. (2024). AI generates covertly racist decisions about people based on their dialect. *Nature*. <https://www.nature.com/articles/s41586-024-07856-5>

#### Barrett & Simmons (2015) 对当前模型的具体支持

Barrett & Simmons (2015) 发表在 *Nature Reviews Neuroscience* 上，属于认知神经科学与情感神经科学领域的权威理论文献。该文提出的核心思想是：心理状态并不是纯语言解释或纯外部事件反应，而是与身体内部状态感知、身体状态预测和稳态调节密切相关。

因此，该文可以支持当前模型中的如下链路：

```text
satisfaction
-> need_gap
-> pressure_memory
-> effective_pressure
-> 心理中介变量
-> 行为 / 认知倾向
```

其中，关键概念可以映射为：

```text
interoception
= 智能体读取自身需求状态，例如 satisfaction、need_gap、effective_pressure

body prediction
= 智能体根据当前缺口、历史压力和近期趋势预测需求是否会继续恶化或恢复

homeostasis
= 智能体通过吃饭、睡觉、赚钱等行为尝试恢复内部平衡
```

据此，可以形成本文的建模论点：

```text
LLM 本身没有真实身体内部状态，也没有天然的稳态调节系统。
如果要模拟饥饿、疲劳、压力负荷如何影响心理和行为，
就需要外置 need_gap、pressure_memory、effective_pressure 这样的状态层。
```

需要注意的是，Barrett & Simmons (2015) 并不能直接证明本文的具体公式唯一正确。它支持的是更上层的建模前提：人类心理状态与身体内部状态、身体状态预测和稳态调节相关，因此在 LLM 智能体中显式加入需求状态与压力状态具有理论合理性。具体的 `need_gap -> pressure_memory -> effective_pressure` 仍属于本文基于该理论前提做出的工程化建模选择，需要通过消融实验、敏感性分析和行为预测实验来验证。

## 2. 审稿人可能认可的部分

如果从审稿人角度看，该方案具有明确研究价值。

首先，它不是让 LLM 直接从需求状态自由生成行为，而是在中间加入文献约束的心理中介层。这提高了模型的可解释性。

其次，每类需求缺口对应一个理论专家，专家由多篇心理学文献蒸馏得到。这使模型能够把心理学理论转化为可复用、可检索、可消融的机制模块。

再次，当多个需求缺口同时激活时，裁判模块负责整合专家输出。这使系统可以处理现实中常见的复合心理状态，例如归属受挫叠加疲劳、安全威胁叠加资源匮乏。

较合适的论文表述是：

```text
我们并不声称精确复现人类心理机制，而是构建一个文献约束的心理中介层，用于将需求压力转化为可解释、可记录、可消融的认知与行为偏置。
```

该架构可以被描述为：

```text
literature-grounded psychological mediator module
文献约束心理中介模块
```

或：

```text
literature-grounded mixture of psychological experts
文献约束的心理专家混合模块
```

## 3. 审稿人可能质疑的地方

### 3.1 专家模块定义不够明确

审稿人会首先追问：

```text
专家到底是什么？
```

需要明确说明专家是以下哪种形式，或由哪些部分组成：

- 人工规则；
- LLM prompt；
- 理论卡；
- 固定 JSON 模板；
- 检索式文献摘要；
- LLM 评估器。

建议不要在论文中直接使用过于宽泛的“专家系统”一词，因为它容易让人联想到传统规则系统。更稳妥的说法是：

```text
文献约束心理中介模块
```

每个专家应至少包含：

- 对应需求缺口；
- 文献来源；
- 触发条件；
- 心理中介变量；
- 条件规则；
- 行为认知模板槽位；
- 证据强度；
- 适用边界。

### 3.2 两条链路可能重复计入需求影响

当前架构中，需求缺口同时进入心理中介链路和决策规划链路。如果处理不当，可能出现重复放大。

风险形式如下：

```text
需求缺口 -> 心理模板 -> 决策
需求缺口 -> 决策
```

这样可能导致同一个需求对行为施加两次影响。

建议明确区分两条链路的职责：

```text
原始需求：告诉智能体要解决什么问题。
心理模板：告诉智能体以什么认知偏置、情绪状态和表达风格去解决问题。
```

例如：

```text
饥饿决定智能体需要寻找食物。
疲劳影响智能体在寻找食物时更急躁、更短视、更少进行复杂规划。
```

这样可以避免把需求和心理状态混成同一个驱动力。

### 3.3 阈值唤醒可能造成心理状态跳变

如果专家只在压力超过阈值时才被唤醒，模型可能产生突兀变化。

例如：

```text
pressure = 0.49 -> 无专家激活
pressure = 0.51 -> 专家突然完全激活
```

这不符合心理状态通常渐进变化的特点。

建议使用软激活机制：

```text
activation = sigmoid(k * (pressure - threshold))
```

如果暂时不使用显式数学公式，也应在裁判提示词中加入规则：

```text
除非出现强事件，否则心理状态只能渐进变化，不能在相邻时间步中剧烈反转。
```

还可以加入：

- 上一轮心理状态惯性；
- 激活滞后机制；
- 恢复期衰减机制；
- 强事件快速激活机制。

### 3.4 多专家融合必须可解释

多个专家同时激活时，不能简单让 LLM 自由总结。审稿人会质疑裁判输出是否稳定、是否可复现。

裁判模块需要有明确融合原则：

- 低层需求优先于高层需求；
- 安全威胁优先于成长探索；
- 短期强事件优先影响情绪；
- 长期持续缺口优先影响稳定倾向；
- 同一槽位只保留一到两条主导指令；
- 冲突时生成合成策略，而不是并列矛盾指令；
- 裁判必须输出每个保留指令的来源专家和原因。

例如，不应输出：

```text
主动接近他人，同时尽量远离他人。
```

应整合为：

```text
在安全可控的熟人关系中尝试接近，对陌生人和高风险互动保持防御。
```

### 3.5 有效性验证需要提前设计

该方案最容易被质疑为手工拼接。要回应这一点，必须提供消融实验和预测验证。

建议至少设置以下对照：

```text
完整模型
vs 无心理专家模块
vs 只使用原始需求状态
vs 多专家输出直接拼接，无裁判融合
vs 随机理论卡或错误理论卡
vs 固定心理模板，不随状态变化
```

如果完整模型能够更好地复现心理学文献中的方向性现象，例如：

- 归属受挫提高社会监测；
- 疲劳提高情绪化表达；
- 资源匮乏提高短视决策；
- 社会支持缓冲压力效应；
- 自主受挫在有行动能力时提高反抗，在长期无力时提高无助；

则可以证明心理中介层确实带来了有效机制。

## 4. 推荐的最终架构

建议将系统明确收束为五层。

### 4.1 Need-State Layer

记录智能体的基础需求状态。

```text
satiety
relax
safety
belonging
autonomy
competence
esteem
growth
```

该层回答：

```text
智能体现在缺什么？
```

### 4.2 Pressure Layer

根据需求状态、持续时间和近期事件生成压力描述。

输入包括：

- 当前缺口；
- 长期缺口；
- 缺口持续时间；
- 恢复趋势；
- 近期事件；
- 上一轮心理状态。

该层回答：

```text
这个缺口对智能体造成了多强、多久、何种类型的压力？
```

### 4.3 Theory-Expert Layer

每类需求缺口对应一个文献蒸馏理论专家。

例如：

```text
physiological_deficit_expert
safety_deficit_expert
belonging_deficit_expert
autonomy_deficit_expert
competence_deficit_expert
esteem_status_deficit_expert
growth_meaning_deficit_expert
```

每个专家输出：

```json
{
  "source_expert": "belonging_deficit_expert",
  "activation": 0.82,
  "mediator_delta": {
    "social_pain": 0.75,
    "social_monitoring": 0.68,
    "conformity_susceptibility": 0.52,
    "rejection_aggression": 0.22
  },
  "slot_effects": {
    "attention": ["优先关注他人是否接纳自己"],
    "memory": ["更容易回忆近期被冷落经历"],
    "appraisal": ["倾向把模糊社交信号解释为可能的排斥"],
    "social": ["若仍有接纳机会，倾向迎合群体"],
    "expression": ["语气更敏感，更在意认可"]
  },
  "reason": "归属缺口较高且近期存在被忽视事件。"
}
```

### 4.4 Arbiter Layer

裁判模块融合多个专家输出。

裁判输出：

```json
{
  "dominant_state": "归属受挫叠加疲劳导致的敏感防御状态",
  "mediators": {
    "social_pain": 0.75,
    "social_monitoring": 0.68,
    "fatigue_control_loss": 0.64,
    "attention_instability": 0.58,
    "conformity_susceptibility": 0.52,
    "negative_affect": 0.48,
    "rejection_aggression": 0.22
  },
  "role_card": {
    "attention": "优先关注他人是否接纳自己，同时更容易被强情绪和社交反馈吸引。",
    "memory": "更容易想起近期被忽视、被冷落或互动失败的经历。",
    "appraisal": "倾向把模糊社交信号解释为可能的排斥，但疲劳会降低细致判断能力。",
    "planning": "优先选择低成本、能快速缓解孤立感的行动，避免复杂长期计划。",
    "social": "若仍有接纳机会，倾向迎合熟人或群体主流；若再次被拒绝，可能转向防御。",
    "expression": "语气较敏感、急躁，容易寻求认可，但攻击性尚不占主导。"
  },
  "fusion_reason": "归属专家激活强度最高，疲劳专家对注意与表达槽位产生调节，二者无直接冲突。"
}
```

### 4.5 Decision Layer

最终行动决策 LLM 接收：

- 当前任务；
- 环境状态；
- 原始需求；
- 记忆；
- 社交上下文；
- 心理中介状态；
- 动态角色卡。

该层负责生成：

- 线下行动；
- 对话；
- 发帖；
- 点赞/点踩；
- 是否接受信息；
- 是否更新观念。

## 5. 关键设计原则

### 5.1 理论专家不直接决定行为

理论专家只输出心理中介和行为认知偏置，不直接输出最终动作。

推荐关系：

```text
专家输出心理偏置
裁判整合心理偏置
决策层结合环境与目标生成动作
```

避免：

```text
归属专家直接命令智能体发帖求认同
```

### 5.2 原始需求与心理中介职责分离

```text
需求层决定目标压力。
心理层决定认知、情绪、表达和社交策略。
行为层决定具体行动。
```

### 5.3 理论卡按需求缺口分类

理论卡不必按单篇文章分类，而应按需求缺口分类。一张理论卡可以蒸馏多篇文献。

例如：

```text
归属感缺口理论卡
= need to belong
+ ostracism / need-threat model
+ social pain
+ social monitoring
+ aggression after exclusion
+ interpersonal reconnection
```

但卡片内部必须保留文献来源、证据强度和适用边界。

### 5.4 多专家输出先结构化，再融合

不要直接拼接多个专家的自然语言模板。

推荐流程：

```text
专家输出 JSON
-> 裁判融合 JSON
-> 生成最终角色卡
-> 注入决策 LLM
```

### 5.5 最终角色卡必须短而可执行

最终角色卡不应包含大量文献解释，只保留对行为生成有用的槽位。

推荐槽位：

```text
attention
memory
appraisal
planning
social
expression
```

每个槽位保留一到两条主导指令即可。

## 6. 论文中的建议表述

可以在方法部分写：

```text
We introduce a literature-grounded psychological mediator layer between needs and agent decisions. Instead of allowing the LLM agent to directly infer behavior from raw need deficits, the mediator layer converts need pressure into structured psychological mediators and cognition-behavior templates. Each need domain is associated with a theory expert distilled from multiple psychological studies. Experts are softly activated by current and accumulated need pressure. When multiple experts are activated, an arbiter integrates their outputs according to predefined conflict-resolution principles and produces a unified dynamic role card for the decision-making agent.
```

中文对应表述：

```text
本文在需求状态与智能体决策之间引入文献约束的心理中介层。该中介层并不直接决定行为，而是将当前与长期需求压力转化为结构化心理中介变量和行为认知模板。每类需求缺口对应一个由多篇心理学文献蒸馏而来的理论专家。专家根据当前压力与长期压力被软激活。当多个专家同时激活时，裁判模块依据预设的冲突裁决原则整合专家输出，生成统一的动态角色卡，并交由决策智能体用于行动、对话和线上表达。
```

## 7. 必须补充的验证实验

为了回应审稿人对有效性的质疑，建议至少加入以下实验。

### 7.1 消融实验

```text
完整模型
无心理专家模块
只使用原始需求
无裁判直接拼接
随机理论卡
固定心理模板
```

### 7.2 文献预测复现实验

检验模型是否能复现心理学文献中的方向性结果：

- 归属受挫后 `social_monitoring` 上升；
- 排斥后在接纳预期高时 `conformity_susceptibility` 上升；
- 排斥后在责怪他人且接纳预期低时 `rejection_aggression` 上升；
- 疲劳后 `attention_instability` 与情绪化表达上升；
- 资源匮乏后 `scarcity_tunnel` 和短期资源获取倾向上升；
- 社会支持高时压力效应被缓冲。

### 7.3 稳定性实验

检验同一输入多次运行时，专家输出和裁判输出是否保持一致。

可以报告：

- JSON 字段一致率；
- 心理中介数值方差；
- 角色卡槽位语义相似度；
- 最终行为选择一致率。

### 7.4 参数与阈值敏感性分析

检验专家激活阈值、软激活强度、裁判优先级规则变化后，核心结论是否稳定。

## 8. 总结判断

该架构可以成立，但需要避免把它描述成一个模糊的“专家系统”。更合适的定位是：

```text
可检索、可消融、可验证的文献约束心理中介架构。
```

如果最终论文能够清楚定义专家、理论卡、裁判、心理中介变量和动态角色卡之间的数据接口，并通过消融实验与文献预测验证证明心理中介层确实改善行为合理性，那么该方案具有较好的方法论说服力。
