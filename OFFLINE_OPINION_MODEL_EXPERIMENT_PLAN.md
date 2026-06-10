# 需求压力、心理中介与线下观念变化模型实验方案

本文档用于验证完整模型链路，而不是单独验证某一个需求维度。

完整链路如下：

```text
satisfaction
-> gap
-> pressure_memory
-> load_saturation
-> effective_pressure
-> psychological mediators
-> role-card prompt modulation
-> behavior / social interaction
-> offline opinion update
```

## 1. 总体实验问题

需要验证四件事：

1. 新模型是否比当前 `offline_update()` 更能解释 `opinion` 变化。
2. 心理中介层是否真的改变行为、社交表达和传播易感性。
3. 压力记忆 `pressure_memory` 是否带来时间累积效应。
4. 角色卡提示词是否让 LLM 行为和心理变量保持一致。

## 2. 实验 1：消融实验

目的：验证每个模块是否有必要。

设置 5 个模型版本：

```text
M0：当前项目模型
    只有 offline_trust 邻居同化 + urgency 同化

M1：加入 gap
    satisfaction -> gap -> opinion

M2：加入 pressure_memory
    satisfaction -> gap -> pressure_memory -> effective_pressure -> opinion

M3：加入 psychological mediators
    satisfaction -> gap -> pressure_memory -> effective_pressure -> mediators -> Psi / Chi -> opinion

M4：加入 role-card prompt modulation
    M3 + 心理中介强度角色卡影响行为、发帖、对话、记忆检索
```

核心比较：

```text
M0 vs M1：瞬时需求缺口是否有解释力
M1 vs M2：时间累积是否有解释力
M2 vs M3：心理中介层是否有解释力
M3 vs M4：角色卡是否改变可观察行为
```

记录指标：

```text
opinion_change
behavior_distribution
social_action_distribution
post_sentiment / post_stance
mediator_values
offline_social_influence
Psi_i
Chi_i
```

判据：

```text
M4 应该表现出最强的行为可解释性。
M3 应该比 M0 更能解释 opinion 变化来源。
M2 应该比 M1 更能表现持续压力和恢复滞后。
```

## 3. 实验 2：因果链路验证实验

目的：确认变量变化顺序符合设计，而不是同时乱跳。

设计一段固定事件序列：

```text
阶段 A：资源充足，社交环境稳定
阶段 B：部分需求长期得不到满足
阶段 C：投放带有明确 opinion_index 的系统新闻
阶段 D：恢复需求供给
阶段 E：继续观察 opinion 和心理状态是否缓慢回落
```

每个阶段运行固定 tick，例如：

```text
A：20 tick
B：40 tick
C：10 tick
D：20 tick
E：40 tick
```

需要验证的时间顺序：

```text
satisfaction 先下降
gap 随后升高
pressure_memory 持续累积
load_saturation 接近饱和
effective_pressure 上升
psychological mediators 上升
role-card 激活等级变化
behavior / social expression 改变
opinion 变化
恢复后 gap 先下降
pressure_memory 和 mediators 后下降
```

判据：

```text
pressure_memory 不能和 gap 同步瞬间归零。
角色卡等级变化必须滞后于持续压力，而不是只由单 tick satisfaction 决定。
opinion 变化必须能被 Psi_i 和 social_influence 分解解释。
```

## 4. 实验 3：同一社交网络，不同内部状态

目的：验证新模型不是简单社交同化。

设置两组智能体，给它们完全相同的：

```text
offline_trust 网络
初始 opinion
新闻输入
邻居 opinion
地图
角色设定
```

只改变内部状态轨迹：

```text
A组：长期低满足度，高 pressure_memory
B组：满足度稳定，低 pressure_memory
```

观察：

```text
Psi_i
Chi_i
Delta_offline
opinion convergence speed
social_action
post_content
```

判据：

```text
如果社交网络完全相同，但 A组 opinion 变化更快或方向不同，
说明内部心理状态进入了观念变化过程。
```

该实验验证：

```text
internal state -> psychological mediators -> Psi / Chi -> opinion
```

## 5. 实验 4：同一内部状态，不同社交网络

目的：验证线下社交传播项仍然有效。

设置两组智能体，给它们完全相同的：

```text
satisfaction
pressure_memory
psychological mediators
initial opinion
role-card
news exposure
```

只改变：

```text
A组：高 offline_trust 邻居，邻居 opinion 与自己差距大
B组：低 offline_trust 邻居，或邻居 opinion 与自己接近
```

观察：

```text
social_influence_i
Chi_i * social_influence_i
Delta_offline
opinion trajectory
```

判据：

```text
A组 opinion 变化幅度应大于 B组。
```

该实验验证：

```text
offline social influence 仍然是独立作用项
```

不是所有变化都由需求压力决定。

## 6. 实验 5：角色卡有效性实验

目的：验证心理中介变量是否真的改变 LLM 输出。

同一批 agent 使用相同状态输入，分别运行：

```text
A组：不插入心理角色卡
B组：插入固定角色卡
C组：插入随 mediator 强度变化的动态角色卡
```

比较：

```text
tool choice
action arguments
post frequency
comment / like / dislike 比例
发帖内容主题
语言风格
是否遵守当前心理表征
```

需要设计文本打分器或规则指标，例如：

```text
resource_terms_count
risk_terms_count
affiliation_terms_count
defensive_terms_count
fatigue_terms_count
```

判据：

```text
动态角色卡组 C 的行为和语言，应与 mediator 强度最一致。
固定角色卡组 B 容易过度稳定，不能反映状态变化。
无角色卡组 A 与 mediator 的相关性较弱。
```

## 7. 实验 6：压力恢复滞后实验

目的：验证模型不是“需求恢复后心理状态立刻正常”。

设计：

```text
阶段 A：长期压力
阶段 B：需求快速恢复
阶段 C：继续观察
```

对比两个模型：

```text
NoMemory：只用 gap
Memory：使用 pressure_memory + load_saturation
```

观察：

```text
gap
pressure_memory
effective_pressure
mediators
opinion
behavior
```

判据：

```text
NoMemory 在需求恢复后心理表征立即下降。
Memory 在需求恢复后仍保留残余心理影响，并逐步衰减。
```

该实验验证文档中的核心假设：

```text
长期负荷不等于当前缺口，但会调节当前反应。
```

## 8. 实验 7：外部冲击与观念传播实验

目的：验证新闻、社交传播、内部心理状态三者耦合。

设置系统新闻：

```text
news_1：opinion_index = 0.2
news_2：opinion_index = 0.8
```

设置 agent 内部状态：

```text
低压力组
高 scarcity 组
高 threat 组
高 affiliation 组
高 status_defense 组
```

观察：

```text
online_update 后 opinion
offline_update 后 opinion
转发/评论/点赞/点踩倾向
对新闻内容的表达方式
不同群体之间 opinion 分化或收敛
```

判据：

```text
高 affiliation 组更容易随高 trust 群体移动。
高 threat 组更谨慎，可能更依赖可信来源。
高 status_defense 组更容易反驳不一致观点。
高 scarcity 组更关注新闻中的资源、价格、机会、损失。
```

## 9. 实验 8：可解释性实验

目的：证明新模型不是只增加复杂度，而是更可解释。

每次 `offline_update` 记录分解项：

```text
Delta_offline
need_drive_delta = eta * Psi_i
social_delta = beta * Chi_i * social_influence_i
Psi_i
Chi_i
social_influence_i
active_mediators
active_role_cards
```

人工抽样检查：

```text
agent 在本 tick 为什么变得更保守/更激进？
是需求压力驱动？
是社交邻居驱动？
是两者叠加？
角色卡是否解释了行为？
```

判据：

```text
至少大部分 opinion 变化能用日志中的分解项解释。
```

这里不要追求主观“合理”，而是看日志是否能追踪因果链。

## 10. 实验 9：稳定性与鲁棒性实验

目的：避免模型过度敏感或数值爆炸。

扫参数：

```text
rho_m
kappa_m
alpha_m
gamma_m
delta_m
eta
beta
v1-v4
u1-u5
```

观察：

```text
opinion 是否快速贴边到 0 或 1
pressure_memory 是否无限增长
role-card 是否频繁跳变
agent 是否陷入单一行为循环
```

判据：

```text
opinion 长期不应全部贴边。
pressure_memory 经 load_saturation 后应有界。
角色卡等级不应每 tick 来回跳。
行为分布不应退化成只 social_step 或只 move。
```

## 11. 实验 10：端到端场景实验

目的：最终展示完整方案。

设计一个小型社会场景：

```text
5-10 个 agent
2 个社交团体
不同 offline_trust 网络
不同初始 opinion
不同资源位置
周期性新闻输入
阶段性资源短缺或休息受限
```

运行：

```text
100-300 tick
```

输出：

```text
opinion 曲线
mediator 曲线
行为比例曲线
社交互动网络
发帖主题变化
active_role_cards 时间线
```

最终判断：

```text
是否出现可解释的群体分化、收敛、压力恢复、从众、反驳、资源关注等现象。
```

## 12. 最小可行实验版本

如果要先快速落地，不要一开始做 10 个实验。先做这 3 个：

```text
1. 消融实验
2. 因果链路验证实验
3. 同一社交网络，不同内部状态实验
```

它们能覆盖完整方案的关键问题：

```text
模块是否必要
链路顺序是否正确
内部心理状态是否真的影响观念变化
```

## 13. 必须新增的日志字段

没有这些字段，实验不可解释：

```text
tick
agent_id
model_version

satisfaction.satiety
satisfaction.relax
satisfaction.money

gap.*
pressure_memory.*
load_saturation.*
effective_pressure.*

scarcity_i
threat_i
affiliation_i
status_defense_i
growth_block_i
fatigue_i
cognitive_control_loss
emotional_reactivity

Psi_i
Chi_i
social_influence_i
need_drive_delta
social_delta
Delta_offline

opinion_before
opinion_after

active_role_cards
action_tool
post_id
post_content
comment_content
like_target
dislike_target
```

## 14. 核心结论

整套实验的重点不是证明“模型看起来合理”，而是验证：

```text
需求压力是否有时间累积
心理中介是否改变行为和表达
心理状态是否调节线下传播
opinion 变化是否能被分解解释
动态角色卡是否比固定 prompt 更有效
```

这才是完整方案的实验闭环。
