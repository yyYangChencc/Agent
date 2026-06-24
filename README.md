# Agent Simulation

基于大语言模型（LLM）的多智能体网格仿真平台。项目将物理世界、需求状态、向量记忆、智能体对话、社交平台和意见传播组合在同一套 tick 循环中，并通过 React + Pixi.js 页面实时展示仿真状态。

## 当前功能

- **LLM 驱动决策**：智能体根据局部观测、历史记录、向量记忆、任务和需求状态生成行动。
- **异步并行执行**：每个 tick 内并发执行各智能体的决策与反思流程。
- **双层需求状态**：`satisfaction` 表示客观满足度，`urgency` 表示主观急迫度；当前需求键为 `satiety`、`relax`、`money`，`urgency` 由满足度缺口和层级封顶规则重新计算。
- **需求压力层**：在 `satisfaction` 之上维护 `need_gap`、`pressure_memory`、`load_saturation` 和 `effective_pressure`，用于表达瞬时需求缺口、累积压力记忆、压力负荷饱和度和当前有效压力；睡眠期间只分步恢复 `relax`，不执行普通需求衰减，也不累积新的压力记忆。
- **心理评测接口**：每隔 `psychological_assessment_interval` 个 tick 汇总一个评测窗口；当窗口内某项 `effective_pressure` 超过阈值时激活对应需求评测器。默认使用 LLM 结合理论卡、窗口经历、需求变化、有效压力和上一轮结果生成心理中介与动态角色卡；LLM 不可用或输出异常时回退理论卡规则评测，多个评测器同时激活时进入现有裁判整合。
- **网格世界**：固定使用 25×25 地图，支持移动、进食、睡眠、进入建筑、建筑内自动交互、离开建筑和近距离对话。
- **场景设施**：Web 服务默认创建 5 个智能体、5 张床、2 家公司、2 家食品店、1 个游乐场和 5 处散落食物。
- **社交平台**：智能体可以发帖、评论、点赞和点踩，并接收关注对象的帖子及系统新闻。
- **观念评测**：不再使用线上帖子均值或线下邻居均值的公式化观念更新；`opinion` 表示智能体对系统投放新闻主题的当前立场。每个 tick 末由观念评测模块只评估该系统新闻主题，并将评测得到的 score 写回 `agent.opinion`。默认使用 LLM 评测，失败时回退规则评测。
- **定时新闻**：`backend/persona/news_events.py` 在 tick 5、15、25、35 投放预设新闻。
- **向量记忆**：使用 ChromaDB 保存多智能体长期记忆，并通过嵌入接口执行语义检索。
- **运行记录**：后端将日志写入 `logs/`，Web 服务将每个智能体的 tick 历史写入 `backend/history/`。
- **Web 可视化**：显示地图、智能体、设施、帖子、需求状态、最近思考和最多 200 个前端历史点的趋势图。

## 技术栈

**后端**：Python、FastAPI、WebSocket、ChromaDB、OpenAI Python SDK

**前端**：React 18、TypeScript、Pixi.js 8、Zustand、Recharts、Tailwind CSS、Vite

## 启动流程

以下命令默认在项目根目录 `Agent/` 下执行。开发模式需要两个终端：一个运行后端 FastAPI 服务，另一个运行前端 Vite 开发服务器。

### 运行环境

- Python 3.10+
- Node.js 18+
- 与 OpenAI Python SDK 接口兼容的对话和嵌入服务

### 1. 安装后端依赖

推荐先创建虚拟环境，再安装 Python 依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

如果已经在可用的 Python 环境中，也可以只执行：

```powershell
pip install -r requirements.txt
```

### 2. 配置环境变量

项目根目录的 `.env` 文件会被 `backend/main.py` 和 `backend/server.py` 加载。当前代码读取以下变量：

```dotenv
OPENAI_API_KEY=<对话接口密钥>
OPENAI_BASE_URL=<对话接口地址>
EMBEDDING_KEY=<嵌入接口密钥>
EMBEDDING_BASE_URL=<嵌入接口地址>
```

`OPENAI_BASE_URL` 和 `EMBEDDING_BASE_URL` 未设置时，代码使用 `https://api.openai.com/v1`。

后端启动时会初始化默认场景并创建 ChromaDB 记忆管理器；如果缺少可用的 `OPENAI_API_KEY` 或 `EMBEDDING_KEY`，涉及 LLM 或嵌入检索的仿真流程无法正常完成。

### 3. 安装前端依赖

```powershell
cd frontend
npm install
cd ..
```

### 4. 开发模式启动

终端 1，在项目根目录启动 FastAPI 服务：

```powershell
python backend/server.py
```

后端监听 `http://localhost:8000`，WebSocket 路径为 `ws://localhost:8000/ws`。

终端 2，启动 Vite 开发服务器：

```powershell
cd frontend
npm run dev
```

浏览器访问 `http://localhost:5173`。`frontend/vite.config.ts` 会将开发环境下的 `/ws` 代理到 `ws://localhost:8000`。

前端页面连接成功后，可以使用顶部控制栏执行：

- **单步**：只推进 1 个 tick。
- **继续**：按当前速度自动推进 tick。
- **暂停**：停止自动推进。
- **重置**：后端重建 runtime，tick 归零，并重新初始化智能体、地图对象和记忆。
- **速度**：切换 1×、2×、5× 自动推进速度。

### 5. 构建后启动

先构建前端：

```powershell
cd frontend
npm run build
cd ..
```

再由 FastAPI 托管 `frontend/dist`：

```powershell
python backend/server.py
```

浏览器访问 `http://localhost:8000`。此模式不需要再运行 `npm run dev`。

## 其他运行入口

### 命令行仿真

以下命令创建 5 个智能体并运行 10 个 tick，在终端打印各智能体的意见值：

```powershell
python backend/main.py
```

该入口不启动 Web 页面，主要用于快速检查后端 tick 循环和意见传播输出。

### 空间交互演示

`backend/test_spatial.py` 是不调用 LLM 的空间交互演示服务，用于按固定步骤展示睡眠、进入建筑、建筑内自动交互、离开建筑和建筑内移动。

终端 1：

```powershell
python backend/test_spatial.py
```

终端 2：

```powershell
cd frontend
npm run dev
```

浏览器访问 `http://localhost:5173`。运行前需要停止 `backend/server.py`，因为 `backend/server.py` 和 `backend/test_spatial.py` 都监听 8000 端口。

### 端口与常见启动顺序

- 开发模式：先启动 `python backend/server.py`，再启动 `cd frontend && npm run dev`，访问 `http://localhost:5173`。
- 构建后运行：先执行 `cd frontend && npm run build`，再启动 `python backend/server.py`，访问 `http://localhost:8000`。
- 空间交互演示：先确保 `backend/server.py` 已停止，再启动 `python backend/test_spatial.py` 和 `cd frontend && npm run dev`，访问 `http://localhost:5173`。

## Web 界面

| 区域 | 当前功能 |
|------|----------|
| 顶部控制栏 | 显示 tick 与连接状态；支持单步、继续、暂停、重置和 1×/2×/5× 速度 |
| 世界画布 | 渲染 25×25 网格、智能体、食物、床、公司、食品店和游乐场 |
| 智能体面板 | 显示角色、情绪、任务、满足度、急迫度、金钱、意见、睡眠/建筑状态和最近思考 |
| 趋势图 | 显示 `satisfaction`、`urgency`、`opinion` 和情绪历史 |
| 物品详情 | 显示物品类型、后端描述、位置、数量和占用者 |
| 社交面板 | 按帖子 ID 倒序展示内容、发布时间、意见值、点赞、点踩和评论 |
| 选中高亮 | 智能体显示黄色描边和观测半径；物品显示黄色描边 |

进入建筑或睡眠中的智能体不会在画布上单独显示，建筑右上角会显示占用数量。

## 每个 tick 的执行流程

1. 世界时间加 1，刷新本 tick 的移动记录，并取得全部智能体。
2. 对已经进入建筑且未睡眠的智能体执行建筑内自动交互。
3. 并发执行每个智能体的行动阶段：本 tick 开始时已经睡眠的智能体只推进睡眠剩余时间并分步恢复 `relax`，未睡眠的智能体执行观测、记忆检索、LLM 决策和动作。
4. 并发执行普通反思与需求状态更新；本 tick 开始时已经睡眠的智能体即使在本 tick 醒来也不会立刻执行普通反思和普通需求衰减。需求变化后同步刷新 `urgency`、`need_gap`、`pressure_memory`、`load_saturation` 和 `effective_pressure`。
5. 执行智能体对话阶段。
6. 按 `NEWS_SCHEDULE` 投放当前 tick 的系统新闻。
7. 执行心理评测阶段：先写入当前评测窗口，再按压力阈值决定是否激活需求评测器或裁判整合接口。
8. 执行观念评测阶段：只围绕 `default_opinion_topic` 指定的系统新闻主题，根据最近经历、最近发帖/评论、记忆、动态角色卡和当前 `opinion` 评测新 score，并写回 `agent.opinion`。
9. 将智能体状态写入 CSV 历史文件。

## 意见传播

意见值范围为 `[-1, 1]`。当前默认主题为 `姜萍事件`：`-1` 表示激烈反对姜萍并可能上升到反对媒体造神、竞赛公信力或教育叙事层面，`0` 表示不关心、没听说过或暂不表态，`1` 表示完全赞成或高度支持姜萍。

- **观念评测**：每个 tick 末只评估系统新闻主题 `default_opinion_topic`；评测输入包括近期经历、近期社交文本、记忆、完整动态角色卡和当前 `opinion`。评测 score 会写回 `agent.opinion`，同时镜像记录到 `opinion_scores[topic]` 和 `last_opinion_assessment` 供展示与分析使用。
- **不再使用公式化更新**：系统不再按线上帖子均值或线下邻居均值直接计算观念变化。
- **信任变化**：点赞他人帖子会将对应 `online_trust` 增加 `0.05`，点踩会减少 `0.05`，结果限制在 `[0, 1]`。
- **系统新闻**：新闻使用 `backend/persona/news_events.py` 中明确设置的 `opinion_index`；当前 `NEWS_SCHEDULE` 是姜萍事件的 100 tick 仿真用时间线。

当前 `backend/persona/opinion/scorer.py` 的 `evaluate_opinion()` 固定返回 `0.0`，因此智能体通过 `send_post` 新发帖时，`opinion_index` 目前统一为中立值 `0.0`。基于自然语言内容计算立场值的逻辑尚未实现。

## 运行产物

- `logs/agent_<timestamp>.log`：完整 DEBUG 日志；控制台显示 INFO 及以上日志。
- `backend/history/<timestamp>/<agent_id>.csv`：Web 服务记录的智能体历史，字段包括 `tick`、`opinion`、`satiety`、`relax`、`money`、`satiety_urgency`、`relax_urgency`、`satiety_gap`、`relax_gap`、`money_gap`、`satiety_pressure_memory`、`relax_pressure_memory`、`money_pressure_memory`、`satiety_effective_pressure`、`relax_effective_pressure`、`money_effective_pressure`、`emotion` 和 `task`。
- `chroma_agents/`：ChromaDB 持久化数据。


## 项目结构

```text
Agent/
├── backend/
│   ├── main.py                         # 命令行仿真入口
│   ├── server.py                       # FastAPI、WebSocket、默认场景与仿真控制
│   ├── test_spatial.py                 # 无 LLM 的空间交互演示服务
│   ├── persona/
│   │   ├── config.py                   # AgentConfig
│   │   ├── runtime.py                  # SimulationRuntime 组合根
│   │   ├── history_recorder.py         # CSV 历史记录
│   │   ├── news_events.py              # 定时新闻
│   │   ├── agents/                     # Agent、策略、提示词与动作解析
│   │   ├── agent_memory/               # ChromaDB 多智能体记忆
│   │   ├── llm/                        # OpenAI 同步/异步客户端
│   │   ├── opinion/                    # 在线/离线更新与文本意见评分
│   │   ├── psychology/                 # 心理评测窗口、需求评测器接口与裁判整合接口
│   │   └── reflect/                    # 任务反思与 micro-reflection
│   ├── social_sys/                     # 社交平台、帖子与评论
│   ├── tools/operator_tools.py         # 世界动作与社交动作
│   └── world/                          # 地图、对象、观测、事件、序列化与 tick 循环
├── frontend/
│   ├── src/
│   │   ├── components/                 # 控制栏、地图、详情、社交面板与趋势图
│   │   ├── hooks/useWebSocket.ts       # WebSocket 连接与自动重连
│   │   └── store/simStore.ts           # Zustand 状态与前端历史
│   ├── package.json
│   └── vite.config.ts
├── .gitignore
├── README.md
└── requirements.txt
```

## 核心配置

配置定义在 `backend/persona/config.py` 的 `AgentConfig` 中。

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `llm_model` | `gpt-5.5` | 对话模型名 |
| `embedding_model` | `text-embedding-3-small` | 嵌入模型名 |
| `observation_radius` | `5` | 观测半径，也是单次移动的最大步数 |
| `eat_distance_sq` | `2.0` | 进食、睡眠和进入建筑的平方距离限制 |
| `max_history` | `12` | 智能体短期历史条数 |
| `memory_top_k` | `5` | 向量记忆检索条数 |
| `satiety_threshold` | `30.0` | 饱腹任务满足阈值 |
| `relax_threshold` | `30.0` | 放松任务满足阈值 |
| `money_threshold` | `15.0` | 金钱/安全需求满足阈值 |
| `satiety_decay_rate` | `0.5` | 每 tick 饱腹度自然减少量 |
| `relax_decay_rate` | `0.8` | 每 tick 放松度自然减少量 |
| `relax_increase_rate` | `1.2` | 无任务时每 tick 轻微恢复的放松度 |
| `relax_moving_usage` | `1.0` | 每移动一步消耗的放松度 |
| `sleep_time` | `8` | 睡眠持续 tick 数 |
| `sleep_relax_recover` | `60.0` | 一次完整睡眠累计恢复的放松度，按 `sleep_time` 分摊到每个睡眠 tick |
| `urgency_floors` | `{"satiety": 0.08, "relax": 0.08, "money": 0.05}` | 各需求满足时保留的最低非零急迫度 |
| `urgency_gap_weights` | `{"satiety": 1.20, "relax": 1.10, "money": 1.00}` | 各需求缺口对急迫度的线性放大强度 |
| `need_layers` | `{"physiological": ["satiety", "relax"], "safety": ["money"]}` | 需求层级分组 |
| `need_layer_order` | `["physiological", "safety"]` | 需求层级顺序 |
| `need_pressure_dt` | `1.0` | 需求压力记忆每次累积或恢复使用的时间步长 |
| `pressure_default_recovery_rate` | `0.15` | 未配置单项需求时使用的压力记忆恢复率 |
| `pressure_default_load_kappa` | `4.0` | 未配置单项需求时使用的压力负荷饱和系数 |
| `pressure_default_current_weight` | `0.60` | 未配置单项需求时使用的瞬时缺口权重 |
| `pressure_default_residual_weight` | `0.15` | 未配置单项需求时使用的残余负荷权重 |
| `pressure_default_amplification_weight` | `0.25` | 未配置单项需求时使用的缺口与负荷交互权重 |
| `pressure_recovery_rates` | `{"satiety": 0.20, "relax": 0.15, "money": 0.05}` | 各需求的压力记忆恢复率 |
| `pressure_load_kappas` | `{"satiety": 4.0, "relax": 4.0, "money": 8.0}` | 各需求的压力负荷饱和系数 |
| `pressure_current_weights` | `{"satiety": 0.60, "relax": 0.60, "money": 0.60}` | 各需求有效压力中的瞬时缺口权重 |
| `pressure_residual_weights` | `{"satiety": 0.15, "relax": 0.15, "money": 0.15}` | 各需求有效压力中的残余负荷权重 |
| `pressure_amplification_weights` | `{"satiety": 0.25, "relax": 0.25, "money": 0.25}` | 各需求有效压力中的缺口与负荷交互权重 |
| `psychological_assessment_interval` | `10` | 心理评测窗口长度，单位为 tick |
| `psychological_pressure_default_threshold` | `0.5` | 未配置单项需求时使用的心理评测激活阈值 |
| `psychological_pressure_thresholds` | `{"satiety": 0.5, "relax": 0.5, "money": 0.5}` | 各需求评测器的有效压力激活阈值 |
| `psychological_assessment_mode` | `"llm"` | 心理评测模式；默认由 LLM 结合理论卡生成中介变量与角色卡 |
| `psychological_llm_fallback_to_rule` | `True` | LLM 心理评测失败时是否回退理论卡规则评测 |
| `opinion_assessment_mode` | `"llm"` | 观念评测模式；默认由 LLM 根据上下文输出系统新闻主题 opinion score，并写回 `agent.opinion` |
| `default_online_trust` | `0.5` | 默认在线信任值 |
| `simulation_step_limit` | `100` | 仿真时间步上限 |
| `micro_reflect_interval` | `3` | 无进展后触发 micro-reflection 的 tick 数 |

默认场景使用分层初始金钱：`agent_1=8`、`agent_2=14`、`agent_3=18`、`agent_4=24`、`agent_5=12`。金钱获取主要来自公司：`company_1` 每次获得 `8` 元并消耗 `8` relax，`company_2` 每次获得 `12` 元并消耗 `12` relax。金钱消耗主要来自食品店和游乐场：`shop_1` 花费 `6` 元恢复 `30` satiety，`shop_2` 花费 `4` 元恢复 `20` satiety，`playground_1` 花费 `5` 元恢复 `18` relax。食品店和游乐场余额不足时不会发生消费，也不会恢复需求。

`satiety` 和 `relax` 限制在 `[0, 100]`，`money` 下限为 `0` 且没有上限；`opinion` 限制在 `[-1, 1]`。

急迫度由满足度缺口直接计算。对需求 \(k\)，满足度为 \(s_k\)，阈值为 \(\theta_k\)，归一化缺口为：

\[
g_k = \max\left(0,\frac{\theta_k - s_k}{\theta_k}\right)
\]

原始急迫度为：

\[
u_k^{raw}=u_k^{floor}+\alpha_k g_k
\]

最终急迫度限制为：

\[
u_k=\mathrm{clip}\left(u_k^{raw},u_k^{floor},1\right)
\]

其中 \(u_k^{floor}\) 表示需求满足时仍保留的最低急迫度，\(\alpha_k\) 表示缺口放大强度。当前生理层由 `satiety` 和 `relax` 组成：

\[
U_{\mathrm{phys}}=\max(u_{\mathrm{satiety}},u_{\mathrm{relax}})
\]

当生理层未满足时，安全层代理需求 `money` 的急迫度被生理层封顶：

\[
u_{\mathrm{money}}=\min(u_{\mathrm{money}}^{raw},U_{\mathrm{phys}})
\]

当生理层已满足时，`money` 按自身缺口计算，不受生理层封顶。每次 `satisfaction` 变化后都会重新计算 `urgency`。

需求压力层的归一化缺口为 `max(0, threshold - satisfaction) / threshold`。当缺口大于 `0` 时，`pressure_memory` 按缺口累积；当缺口为 `0` 时，`pressure_memory` 按恢复率衰减。`load_saturation` 由 `pressure_memory` 经指数饱和函数得到，`effective_pressure` 由瞬时缺口、残余负荷以及二者交互项加权得到。睡眠期间不执行普通需求衰减，只按 `sleep_relax_recover / sleep_time` 分步恢复 `relax`；该恢复会刷新当前 `need_gap` 和 `effective_pressure`，但不会累积新的 `pressure_memory`。
