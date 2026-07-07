# Agent Simulation

基于大语言模型（LLM）的多智能体网格仿真平台。项目将物理世界、需求状态、向量记忆、智能体对话、社交平台和意见传播组合在同一套 tick 循环中，并通过 React + Pixi.js 页面实时展示仿真状态。

## 当前功能

- **LLM 驱动决策**：智能体根据局部观测、历史记录、向量记忆、任务和需求状态生成行动。
- **异步并行执行**：每个 tick 内并发执行各智能体的决策与反思流程。
- **双层需求状态**：`satisfaction` 表示客观满足度，`urgency` 表示主观急迫度；当前运行时需求键为 `satiety`、`relax`、`money`、`belonging`、`esteem`、`self_actualization`。系统不设置独立 `safety` 满足度，`money` 作为经济安全代理需求。
- **需求压力层**：在 `satisfaction` 之上维护 `need_gap`、`pressure_memory`、`load_saturation` 和 `effective_pressure`，用于表达瞬时需求缺口、累积压力记忆、压力负荷饱和度和当前有效压力；睡眠期间只分步恢复 `relax`，不执行普通需求衰减，也不累积新的压力记忆。
- **心理评测接口**：每隔 `psychological_assessment_interval` 个 tick 汇总一个评测窗口；当窗口内某项 `effective_pressure` 超过阈值时激活对应需求评测器。默认使用 LLM 结合理论卡、窗口经历、需求变化、有效压力和上一轮结果生成心理中介与动态角色卡；LLM 不可用或输出异常时回退理论卡规则评测，多个评测器同时激活时进入现有裁判整合。
- **网格世界**：固定使用 25×25 地图，支持移动、进食、睡眠、进入建筑、建筑内自动交互、离开建筑和近距离对话。
- **场景系统**：初始化逻辑已拆到 `backend/scenarios/`。默认场景为 `default_town`；实验入口还可选择 `jiang_ping_polarization`，用于姜萍事件舆论极化测试。
- **社交平台**：智能体可以发帖、评论、点赞和点踩，并接收关注对象的帖子及系统新闻。
- **观念评测**：不再使用线上帖子均值或线下邻居均值的公式化观念更新；`opinion` 表示智能体对系统投放新闻主题的当前立场。每个 tick 末由观念评测模块只评估该系统新闻主题，并将评测得到的 score 写回 `agent.opinion`。默认使用 LLM 评测，失败时回退规则评测。
- **定时新闻与投放者**：`World` 从场景注入的 `news_schedule` 和 `influencer_schedule` 读取排期；官方新闻以系统新闻注入，无实体投放者只在线上按时间线发帖。
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

后端启动时会初始化 `default_town` 场景并创建记忆管理器；如果缺少可用的 `OPENAI_API_KEY` 或 `EMBEDDING_KEY`，涉及 LLM 或嵌入检索的仿真流程无法正常完成。测试环境未安装 `chromadb` 时会使用内存后备，该模式不持久化向量记忆。

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
- **场景选择**：选择 `default_town` 或 `jiang_ping_polarization`，点击“加载场景”后重建对应 runtime。
- **速度**：切换 1×、2×、5× 自动推进速度。

Web 前端默认加载 `default_town`。如果想让后端启动时直接进入姜萍事件极化场景，可以在启动前设置：

```powershell
$env:SIM_SCENARIO="jiang_ping_polarization"
python backend/server.py
```

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

以下命令加载 `default_town` 默认场景并运行 10 个 tick，在终端打印各智能体的意见值：

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

1. 世界时间加 1，刷新本 tick 的移动记录，并取得全部实体智能体。
2. 按场景注入的 `news_schedule` 投放官方新闻，按 `influencer_schedule` 投放无实体线上账号帖子。
3. 对已经进入建筑且未睡眠的智能体执行建筑内自动交互。
4. 并发执行每个实体智能体的行动阶段：本 tick 开始时已经睡眠的智能体只推进睡眠剩余时间并分步恢复 `relax`，未睡眠的智能体执行观测、记忆检索、LLM 决策和动作。
5. 并发执行普通反思与需求状态更新；本 tick 开始时已经睡眠的智能体即使在本 tick 醒来也不会立刻执行普通反思和普通需求衰减。需求变化后同步刷新 `urgency`、`need_gap`、`pressure_memory`、`load_saturation` 和 `effective_pressure`。
6. 执行智能体对话阶段。
7. 并发执行评测阶段；单个智能体内部先执行心理评测并更新动态角色卡，再围绕 `default_opinion_topic` 执行观念评测并写回 `agent.opinion`。
8. 将智能体状态写入 CSV 和 JSONL 历史文件。

## 意见传播

意见值范围为 `[-1, 1]`。当前默认主题为 `姜萍事件`：`-1` 表示激烈反对姜萍并可能上升到反对媒体造神、竞赛公信力或教育叙事层面，`0` 表示不关心、没听说过或暂不表态，`1` 表示完全赞成或高度支持姜萍。

- **观念评测**：每个 tick 末只评估系统新闻主题 `default_opinion_topic`；评测输入包括近期经历、近期社交文本、记忆、完整动态角色卡和当前 `opinion`。评测 score 会写回 `agent.opinion`，同时镜像记录到 `opinion_scores[topic]` 和 `last_opinion_assessment` 供展示与分析使用。
- **不再使用公式化更新**：系统不再按线上帖子均值或线下邻居均值直接计算观念变化。
- **信任变化**：点赞他人帖子会将对应 `online_trust` 增加 `0.05`，点踩会减少 `0.05`，结果限制在 `[0, 1]`。
- **系统新闻**：新闻由所选场景的 `official_news_schedule` 注入。`default_town` 的官方新闻排期定义在 `backend/scenarios/default_town.py`；`jiang_ping_polarization` 只投放非结论性中性背景，不投放真实最终违规结果。
- **无实体投放者**：`jiang_ping_polarization` 场景注册 6 个线上投放者账号。投放者不存在于 `world.agents`，没有地图位置、需求、心理评测、记忆和行动循环；它们只按 `influencer_schedule` 发布 `source_type="influencer"` 的普通社交帖子。

智能体通过 `send_post` 新发帖时，动作参数必须显式提供 `opinion_index`，表示作者发布该帖时对帖子主题的立场快照。工具层只校验并记录该数值，不再对普通智能体发帖调用 `evaluate_opinion()` 二次评分。`backend/persona/opinion/scorer.py` 的 `evaluate_opinion()` 仍保留给旧数据兼容、外部文本评分或规则回退测试使用。

## 实验入口

当前建议从固定实验脚本 [backend/run_experiment.py](/D:/agent/Agent/backend/run_experiment.py) 开始，而不是从 Web 前端或 [backend/main.py](/D:/agent/Agent/backend/main.py) 开始。原因是该入口会固定随机种子、保存配置快照、导出 CSV/JSONL，并自动调用 [backend/analyze_history.py](/D:/agent/Agent/backend/analyze_history.py) 生成摘要、曲线图和极化统计。

第一步先跑无外部模型的 20 tick 冒烟测试，确认世界循环、场景加载、投放者排期和导出链路正常：

```powershell
python backend/run_experiment.py --scenario jiang_ping_polarization --ticks 20 --version full --opinion-mode rule --psychology-mode rule --llm mock
```

第二步跑正式 100 tick 极化实验：

```powershell
python backend/run_experiment.py --scenario jiang_ping_polarization --ticks 100 --version full --opinion-mode llm --psychology-mode llm --llm real --seed 42
```

第三步跑最小消融，用同一场景和同一种子对比心理中介与动态角色卡的作用：

```powershell
python backend/run_experiment.py --scenario jiang_ping_polarization --ticks 100 --version no_psychology --opinion-mode llm --psychology-mode llm --llm real --seed 42
python backend/run_experiment.py --scenario jiang_ping_polarization --ticks 100 --version no_role_card --opinion-mode llm --psychology-mode llm --llm real --seed 42
```

默认输出目录是 `backend/history/<timestamp>/`。每次运行会写入 `config_snapshot.json`、每个智能体的 CSV、每个智能体的 JSONL，并由 `backend/analyze_history.py` 生成 `experiment_summary.md`、`opinion_trends.svg`、`effective_pressure_trends.svg`、`mediator_peak_trends.svg`、`polarization_report.md`、`polarization_metrics.csv` 和 `polarization_agent_shift.csv`。

已有 history 目录也可以手动重新统计：

```powershell
python backend/analyze_history.py backend/history/20260628_181756 --data-length 70 --window-size 10
```

如果只想按精确列名绘图，也使用同一个程序：

```powershell
python backend/analyze_history.py backend/history/20260628_181756 --plot-columns opinion --data-length 70 --x-column tick
```

如果只是调试默认小镇场景，可以显式改用：

```powershell
python backend/run_experiment.py --scenario default_town --ticks 100 --version full --opinion-mode llm --psychology-mode llm --llm real --seed 42
```

`config_snapshot.json` 会记录 `scenario_name`、`scenario_agent_ids`、`influencers`、`official_news_schedule` 和 `influencer_schedule`，用于复现实验初始化条件。

## 运行产物

- `logs/agent_<timestamp>.log`：完整 DEBUG 日志；控制台显示 INFO 及以上日志。
- `backend/history/<timestamp>/<agent_id>.csv`：Web 服务和实验脚本记录的智能体历史，字段包括需求、急迫度、缺口、压力记忆、负荷饱和、有效压力、心理中介、动态角色卡、行动工具、社交内容、观念评测前后值、评测理由和任务。
- `backend/history/<timestamp>/<agent_id>.jsonl`：逐 tick 解释链，保留与 CSV 同步的结构化记录，便于追踪单个智能体的观念变化原因；其中 `seen_topic_posts` 表示智能体实际浏览过的当前主题帖子，`visible_topic_posts` 表示该 tick 结束时平台上对该智能体可见的当前主题帖子。
- `backend/history/<timestamp>/experiment_summary.md`：固定实验汇总，包含配置快照、最终观念分布、压力峰值、行为分布、发帖立场和明显观念变化。
- `backend/history/<timestamp>/*.svg`：实验汇总图，包括观念曲线、有效压力曲线和心理中介峰值曲线。
- `backend/history/<timestamp>/polarization_report.md`：舆论极化统计报告，包含前后窗口指标、配对符号检验和 bootstrap 置信区间。
- `backend/history/<timestamp>/polarization_metrics.csv`：逐 tick 极化指标，包括立场离散度、双边阵营占比、中立占比和综合极化指数。
- `backend/history/<timestamp>/polarization_agent_shift.csv`：每个智能体在基线窗口和末端窗口之间的立场变化。
- `chroma_agents/`：ChromaDB 持久化数据。


## 项目结构

```text
Agent/
├── backend/
│   ├── analyze_history.py              # 统一 history 统计、图表和极化分析入口
│   ├── main.py                         # 命令行仿真入口
│   ├── server.py                       # FastAPI、WebSocket、默认场景与仿真控制
│   ├── test_spatial.py                 # 无 LLM 的空间交互演示服务
│   ├── persona/
│   │   ├── config.py                   # AgentConfig
│   │   ├── runtime.py                  # SimulationRuntime 组合根
│   │   ├── history_recorder.py         # CSV/JSONL 历史记录
│   │   ├── agents/                     # Agent、策略、提示词与动作解析
│   │   ├── agent_memory/               # ChromaDB 多智能体记忆
│   │   ├── llm/                        # OpenAI 同步/异步客户端与 mock 验收客户端
│   │   ├── opinion/                    # 系统主题观念评测与文本评分兼容工具
│   │   ├── psychology/                 # 心理评测窗口、需求评测器接口与裁判整合接口
│   │   └── reflect/                    # 任务反思与 micro-reflection
│   ├── scenarios/                      # 场景注册表、地图设计、默认小镇与姜萍极化场景
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
| `llm_model` | `deepseek-v4-flash` | 对话模型名 |
| `llm_timeout_seconds` | `120.0` | 单次 LLM 对话调用超时时间 |
| `llm_connect_timeout_seconds` | `5.0` | LLM 对话服务连接超时时间 |
| `llm_max_concurrent_requests` | `2` | LLM 对话请求全局并发上限，用于避免 429 并发限流 |
| `llm_rate_limit_retries` | `3` | LLM 对话遇到 429 后的退避重试次数 |
| `llm_rate_limit_backoff_seconds` | `2.0` | LLM 对话 429 退避重试的基础等待秒数 |
| `embedding_model` | `text-embedding-3-small` | 嵌入模型名 |
| `embedding_timeout_seconds` | `15.0` | 单次嵌入调用超时时间 |
| `embedding_max_concurrent_requests` | `2` | 嵌入请求全局并发上限 |
| `observation_radius` | `5` | 观测半径；移动距离不再由该值封顶 |
| `eat_distance_sq` | `2.0` | 进食、睡眠和进入建筑的平方距离限制 |
| `max_history` | `12` | 智能体短期历史条数 |
| `memory_top_k` | `5` | 向量记忆检索条数 |
| `memory_retrieval_timeout_seconds` | `8.0` | 记忆检索超时预算，超时后返回已取得结果 |
| `memory_retrieval_ttl_ticks` | `3` | 相同检索在短时间内复用结果，减少相邻 tick 重复检索 |
| `satiety_threshold` | `30.0` | 饱腹任务满足阈值 |
| `relax_threshold` | `30.0` | 放松任务满足阈值 |
| `money_threshold` | `15.0` | 金钱/安全需求满足阈值 |
| `satiety_decay_rate` | `0.5` | 每 tick 饱腹度自然减少量 |
| `relax_decay_rate` | `0.0` | 兼容旧配置；当前规则不再每 tick 自然衰减放松度 |
| `relax_increase_rate` | `1.2` | 未移动且未工作时每 tick 轻微恢复的放松度 |
| `relax_moving_usage` | `1.0` | 每移动一步消耗的放松度 |
| `sleep_time` | `8` | 睡眠持续 tick 数 |
| `sleep_relax_recover` | `60.0` | 一次完整睡眠累计恢复的放松度，按 `sleep_time` 分摊到每个睡眠 tick |
| `belonging_threshold` | `30.0` | 归属需求满足阈值 |
| `esteem_threshold` | `30.0` | 尊重需求满足阈值 |
| `self_actualization_threshold` | `30.0` | 自我实现需求满足阈值 |
| `urgency_floors` | `{"satiety": 0.08, "relax": 0.08, "money": 0.05, "belonging": 0.04, "esteem": 0.04, "self_actualization": 0.03}` | 各需求满足时保留的最低非零急迫度 |
| `urgency_gap_weights` | `{"satiety": 1.20, "relax": 1.10, "money": 1.00, "belonging": 0.90, "esteem": 0.85, "self_actualization": 0.80}` | 各需求缺口对急迫度的线性放大强度 |
| `need_layers` | `{"physiological": ["satiety", "relax"], "safety": ["money"], "belonging": ["belonging"], "esteem": ["esteem"], "self_actualization": ["self_actualization"]}` | 需求层级分组；其中 `safety` 层由 `money` 代理 |
| `need_layer_order` | `["physiological", "safety", "belonging", "esteem", "self_actualization"]` | 需求层级顺序 |
| `need_pressure_dt` | `1.0` | 需求压力记忆每次累积或恢复使用的时间步长 |
| `pressure_default_recovery_rate` | `0.15` | 未配置单项需求时使用的压力记忆恢复率 |
| `pressure_default_load_kappa` | `4.0` | 未配置单项需求时使用的压力负荷饱和系数 |
| `pressure_default_current_weight` | `0.60` | 未配置单项需求时使用的瞬时缺口权重 |
| `pressure_default_residual_weight` | `0.15` | 未配置单项需求时使用的残余负荷权重 |
| `pressure_default_amplification_weight` | `0.25` | 未配置单项需求时使用的缺口与负荷交互权重 |
| `pressure_recovery_rates` | `{"satiety": 0.20, "relax": 0.15, "money": 0.05, "belonging": 0.10, "esteem": 0.10, "self_actualization": 0.08}` | 各需求的压力记忆恢复率 |
| `pressure_load_kappas` | `{"satiety": 4.0, "relax": 4.0, "money": 8.0, "belonging": 6.0, "esteem": 6.0, "self_actualization": 8.0}` | 各需求的压力负荷饱和系数 |
| `pressure_current_weights` | `{"satiety": 0.60, "relax": 0.60, "money": 0.60, "belonging": 0.55, "esteem": 0.55, "self_actualization": 0.50}` | 各需求有效压力中的瞬时缺口权重 |
| `pressure_residual_weights` | `{"satiety": 0.15, "relax": 0.15, "money": 0.15, "belonging": 0.15, "esteem": 0.15, "self_actualization": 0.15}` | 各需求有效压力中的残余负荷权重 |
| `pressure_amplification_weights` | `{"satiety": 0.25, "relax": 0.25, "money": 0.25, "belonging": 0.20, "esteem": 0.20, "self_actualization": 0.20}` | 各需求有效压力中的缺口与负荷交互权重 |
| `psychological_assessment_interval` | `10` | 心理评测窗口长度，单位为 tick |
| `psychological_pressure_default_threshold` | `0.5` | 未配置单项需求时使用的心理评测激活阈值 |
| `psychological_pressure_thresholds` | `{"satiety": 0.5, "relax": 0.5, "money": 0.5, "belonging": 0.5, "esteem": 0.5, "self_actualization": 0.5}` | 各需求评测器的有效压力激活阈值 |
| `psychological_assessment_mode` | `"llm"` | 心理评测模式；默认由 LLM 结合理论卡生成中介变量与角色卡 |
| `psychological_llm_fallback_to_rule` | `True` | LLM 心理评测失败时是否回退理论卡规则评测 |
| `dynamic_role_card_enabled` | `True` | 是否把动态心理角色卡注入行为和观念评测 prompt，消融实验可关闭 |
| `opinion_assessment_mode` | `"llm"` | 观念评测模式；默认由 LLM 根据上下文输出系统新闻主题 opinion score，并写回 `agent.opinion` |
| `opinion_assessment_interval` | `5` | 观念评测事件触发之外的间隔兜底 |
| `opinion_assessment_triggered_only` | `True` | 无新主题证据时不重复调用观念评测 |
| `opinion_max_delta_per_assessment` | `0.25` | 单次观念评测最大变化幅度 |
| `post_opinion_scoring_mode` | `"llm"` | 兼容旧数据或外部文本评分的 `evaluate_opinion()` 模式；普通 `send_post` 已改为动作显式提供 `opinion_index` |
| `post_opinion_llm_fallback_to_rule` | `True` | `evaluate_opinion()` 在 LLM 失败时是否规则回退 |
| `default_online_trust` | `0.5` | 默认在线信任值 |
| `simulation_step_limit` | `100` | 仿真时间步上限 |
| `micro_reflect_interval` | `3` | 无进展后触发 micro-reflection 的 tick 数 |

默认场景使用分层初始金钱：`agent_1=8`、`agent_2=14`、`agent_3=18`、`agent_4=24`、`agent_5=12`。金钱获取主要来自公司：`company_1` 每次获得 `8` 元并消耗 `8` relax，`company_2` 每次获得 `12` 元并消耗 `12` relax。金钱消耗主要来自食品店和游乐场：`shop_1` 花费 `6` 元恢复 `30` satiety，`shop_2` 花费 `4` 元恢复 `20` satiety，`playground_1` 花费 `5` 元恢复 `18` relax。食品店和游乐场余额不足时不会发生消费，也不会恢复需求。

`satiety`、`relax`、`belonging`、`esteem`、`self_actualization` 限制在 `[0, 100]`，`money` 下限为 `0` 且没有上限；`opinion` 限制在 `[-1, 1]`。

理论卡文件位于 `backend/persona/psychology/theory_cards.json`。当前理论卡只覆盖上述六个运行时需求键；`money` 卡的 `maslow_need` 为 `safety`，表示经济资源对安全需求的代理，不存在独立 `need_key="safety"` 理论卡。

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
