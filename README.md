# Agent Simulation

基于大语言模型（LLM）的多智能体网格仿真平台。项目将物理世界、需求状态、向量记忆、智能体对话、社交平台和意见传播组合在同一套 tick 循环中，并通过 React + Pixi.js 页面实时展示仿真状态。

## 当前功能

- **LLM 驱动决策**：智能体根据局部观测、历史记录、向量记忆、任务和需求状态生成行动。
- **异步并行执行**：每个 tick 内并发执行各智能体的决策与反思流程。
- **双层需求状态**：`satisfaction` 表示客观满足度，`urgency` 表示主观急迫度；当前需求键为 `satiety`、`relax`、`money`。
- **网格世界**：固定使用 25×25 地图，支持移动、进食、睡眠、进入建筑、建筑内自动交互、离开建筑和近距离对话。
- **场景设施**：Web 服务默认创建 5 个智能体、5 张床、2 家公司、2 家食品店、1 个游乐场和 5 处散落食物。
- **社交平台**：智能体可以发帖、评论、点赞和点踩，并接收关注对象的帖子及系统新闻。
- **意见传播**：浏览帖子后执行在线意见更新；每隔固定 tick，根据 `offline_trust` 对意见和 `urgency` 执行离线同化。
- **定时新闻**：`backend/persona/news_events.py` 在 tick 5、15、25、35 投放预设新闻。
- **向量记忆**：使用 ChromaDB 保存多智能体长期记忆，并通过嵌入接口执行语义检索。
- **运行记录**：后端将日志写入 `logs/`，Web 服务将每个智能体的 tick 历史写入 `backend/history/`。
- **Web 可视化**：显示地图、智能体、设施、帖子、需求状态、最近思考和最多 200 个前端历史点的趋势图。

## 技术栈

**后端**：Python、FastAPI、WebSocket、ChromaDB、OpenAI Python SDK

**前端**：React 18、TypeScript、Pixi.js 8、Zustand、Recharts、Tailwind CSS、Vite

## 安装

### 运行环境

- Python 3.10+
- Node.js 18+
- 与 OpenAI Python SDK 接口兼容的对话和嵌入服务

### Python 依赖

在项目根目录执行：

```bash
pip install -r requirements.txt
```

### 环境变量

项目根目录的 `.env` 文件会被 `backend/main.py` 和 `backend/server.py` 加载。当前代码读取以下变量：

```dotenv
OPENAI_API_KEY=<对话接口密钥>
OPENAI_BASE_URL=<对话接口地址>
EMBEDDING_KEY=<嵌入接口密钥>
EMBEDDING_BASE_URL=<嵌入接口地址>
```

`OPENAI_BASE_URL` 和 `EMBEDDING_BASE_URL` 未设置时，代码使用 `https://api.openai.com/v1`。

### 前端依赖

```bash
cd frontend
npm install
```

## 启动 Web 界面

### 开发模式

终端 1，在项目根目录启动 FastAPI 服务：

```bash
python backend/server.py
```

终端 2，启动 Vite 开发服务器：

```bash
cd frontend
npm run dev
```

浏览器访问 `http://localhost:5173`。Vite 会将 `/ws` 代理到 `ws://localhost:8000`。

### 构建后运行

先构建前端：

```bash
cd frontend
npm run build
cd ..
```

再由 FastAPI 托管 `frontend/dist`：

```bash
python backend/server.py
```

浏览器访问 `http://localhost:8000`。

## 其他运行入口

### 命令行仿真

以下命令创建 5 个智能体并运行 10 个 tick，在终端打印各智能体的意见值：

```bash
python backend/main.py
```

### 空间交互演示

`backend/test_spatial.py` 是不调用 LLM 的空间交互演示服务，用于按固定步骤展示睡眠、进入建筑、建筑内自动交互、离开建筑和建筑内移动。

终端 1：

```bash
python backend/test_spatial.py
```

终端 2：

```bash
cd frontend
npm run dev
```

浏览器访问 `http://localhost:5173`。运行前需要停止 `backend/server.py`，因为两个服务都监听 8000 端口。

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

1. 世界时间加 1，并取得全部智能体。
2. 并发执行每个智能体的观测、记忆检索、LLM 决策和动作。
3. 并发执行反思与需求状态更新。
4. 执行在线意见更新；到达 `offline_update_interval` 时执行离线意见与 `urgency` 同化。
5. 执行智能体对话阶段。
6. 按 `NEWS_SCHEDULE` 投放当前 tick 的系统新闻。
7. 将智能体状态写入 CSV 历史文件。

## 意见传播

意见值范围为 `[0, 1]`。

- **在线更新**：智能体调用 `social_step` 浏览帖子后，向已读帖子的信任加权平均意见移动。
- **离线更新**：每隔 `offline_update_interval` 个 tick，与 `offline_trust >= friend_trust_threshold` 的智能体执行意见同化，并同步影响各需求的 `urgency`。
- **信任变化**：点赞他人帖子会将对应 `online_trust` 增加 `0.05`，点踩会减少 `0.05`，结果限制在 `[0, 1]`。
- **系统新闻**：新闻使用 `backend/persona/news_events.py` 中明确设置的 `opinion_index`。

当前 `backend/persona/opinion/scorer.py` 的 `evaluate_opinion()` 固定返回 `0.5`，因此智能体通过 `send_post` 新发帖时，`opinion_index` 目前统一为 `0.5`。基于自然语言内容计算立场值的逻辑尚未实现。

## 运行产物

- `logs/agent_<timestamp>.log`：完整 DEBUG 日志；控制台显示 INFO 及以上日志。
- `backend/history/<timestamp>/<agent_id>.csv`：Web 服务记录的智能体历史，字段包括 `tick`、`opinion`、`satiety`、`relax`、`money`、急迫度、情绪和任务。
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
| `money_threshold` | `0.3` | 金钱任务满足阈值 |
| `satiety_decay_rate` | `0.5` | 每 tick 饱腹度自然减少量 |
| `relax_decay_rate` | `2.0` | 每 tick 放松度自然减少量 |
| `relax_moving_usage` | `2.0` | 每移动一步消耗的放松度 |
| `sleep_time` | `8` | 睡眠持续 tick 数 |
| `sleep_relax_recover` | `50.0` | 睡眠结束恢复的放松度 |
| `online_opinion_lr` | `0.1` | 在线意见更新强度 |
| `self_confidence` | `0.5` | 对外部意见的抵抗系数 |
| `offline_opinion_lr` | `0.1` | 离线同化强度 |
| `offline_update_interval` | `3` | 离线更新间隔 |
| `default_online_trust` | `0.5` | 默认在线信任值 |
| `default_offline_trust` | `0.5` | 默认离线信任值 |
| `friend_trust_threshold` | `0.6` | 纳入离线同化的信任阈值 |
| `micro_reflect_interval` | `3` | 无进展后触发 micro-reflection 的 tick 数 |

`satiety` 和 `relax` 限制在 `[0, 100]`，`money` 下限为 `0` 且没有上限；`opinion` 限制在 `[0, 1]`。
