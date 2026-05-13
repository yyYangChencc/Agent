# Agent Simulation

基于大语言模型（LLM）的多智能体网格仿真平台，包含社交媒体子系统、双层意见传播模型，以及 React + Pixi.js 实时 Web 可视化界面。

## 功能概览

- **LLM 驱动决策**：每个智能体在每个 tick 调用 LLM，根据观测信息、记忆和当前需求状态生成行动
- **Need / Demand 双层需求模型**：`need`（客观物理状态）与 `demand`（主观急迫度）分离，驱动智能体的目标导向行为
- **意见传播系统**：在线更新（看到帖子后实时偏移）+ 离线更新（每 m 个 tick 与好友同化），可观测不同角色间的立场演变
- **社交媒体平台**：智能体可发帖、评论、点赞/踩，影响彼此的在线信任和意见
- **向量记忆**：基于 ChromaDB 的长期记忆，支持语义检索
- **Web 可视化**：25×25 网格实时渲染，支持点击智能体/物品查看详情，控制仿真节奏

## 快速开始

### 依赖要求

- Python 3.10+
- Node.js 18+
- 支持 OpenAI API 格式的 LLM 服务（本地或云端）

### 安装

```bash
# 克隆仓库
git clone <repo-url>
cd agent

# 安装 Python 依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env，填入：
#   OPENAI_API_KEY=<你的 API Key>
#   OPENAI_BASE_URL=<API 端点，如 https://api.openai.com/v1>

# 安装前端依赖
cd frontend
npm install
cd ..
```

### 启动

**开发模式**（推荐，支持热重载）：

```bash
# 终端 1：启动后端
python backend/server.py

# 终端 2：启动前端开发服务器
cd frontend
npm run dev
```

浏览器访问 `http://localhost:5173`

**生产模式**（单进程托管前后端）：

```bash
cd frontend && npm run build && cd ..
uvicorn backend.server:app --port 8000
```

浏览器访问 `http://localhost:8000`

### 命令行运行（无界面）

```bash
# 运行5个智能体、10个 tick 的仿真，打印意见变化
python backend/main.py

# 运行社交子系统演示
python backend/test.py
```

## 界面说明

| 区域 | 功能 |
|------|------|
| 顶部控制栏 | 单步 / 继续 / 暂停 / 重置；速度倍率 1×/2×/5×；当前 tick 数与连接状态 |
| 中央画布 | 25×25 网格，彩色圆形 = 智能体，绿色方块 = 食物；点击可选中 |
| 右侧面板 | 智能体列表（含饱腹度/放松度迷你条）；点击后展开 need/demand 仪表盘、思考内容 |
| 物品详情 | 点击食物方块，右侧显示种类、描述、功能、坐标、剩余数量 |
| 选中高亮 | 智能体选中：黄色描边 + 半透明观测范围圆；食物选中：黄色描边矩形 |

## 项目结构

```
agent/
├── backend/
│   ├── server.py           # FastAPI 应用入口（WebSocket + 仿真控制）
│   ├── main.py             # 命令行仿真入口
│   ├── persona/
│   │   ├── runtime.py      # 组合根，SimulationRuntime.build() 初始化所有服务
│   │   ├── config.py       # AgentConfig 数据类（所有可调参数）
│   │   ├── agents/         # agent.py / policy.py / prompt.py / parser.py
│   │   ├── agent_memory/   # ChromaDB 向量记忆
│   │   ├── reflect/        # 任务生命周期、卡顿检测、micro-reflection
│   │   └── opinion/        # 在线/离线意见更新
│   ├── world/
│   │   ├── world.py        # tick 循环、并行执行、事件广播、奖励计算
│   │   ├── map.py          # 网格地图
│   │   ├── objects.py      # 场景物品基类及 food 子类
│   │   ├── observer.py     # 观测半径内实体扫描
│   │   └── serializer.py   # 世界状态序列化（→ JSON → WebSocket）
│   ├── tools/
│   │   └── operator_tools.py  # move / eat / speak / 社交动作
│   └── social_sys/         # 社交媒体平台（帖子、评论、关注）
├── frontend/
│   └── src/
│       ├── store/simStore.ts       # Zustand 全局状态
│       ├── hooks/useWebSocket.ts   # WebSocket 连接与消息分发
│       └── components/
│           ├── WorldCanvas.tsx     # Pixi.js v8 网格渲染
│           ├── AgentPanel.tsx      # 智能体/物品详情面板
│           └── ControlBar.tsx      # 控制栏
├── .env.example
├── requirements.txt
└── CLAUDE.md
```

## 核心参数

所有参数集中在 `backend/persona/config.py` 的 `AgentConfig` 中：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `llm_model` | `gpt-4o-mini` | 使用的 LLM 模型名 |
| `observation_radius` | `5` | 智能体每 tick 的观测半径（格） |
| `satiety_decay_rate` | `0.01` | 每 tick 饱腹度自然衰减量 |
| `relax_decay_rate` | `0.05` | 每 tick 放松度自然衰减量 |
| `micro_reflect_interval` | `3` | 连续 N tick 无进展后触发 micro-reflection |
| `online_opinion_lr` | `0.1` | 在线意见更新学习率 |
| `self_confidence` | `0.5` | 自我立场坚守度（越高越不易被影响） |
| `offline_update_interval` | `3` | 每隔 N tick 触发一次离线意见同化 |


## 技术栈

**后端**：Python · FastAPI · WebSocket · ChromaDB · OpenAI API  
**前端**：React 18 · Pixi.js v8 · Zustand · Tailwind CSS · Vite
