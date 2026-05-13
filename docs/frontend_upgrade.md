# 前端界面升级清单与技术方案

## 升级功能清单

### A. 世界可视化
- 实时 2D 网格渲染（地图、智能体位置、食物等对象）
- 智能体移动动画（帧间平滑过渡）
- 对象颜色 / 图标区分（agent、food、Uninteractable 对象）
- 观测半径可视化圆圈

### B. 智能体状态面板
- 每个智能体的需求值实时仪表盘（hunger、relax 进度条 + 阈值标记线）
- 当前任务与情绪标签展示
- 上一轮 `<Think>` 块内容展示（LLM 推理过程可视化）
- role / speaking_style 信息卡片

### C. 仿真控制
- 手动单步执行（Step）
- 暂停 / 继续 / 调速（1×、2×、5×）
- 运行时修改 AgentConfig 参数（relax_decay_rate、hunger_threshold 等）
- 场景编辑器：拖拽放置食物 / 障碍物

### D. 社交平台视图
- 帖子 Feed 实时展示（发帖人、内容、时间步）
- 点赞 / 点踩 / 评论数统计
- 发帖历史时间线，按智能体过滤

### E. 记忆库查看器
- 按智能体浏览 ChromaDB 记忆条目
- 按 `type`（trajectory / reflection）或任务名过滤
- 关键词向量搜索，返回最相关记忆

### F. 历史回放与数据分析
- 每 tick 保存世界状态快照（位置、需求、任务）
- 时间轴拖动回放，逐帧查看
- 需求值 / 奖励折线图（Plotly / Chart.js）
- 智能体间交互热力图

---

## 技术栈选项

### 方案一：Pygame（纯桌面 Python）
| 项目 | 说明 |
|------|------|
| 适用场景 | 快速本地原型、不需要 Web 访问 |
| 优点 | 与当前 Python 项目直接集成、低延迟、无需前后端分离 |
| 缺点 | UI 布局能力弱、复杂面板代码量大、无法远程访问 |
| 推荐搭配 | Pygame（渲染） + Dear PyGui（调试面板） |

### 方案二：Streamlit / Gradio（Python Web 快速原型）
| 项目 | 说明 |
|------|------|
| 适用场景 | 数据展示、记忆库浏览、需求图表 |
| 优点 | 纯 Python 编写、极速上手、内置图表组件 |
| 缺点 | 实时性差（轮询而非推送）、自定义 2D Canvas 渲染困难 |
| 推荐搭配 | Streamlit + Plotly（图表） + st-aggrid（表格） |

### 方案三：FastAPI + WebSocket + 原生 HTML / Canvas
| 项目 | 说明 |
|------|------|
| 适用场景 | 轻量 Web 前端、不引入 JS 框架 |
| 优点 | 依赖极少、渲染逻辑完全可控、WebSocket 实时推送 |
| 缺点 | 手写 Canvas JS 代码量大、组件复用能力弱 |

### 方案四：FastAPI + WebSocket + React/Vue + Pixi.js（推荐完整方案）
| 项目 | 说明 |
|------|------|
| 适用场景 | 完整前端界面，接近 wordX 效果 |
| 优点 | FastAPI 异步非阻塞、WebSocket 推送每 tick 状态、Pixi.js 高性能 2D 渲染（精灵 / 动画 / 网格）、React/Vue 负责状态面板和控制 UI |
| 缺点 | 前后端分离增加复杂度，需要前端工程基础 |
| 推荐搭配 | FastAPI + uvicorn + React + Pixi.js + Zustand（状态管理） |

#### 方案四实现细节

##### 文件结构

```
E:\agent\
├── server.py                      # FastAPI + WebSocket 服务器入口（新建）
├── world/
│   └── serializer.py              # 世界状态 → JSON 序列化（新建）
└── frontend/                      # React 前端（新建）
    ├── package.json
    ├── vite.config.ts             # /ws 代理到后端 8000
    └── src/
        ├── main.tsx
        ├── App.tsx                # 整体布局
        ├── store/
        │   └── simStore.ts       # Zustand 全局状态
        ├── hooks/
        │   └── useWebSocket.ts   # WebSocket 连接与消息分发
        └── components/
            ├── WorldCanvas.tsx   # Pixi.js 2D 网格渲染
            ├── AgentPanel.tsx    # need/demand 仪表盘
            ├── ControlBar.tsx    # 步进/暂停/速度控制
            ├── SocialFeed.tsx    # 社交平台 Feed（阶段二）
            └── MemoryViewer.tsx  # 记忆库查看（阶段二）
```

##### 世界状态 JSON（后端 → 前端，每 tick 推送）

```json
{
  "type": "tick",
  "state": {
    "time": 5,
    "map_size": [25, 25],
    "agents": [
      {
        "id": "agent_1",
        "pos": [3, 3],
        "role": "保守主义者",
        "emotion": "平静",
        "task": "eat something",
        "current_focus": "搜寻并吃到食物",
        "need": {"satiety": 0.12, "relax": 0.65},
        "demand": {"satiety": 0.88, "relax": 0.35},
        "demand_threshold": {"satiety": 0.3, "relax": 0.3},
        "opinion": 0.15,
        "last_think": "<Think>...</Think>"
      }
    ],
    "objects": [
      {"id": "food_1", "pos": [10, 10], "type": "food", "num": 2}
    ]
  }
}
```

##### 控制命令（前端 → 后端）

```json
{"cmd": "step"}
{"cmd": "pause"}
{"cmd": "resume"}
{"cmd": "set_speed", "value": 2}
{"cmd": "reset"}
{"cmd": "update_config", "key": "satiety_decay_rate", "value": 0.02}
```

##### 后端回调钩子（world.py 修改）

```python
# World.__init__ 新增
self.on_tick_end = None   # Callable | None，每 tick 结束后触发

# World.step() 末尾（conversation_phase 之后）追加
if self.on_tick_end:
    self.on_tick_end()
```

`server.py` 将此钩子设为广播 WebSocket 消息的触发器，实现每 tick 自动推送。

##### 前端布局

```
┌──────────────── ControlBar（顶部）────────────────────┐
│  [Step] [Pause/Resume] [1× 2× 5×] [Reset]  tick: 5   │
├──────────────────────────┬───────────────────────────┤
│                          │  AgentPanel（右侧）         │
│   WorldCanvas            │  satiety ██████░░ 0.12 ▏  │
│   (Pixi.js 600×600)      │          ← 阈值 0.30 红色  │
│                          │  relax   ████████░░ 0.65  │
│                          │  任务: eat something       │
│                          │  情绪: 平静                 │
└──────────────────────────┴───────────────────────────┘
```

##### Pixi.js 渲染规则

- 格子大小：600 ÷ 25 = 24 px/格
- 智能体：彩色圆形（半径 10），颜色按 ID 散列，圆上方 Text 显示 ID
- 食物：绿色方块（12×12），num > 1 时右下角显示数量
- 选中智能体：黄色描边 + 半透明圆表示 `observation_radius`
- 位移动画：新 tick 到达时，用 `gsap.to` 做 0.3s 平滑位置过渡

##### AgentPanel 进度条规则

```
satiety:  [██████░░░░]  0.12 / 阈值▏0.30  ← need < threshold，红色警告
relax:    [████████░░]  0.65 / 阈值▏0.30  ← need > threshold，绿色满足
```

进度条宽度 = `need[k] × 100%`；阈值标记线为 `threshold[k] × 100%` 处的竖线。

##### 依赖安装与启动

```bash
# 后端
pip install fastapi "uvicorn[standard]"

# 前端
cd frontend
npm create vite@latest . -- --template react-ts
npm install pixi.js zustand gsap
npm install -D tailwindcss

# 开发模式（两个终端）
uvicorn server:app --reload --port 8000
cd frontend && npm run dev        # dev server 端口 5173，/ws 代理到 8000

# 生产部署（单端口）
cd frontend && npm run build      # 输出到 frontend/dist/
uvicorn server:app --port 8000    # FastAPI 同时托管静态文件
```

##### 功能分阶段

| 阶段 | 功能 | 预计工时 |
|------|------|---------|
| 阶段一 | WebSocket 服务器 + Pixi.js 网格 + 智能体状态面板 + 控制栏 | 3~5 天 |
| 阶段二 | `<Think>` 推理展示 + 社交 Feed + 记忆库查看器 + 折线图 | 1~2 周 |

### 方案五：FastAPI + WebSocket + Phaser.js
| 项目 | 说明 |
|------|------|
| 适用场景 | 需要更丰富游戏交互效果（相机缩放、场景切换、粒子） |
| 优点 | 内置 Tilemap、相机、动画状态机，适合仿真演示 |
| 缺点 | 框架较重，学习成本高于 Pixi.js，与 React 集成需要额外适配 |

---

## 推荐升级路线

### 阶段一：快速可视化（1~2 天）
- 工具：**Streamlit + Plotly**
- 实现：需求值折线图、记忆库浏览器、帖子 Feed 展示
- 运行方式：仿真结束后读取日志 / ChromaDB 展示，无需实时

### 阶段二：实时 Web 前端（1~2 周）
- 工具：**FastAPI + WebSocket + React + Pixi.js**
- 实现：
  1. `SimulationRuntime` 改为在 FastAPI 后台线程中运行
  2. 每 tick 结束后通过 WebSocket 广播世界状态 JSON
  3. Pixi.js 前端接收状态，渲染网格与智能体
  4. React 侧边栏展示状态面板、控制按钮、记忆库
- 世界状态 JSON 示例：
  ```json
  {
    "time": 5,
    "agents": [{"id": "agent_1", "pos": [3,3], "task": "eat something", "demand": {"hunger": 0.85}}],
    "objects": [{"id": "food_1", "pos": [10,10], "num": 3}]
  }
  ```
