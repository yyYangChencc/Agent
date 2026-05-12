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
