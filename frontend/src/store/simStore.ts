import { create } from 'zustand'

// 与后端 serializer.py snapshot() 输出字段保持一致
export interface AgentState {
  id: string
  pos: [number, number]       // [col, row]，对应画布坐标 pos * CELL
  role: string
  emotion: string
  task: string
  current_focus: string       // micro-reflect 更新的当前策略焦点
  salary: number              // 工资：参与 work 动作时获得的 money 增量
  need: Record<string, number>            // 客观需求，0→1
  demand: Record<string, number>          // 主观急迫度，1→0
  demand_threshold: Record<string, number> // 任务完成阈值
  opinion: number             // 意见倾向，0~1
  last_think: string          // 最近一次 <Think> 内容
  sleeping: boolean           // 是否处于睡眠状态
  sleep_ticks_remaining: number  // 剩余睡眠步数
  inside_building_id: string | null  // 当前所在建筑 ID，null 表示在室外
}

export interface ObjectState {
  id: string
  pos: [number, number]
  type: string
  kind: string                // 物品种类标识，与 objects.py 中 self.kind 一致
  description: string         // 物品的人类可读描述，由后端 objects.get_desc() 动态生成
  num: number | null          // null 表示无数量属性；<=0 时隐藏渲染
  occupant_count: number      // 当前在建筑内的智能体数量
  occupants: string[]         // 当前占用者的智能体 ID 列表
}

export interface CommentState {
  id: string
  author_id: string
  content: string
  time: number | null
}

export interface PostState {
  id: number
  author_id: string
  content: string
  time: number
  likes: number
  dislikes: number
  comments: CommentState[]
  opinion_index: number
}

export interface WorldState {
  time: number
  map_size: [number, number]
  agents: AgentState[]
  objects: ObjectState[]
  posts: PostState[]
}

export interface AgentHistoryPoint {
  tick: number
  opinion: number
  satiety: number
  relax: number
  money: number
  satiety_demand: number
  relax_demand: number
  emotion: string
}

const MAX_HISTORY = 200

interface SimStore {
  worldState: WorldState | null
  running: boolean
  speed: number
  selectedAgentId: string | null
  selectedObjectId: string | null   // 当前选中的场景物品 ID
  ws: WebSocket | null
  connected: boolean
  agentHistory: Record<string, AgentHistoryPoint[]>
  setWorldState: (state: WorldState) => void
  setStatus: (running: boolean, speed: number) => void
  selectAgent: (id: string | null) => void
  selectObject: (id: string | null) => void  // 选中物品，同时取消智能体选中
  setWs: (ws: WebSocket | null) => void
  setConnected: (connected: boolean) => void
  sendCmd: (cmd: object) => void
  resetHistory: () => void
}

export const useSimStore = create<SimStore>((set, get) => ({
  worldState: null,
  running: false,
  speed: 1.0,
  selectedAgentId: null,
  selectedObjectId: null,
  ws: null,
  connected: false,
  agentHistory: {},

  setWorldState: (worldState) => {
    const prev = get().agentHistory
    const next: Record<string, AgentHistoryPoint[]> = { ...prev }
    for (const a of worldState.agents) {
      const point: AgentHistoryPoint = {
        tick: worldState.time,
        opinion: a.opinion,
        satiety: a.need.satiety ?? 0,
        relax: a.need.relax ?? 0,
        money: a.need.money ?? 0,
        satiety_demand: a.demand.satiety ?? 0,
        relax_demand: a.demand.relax ?? 0,
        emotion: a.emotion,
      }
      const arr = prev[a.id] ?? []
      const updated = [...arr, point]
      next[a.id] = updated.length > MAX_HISTORY ? updated.slice(-MAX_HISTORY) : updated
    }
    set({ worldState, agentHistory: next })
  },

  setStatus: (running, speed) => set({ running, speed }),
  // 选中智能体时清除物品选中，保持互斥
  selectAgent: (id) => set({ selectedAgentId: id, selectedObjectId: null }),
  // 选中物品时清除智能体选中，保持互斥
  selectObject: (id) => set({ selectedObjectId: id, selectedAgentId: null }),
  setWs: (ws) => set({ ws }),
  setConnected: (connected) => set({ connected }),
  resetHistory: () => set({ agentHistory: {} }),

  sendCmd: (cmd) => {
    const { ws } = get()
    // readyState 校验防止在连接建立前或关闭后发送消息
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(cmd))
    }
  },
}))
