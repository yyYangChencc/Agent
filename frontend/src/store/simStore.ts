import { create } from 'zustand'

// 与后端 serializer.py snapshot() 输出字段保持一致
export interface AgentState {
  id: string
  pos: [number, number]       // [col, row]，对应画布坐标 pos * CELL
  role: string
  emotion: string
  task: string
  current_focus: string       // micro-reflect 更新的当前策略焦点
  need: Record<string, number>            // 客观需求，0→1
  demand: Record<string, number>          // 主观急迫度，1→0
  demand_threshold: Record<string, number> // 任务完成阈值
  opinion: number             // 意见倾向，0~1
  last_think: string          // 最近一次 <Think> 内容
}

export interface ObjectState {
  id: string
  pos: [number, number]
  type: string
  num: number | null          // null 表示无数量属性；<=0 时隐藏渲染
}

export interface WorldState {
  time: number
  map_size: [number, number]
  agents: AgentState[]
  objects: ObjectState[]
}

interface SimStore {
  worldState: WorldState | null
  running: boolean
  speed: number
  selectedAgentId: string | null
  ws: WebSocket | null
  connected: boolean
  setWorldState: (state: WorldState) => void
  setStatus: (running: boolean, speed: number) => void
  selectAgent: (id: string | null) => void
  setWs: (ws: WebSocket | null) => void
  setConnected: (connected: boolean) => void
  sendCmd: (cmd: object) => void
}

export const useSimStore = create<SimStore>((set, get) => ({
  worldState: null,
  running: false,
  speed: 1.0,
  selectedAgentId: null,
  ws: null,
  connected: false,

  setWorldState: (worldState) => set({ worldState }),
  setStatus: (running, speed) => set({ running, speed }),
  selectAgent: (id) => set({ selectedAgentId: id }),
  setWs: (ws) => set({ ws }),
  setConnected: (connected) => set({ connected }),

  sendCmd: (cmd) => {
    const { ws } = get()
    // readyState 校验防止在连接建立前或关闭后发送消息
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(cmd))
    }
  },
}))
