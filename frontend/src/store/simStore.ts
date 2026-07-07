import { create } from 'zustand'

// 与后端 world/serializer.py 的 snapshot() 输出字段保持一致。
// 如果后端新增/删除字段，需要同步更新这里的类型和 setWorldState 的历史采样逻辑。
export interface AgentState {
  id: string
  pos: [number, number]       // [row, col]
  emotion: string
  task: string
  current_focus: string       // micro-reflect 更新的当前策略焦点
  salary: number              // 工资：公司自动交互时获得的 money 增量
  satisfaction: Record<string, number>            // 客观需求；satiety/relax 为 0~100，money 无上限
  urgency: Record<string, number>          // 主观急迫度，0~1
  satisfaction_threshold: Record<string, number> // 任务完成阈值
  need_gap: Record<string, number>
  pressure_memory: Record<string, number>
  load_saturation: Record<string, number>
  effective_pressure: Record<string, number>
  last_psychological_assessment: Record<string, unknown> | null
  opinion: number             // 对系统新闻主题的意见倾向，-1~1
  opinion_scores: Record<string, number> // 系统新闻主题 score 镜像，用于展示和历史分析
  last_opinion_assessment: Record<string, unknown> | null
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

export type MapBounds = [number, number, number, number]
export type MapPosition = [number, number]

export interface MapTerrainState {
  id: string
  name: string
  kind: string
  bounds: MapBounds
  color: string
  alpha: number
}

export interface MapRegionState {
  id: string
  name: string
  kind: string
  bounds: MapBounds
  label_pos: MapPosition
  color: string
  description: string
}

export interface MapRoadState {
  id: string
  name: string
  kind: string
  cells: MapPosition[]
  color: string
  width: number
}

export interface MapObjectRegionState {
  region_id: string
  entrance: MapPosition
}

export interface MapDesignState {
  version: number
  map_size: [number, number]
  position_format: string
  bounds_format: string
  terrain: MapTerrainState[]
  regions: MapRegionState[]
  roads: MapRoadState[]
  object_regions: Record<string, MapObjectRegionState>
}

export interface CommentState {
  id: string
  author_id: string
  content: string
  time: number | null
  agreement_to_post: number
}

export interface PostState {
  id: number
  author_id: string
  topic: string
  content: string
  time: number
  likes: number
  dislikes: number
  comments: CommentState[]
  opinion_index: number
  is_news: boolean
  is_rumor: boolean
  source_type: string
}

export interface MovementState {
  agent_id: string
  path: MapPosition[]
}

export interface WorldState {
  time: number
  scenario_name: string
  simulation_step_limit: number
  map_size: [number, number]
  map_design: MapDesignState | null
  movements: MovementState[]
  agents: AgentState[]
  objects: ObjectState[]
  posts: PostState[]
}

export interface ArchivedRunState {
  run_name: string
  output_dir: string
  tick_count: number
  reason: string
  summary_path: string
  summary_url: string
  config_url: string
  charts: { name: string; url: string }[]
  error: string
}

export interface AgentHistoryPoint {
  tick: number
  opinion: number
  satiety: number
  relax: number
  money: number
  satiety_urgency: number
  relax_urgency: number
  satiety_pressure: number
  relax_pressure: number
  emotion: string
}

const MAX_HISTORY = 200

interface SimStore {
  worldState: WorldState | null
  running: boolean
  speed: number
  scenarioName: string
  scenarios: string[]
  archivedRun: ArchivedRunState | null
  selectedAgentId: string | null
  selectedObjectId: string | null   // 当前选中的场景物品 ID
  showMapRegions: boolean           // 是否显示地图区域划分
  ws: WebSocket | null
  connected: boolean
  agentHistory: Record<string, AgentHistoryPoint[]>
  setWorldState: (state: WorldState) => void
  setStatus: (running: boolean, speed: number, scenarioName?: string, scenarios?: string[], archivedRun?: ArchivedRunState | null) => void
  selectAgent: (id: string | null) => void
  selectObject: (id: string | null) => void  // 选中物品，同时取消智能体选中
  toggleMapRegions: () => void
  setWs: (ws: WebSocket | null) => void
  setConnected: (connected: boolean) => void
  sendCmd: (cmd: object) => void
  resetHistory: () => void
}

export const useSimStore = create<SimStore>((set, get) => ({
  worldState: null,
  running: false,
  speed: 1.0,
  scenarioName: 'default_town',
  scenarios: ['default_town', 'jiang_ping_polarization'],
  archivedRun: null,
  selectedAgentId: null,
  selectedObjectId: null,
  showMapRegions: false,
  ws: null,
  connected: false,
  agentHistory: {},

  setWorldState: (worldState) => {
    const prev = get().agentHistory
    const next: Record<string, AgentHistoryPoint[]> = { ...prev }
    for (const a of worldState.agents) {
      // 趋势图只保留最近 MAX_HISTORY 个点，避免长时间运行后前端状态无限增长。
      const point: AgentHistoryPoint = {
        tick: worldState.time,
        opinion: a.opinion,
        satiety: a.satisfaction.satiety ?? 0,
        relax: a.satisfaction.relax ?? 0,
        money: a.satisfaction.money ?? 0,
        satiety_urgency: a.urgency.satiety ?? 0,
        relax_urgency: a.urgency.relax ?? 0,
        satiety_pressure: a.effective_pressure?.satiety ?? 0,
        relax_pressure: a.effective_pressure?.relax ?? 0,
        emotion: a.emotion,
      }
      const arr = prev[a.id] ?? []
      const updated = [...arr, point]
      next[a.id] = updated.length > MAX_HISTORY ? updated.slice(-MAX_HISTORY) : updated
    }
    set({ worldState, scenarioName: worldState.scenario_name, agentHistory: next })
  },

  setStatus: (running, speed, scenarioName, scenarios, archivedRun) => set((state) => ({
    running,
    speed,
    scenarioName: scenarioName ?? state.scenarioName,
    scenarios: scenarios ?? state.scenarios,
    archivedRun: archivedRun === undefined ? state.archivedRun : archivedRun,
  })),
  // 选中智能体时清除物品选中，保持互斥
  selectAgent: (id) => set({ selectedAgentId: id, selectedObjectId: null }),
  // 选中物品时清除智能体选中，保持互斥
  selectObject: (id) => set({ selectedObjectId: id, selectedAgentId: null }),
  toggleMapRegions: () => set((state) => ({ showMapRegions: !state.showMapRegions })),
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
