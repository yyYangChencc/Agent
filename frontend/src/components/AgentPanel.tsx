import { useState } from 'react'
import { useSimStore, AgentState, ObjectState } from '../store/simStore'

// 物品种类元数据：按 objects.py 中 self.kind 值为键，提供前端展示信息
const OBJECT_META: Record<string, { label: string; desc: string; func: string }> = {
  food: {
    label: '食物',
    desc: '地图上可供采集的食物资源，智能体移动至相邻格后可执行 eat 动作消耗。',
    func: '每次 eat 动作使智能体 satiety（饱腹度）need 值增加；num 降至 0 时资源耗尽，方块隐藏。',
  },
  objects: {
    label: '通用物品',
    desc: '场景中的基础物品。',
    func: '具体功能由子类型决定。',
  },
}

// need 值进度条：显示当前值、满足阈值线、急迫度
// threshold 用白色竖线标注，未达标时进度条变红，提供直观警示
function NeedBar({
  label,
  value,
  threshold,
  demand,
}: {
  label: string
  value: number
  threshold: number
  demand: number
}) {
  const pct = Math.round(value * 100)
  const thPct = Math.round(threshold * 100)
  // need 超过 threshold 才算任务完成条件（见 reflect.py task_reset）
  const satisfied = value >= threshold

  return (
    <div className="mb-2">
      <div className="flex justify-between text-xs text-gray-400 mb-0.5">
        <span>{label}</span>
        <span>{pct}%</span>
      </div>
      <div className="relative h-3 bg-gray-700 rounded overflow-visible">
        {/* need 当前值：满足绿色，未满足红色 */}
        <div
          className={`h-full rounded transition-all ${satisfied ? 'bg-green-500' : 'bg-red-500'}`}
          style={{ width: `${pct}%` }}
        />
        {/* 阈值竖线：对应 agent.demand_threshold，超过此线任务视为完成 */}
        <div
          className="absolute top-0 h-full w-0.5 bg-white opacity-70"
          style={{ left: `${thPct}%` }}
        />
      </div>
      {/* demand 为主观急迫度（1=极度渴望，0=已满足），与 need 方向相反 */}
      <div className="text-xs text-gray-500 mt-0.5">urgency {(demand * 100).toFixed(0)}%</div>
    </div>
  )
}

// 点击物品后展开的详情面板，显示种类、描述、功能、位置、数量
function ObjectDetail({ object }: { object: ObjectState }) {
  const meta = OBJECT_META[object.kind] ?? { label: object.kind, desc: '', func: '' }
  const numText =
    object.num === null
      ? '无限'
      : object.num <= 0
      ? '已耗尽'
      : String(object.num)
  const numColor =
    object.num === null
      ? 'text-gray-400'
      : object.num <= 0
      ? 'text-red-400'
      : 'text-green-400'

  return (
    <div className="p-3 space-y-2">
      <div className="text-sm font-semibold text-gray-200">{object.id}</div>
      <div className="text-xs text-gray-400">
        种类：<span className="text-yellow-300">{meta.label}</span>
      </div>
      {meta.desc && (
        <div className="text-xs text-gray-400 leading-relaxed">{meta.desc}</div>
      )}
      {meta.func && (
        <div className="text-xs text-blue-300 leading-relaxed">{meta.func}</div>
      )}
      <div className="text-xs text-gray-400">
        位置：<span className="text-gray-200 font-mono">({object.pos[0]}, {object.pos[1]})</span>
      </div>
      <div className="text-xs text-gray-400">
        剩余数量：<span className={`font-semibold ${numColor}`}>{numText}</span>
      </div>
    </div>
  )
}

// 点击智能体后展开的详情面板，包含 need/demand 仪表盘和思考内容
function AgentDetail({ agent }: { agent: AgentState }) {
  // last_think 内容可能很长，默认折叠
  const [thinkOpen, setThinkOpen] = useState(false)

  return (
    <div className="p-3 space-y-2">
      <div className="text-sm font-semibold text-gray-200">{agent.id}</div>
      <div className="text-xs text-gray-400">
        role: <span className="text-gray-200">{agent.role}</span>
        &nbsp;|&nbsp;emotion: <span className="text-gray-200">{agent.emotion}</span>
      </div>
      <div className="text-xs text-gray-400">
        task: <span className="text-yellow-400">{agent.task}</span>
      </div>
      {/* current_focus 由 micro-reflect 动态更新，反映智能体当前卡顿后的策略调整 */}
      {agent.current_focus && (
        <div className="text-xs text-blue-300 italic">{agent.current_focus}</div>
      )}

      <NeedBar
        label="satiety"
        value={agent.need.satiety ?? 0}
        threshold={agent.demand_threshold.satiety ?? 0.3}
        demand={agent.demand.satiety ?? 1}
      />
      <NeedBar
        label="relax"
        value={agent.need.relax ?? 0}
        threshold={agent.demand_threshold.relax ?? 0.3}
        demand={agent.demand.relax ?? 1}
      />

      {/* opinion 范围 0~1，越高代表越支持正向立场 */}
      <div className="text-xs text-gray-400">opinion: {agent.opinion.toFixed(3)}</div>

      {/* last_think 从 agent.history 中提取最近一次 <Think> 标签内容 */}
      {agent.last_think && (
        <div>
          <button
            className="text-xs text-gray-500 hover:text-gray-300 underline"
            onClick={() => setThinkOpen((v) => !v)}
          >
            {thinkOpen ? '收起思考' : '展开思考'}
          </button>
          {thinkOpen && (
            <div className="mt-1 text-xs text-gray-400 bg-gray-800 rounded p-2 whitespace-pre-wrap max-h-32 overflow-y-auto">
              {agent.last_think}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// 列表行：紧凑展示智能体 ID、当前任务、satiety/relax 迷你进度条
function AgentRow({ agent, selected, onClick }: { agent: AgentState; selected: boolean; onClick: () => void }) {
  const satiety = agent.need.satiety ?? 0
  const relax = agent.need.relax ?? 0

  return (
    <div
      onClick={onClick}
      className={`cursor-pointer px-3 py-2 border-b border-gray-700 hover:bg-gray-700 transition-colors ${
        selected ? 'bg-gray-700' : ''
      }`}
    >
      <div className="flex items-center justify-between">
        <span className="text-xs font-mono text-gray-200">{agent.id}</span>
        {/* 任务文本可能较长，截断显示 */}
        <span className="text-xs text-gray-500 truncate max-w-[80px]">{agent.task}</span>
      </div>
      {/* 橙色=饱腹度，蓝色=放松度，迷你条便于一眼对比多智能体状态 */}
      <div className="flex gap-2 mt-1">
        <div className="flex-1 h-1.5 bg-gray-600 rounded overflow-hidden">
          <div className="h-full bg-orange-400 rounded" style={{ width: `${satiety * 100}%` }} />
        </div>
        <div className="flex-1 h-1.5 bg-gray-600 rounded overflow-hidden">
          <div className="h-full bg-blue-400 rounded" style={{ width: `${relax * 100}%` }} />
        </div>
      </div>
    </div>
  )
}

// 右侧智能体面板：上半部分为列表，下半部分为选中智能体或物品详情
export function AgentPanel() {
  const worldState = useSimStore((s) => s.worldState)
  const selectedAgentId = useSimStore((s) => s.selectedAgentId)
  const selectedObjectId = useSimStore((s) => s.selectedObjectId)
  const selectAgent = useSimStore((s) => s.selectAgent)

  const agents = worldState?.agents ?? []
  const selected = agents.find((a) => a.id === selectedAgentId) ?? null
  // 物品选中：从世界状态中找到对应物品
  const selectedObject = worldState?.objects.find((o) => o.id === selectedObjectId) ?? null

  return (
    <div className="flex flex-col w-64 flex-shrink-0 bg-gray-800 border-l border-gray-700 overflow-hidden">
      <div className="px-3 py-2 border-b border-gray-700 text-xs font-semibold text-gray-400 uppercase tracking-wider">
        Agents
      </div>
      {/* 列表区域可滚动，支持大量智能体 */}
      <div className="flex-1 overflow-y-auto">
        {agents.map((a) => (
          <AgentRow
            key={a.id}
            agent={a}
            selected={a.id === selectedAgentId}
            // 再次点击已选中项则取消选中
            onClick={() => selectAgent(a.id === selectedAgentId ? null : a.id)}
          />
        ))}
      </div>
      {/* 详情区域固定在面板底部，智能体和物品互斥显示 */}
      {selected && (
        <div className="border-t border-gray-700 overflow-y-auto max-h-80">
          <AgentDetail agent={selected} />
        </div>
      )}
      {!selected && selectedObject && (
        <div className="border-t border-gray-700 overflow-y-auto max-h-80">
          <ObjectDetail object={selectedObject} />
        </div>
      )}
    </div>
  )
}
