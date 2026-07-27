import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import {
  AgentLLMDebugCall,
  AgentLLMDebugResponse,
  LLMTraceStatus,
  fetchAgentLLMDebug,
} from '../api/agentLLMDebug'

interface AgentLLMDebugModalProps {
  agentId: string
  displayTick: number
  expectedCallCount: number
  onClose: () => void
}

const STAGE_LABELS: Record<string, string> = {
  world_decision_initial: '行动决策（首次）',
  world_decision_after_memory: '行动决策（记忆后）',
  world_decision: '行动决策',
  memory_planning_initial: '记忆规划（首次）',
  memory_planning_after_memory: '记忆规划（记忆后）',
  memory_planning: '记忆规划',
  social_decision_initial: '社交决策（首次）',
  social_decision_after_memory: '社交决策（记忆后）',
  social_decision: '社交决策',
  conversation_decision_initial: '对话决策（首次）',
  conversation_decision_after_memory: '对话决策（记忆后）',
  conversation_decision: '对话决策',
  task_selection: '任务选择',
  micro_reflection: '任务反思',
  trajectory_summary: '任务轨迹总结',
  short_term_memory_summary: '短期记忆总结',
  psychological_assessment: '心理评测',
  opinion_honest_belief: '观念评测',
  opinion_flan_scoring: 'FLAN 评分',
  opinion_voting: '观念投票',
  person_profile_summary: '人物档案总结',
}

const STATUS_LABELS: Record<LLMTraceStatus, string> = {
  completed: '完成',
  empty_response: '空响应',
  failed: '异常',
  invalid_response: '校验失败',
}

const STATUS_STYLES: Record<LLMTraceStatus, string> = {
  completed: 'border-[#3f6b4f] bg-[#d9ead4] text-[#254a31]',
  empty_response: 'border-[#a06a28] bg-[#f5dfad] text-[#744716]',
  failed: 'border-[#9d4545] bg-[#f0caca] text-[#762f2f]',
  invalid_response: 'border-[#a06a28] bg-[#f5dfad] text-[#744716]',
}

function stageLabel(stage: string): string {
  return STAGE_LABELS[stage] ?? stage
}

function prettyJSON(value: unknown): string {
  return value === null ? 'null' : JSON.stringify(value, null, 2)
}

function PromptSection({ title, value }: { title: string; value: string }) {
  return (
    <section className="min-w-0 border-b border-[#657054] last:border-b-0">
      <h3 className="border-b border-[#657054] bg-[#c9d39f] px-3 py-2 text-xs font-bold text-[#243225]">{title}</h3>
      <pre className="m-0 min-h-20 overflow-x-auto whitespace-pre-wrap break-words bg-[#f8f5d8] p-3 font-mono text-xs leading-5 text-[#25251c]">
        {value || '（空）'}
      </pre>
    </section>
  )
}

function CallListItem({ call, selected, onSelect }: {
  call: AgentLLMDebugCall
  selected: boolean
  onSelect: () => void
}) {
  return (
    <button
      type="button"
      onClick={onSelect}
      className={`w-full border-b border-[#9ca879] px-3 py-2 text-left transition-colors ${
        selected ? 'bg-[#c9d39f]' : 'bg-[#eef0cf] hover:bg-[#f8f5d8]'
      }`}
    >
      <div className="flex min-w-0 items-start justify-between gap-2">
        <span className="min-w-0 break-words text-xs font-bold text-[#243225]">{stageLabel(call.stage)}</span>
        <span className={`flex-shrink-0 rounded border px-1.5 py-0.5 text-[10px] font-semibold ${STATUS_STYLES[call.status]}`}>
          {STATUS_LABELS[call.status]}
        </span>
      </div>
      <div className="mt-1 break-all font-mono text-[10px] text-[#6c584c]">{call.stage}</div>
      <div className="mt-1 flex flex-wrap gap-x-2 text-[10px] text-[#6c584c]">
        <span>{call.call_id}</span>
        <span>尝试 {call.attempt}</span>
        <span>{call.duration_ms.toFixed(1)} ms</span>
      </div>
    </button>
  )
}

function CallDetail({ call }: { call: AgentLLMDebugCall }) {
  const hasError = Boolean(call.error_type || call.error || call.validation_error)

  return (
    <div className="min-w-0">
      <div className="border-b border-[#657054] bg-[#eef0cf] px-3 py-2">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-sm font-bold text-[#243225]">{stageLabel(call.stage)}</span>
          <span className={`rounded border px-1.5 py-0.5 text-[10px] font-semibold ${STATUS_STYLES[call.status]}`}>
            {STATUS_LABELS[call.status]}
          </span>
        </div>
        <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1 font-mono text-[11px] text-[#6c584c]">
          <span>stage={call.stage}</span>
          <span>call_id={call.call_id}</span>
          <span>attempt={call.attempt}</span>
          <span>sequence={call.sequence}</span>
          <span>duration_ms={call.duration_ms.toFixed(3)}</span>
        </div>
      </div>

      {hasError && (
        <section className="border-b border-[#9d4545] bg-[#f0caca] px-3 py-2 text-xs text-[#762f2f]">
          <div className="font-bold">错误信息</div>
          {call.error_type && <div className="mt-1 font-mono">类型：{call.error_type}</div>}
          {call.error && <div className="mt-1 whitespace-pre-wrap break-words">异常：{call.error}</div>}
          {call.validation_error && <div className="mt-1 whitespace-pre-wrap break-words">校验：{call.validation_error}</div>}
        </section>
      )}

      <PromptSection title="System Prompt" value={call.system_prompt} />
      <PromptSection title="User Prompt" value={call.user_prompt} />
      <PromptSection title="LLM 原始响应" value={call.response} />
      <PromptSection title="响应格式" value={prettyJSON(call.response_format)} />
      <PromptSection title="调用元数据" value={prettyJSON(call.metadata)} />
    </div>
  )
}

export function AgentLLMDebugModal({
  agentId,
  displayTick,
  expectedCallCount,
  onClose,
}: AgentLLMDebugModalProps) {
  const [data, setData] = useState<AgentLLMDebugResponse | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const [selectedSequence, setSelectedSequence] = useState<number | null>(null)
  const requestRef = useRef<AbortController | null>(null)

  const loadCalls = useCallback(() => {
    requestRef.current?.abort()
    const controller = new AbortController()
    requestRef.current = controller
    setError('')
    setLoading(true)

    void fetchAgentLLMDebug(agentId, displayTick, controller.signal)
      .then((payload) => {
        if (controller.signal.aborted) return
        setData(payload)
        setSelectedSequence((current) => (
          payload.calls.some((call) => call.sequence === current)
            ? current
            : payload.calls[0]?.sequence ?? null
        ))
      })
      .catch((requestError: unknown) => {
        if (controller.signal.aborted) return
        setError(requestError instanceof Error ? requestError.message : '未知错误')
      })
      .finally(() => {
        if (!controller.signal.aborted && requestRef.current === controller) setLoading(false)
      })
  }, [agentId, displayTick])

  useEffect(() => {
    setData(null)
    setSelectedSequence(null)
    loadCalls()
    return () => requestRef.current?.abort()
  }, [loadCalls])

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [onClose])

  const selectedCall = useMemo(
    () => data?.calls.find((call) => call.sequence === selectedSequence) ?? data?.calls[0] ?? null,
    [data, selectedSequence],
  )

  const modal = (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-1 sm:p-3"
      onMouseDown={(event) => {
        if (event.currentTarget === event.target) onClose()
      }}
    >
      <section
        role="dialog"
        aria-modal="true"
        aria-labelledby="agent-llm-debug-title"
        className="flex h-[96vh] w-[98vw] min-w-0 flex-col overflow-hidden rounded border-2 border-[#25251c] bg-[#dde3c3] shadow-2xl"
      >
        <header className="flex min-w-0 items-center justify-between gap-3 border-b-2 border-[#25251c] bg-[#c9d39f] px-3 py-2 sm:px-4">
          <div className="min-w-0">
            <h2 id="agent-llm-debug-title" className="text-base font-bold text-[#243225] sm:text-lg">LLM 调试</h2>
            <div className="flex flex-wrap gap-x-3 font-mono text-[11px] text-[#6c584c] sm:text-xs">
              <span>{agentId}</span>
              <span>展示 tick {displayTick}</span>
              <span>{data?.count ?? expectedCallCount} 次调用尝试</span>
            </div>
          </div>
          <div className="flex flex-shrink-0 items-center gap-2">
            <button
              type="button"
              aria-label="刷新 LLM 调试记录"
              title="刷新"
              disabled={loading}
              onClick={loadCalls}
              className="flex h-8 w-8 items-center justify-center rounded border border-[#25251c] bg-[#f8f5d8] text-lg text-[#243225] hover:bg-white disabled:cursor-not-allowed disabled:opacity-50"
            >
              ↻
            </button>
            <button
              type="button"
              aria-label="关闭 LLM 调试面板"
              title="关闭"
              onClick={onClose}
              className="flex h-8 w-8 items-center justify-center rounded border border-[#25251c] bg-[#f8f5d8] text-xl leading-none text-[#243225] hover:bg-white"
            >
              ×
            </button>
          </div>
        </header>

        {loading && !data && (
          <div className="flex flex-1 items-center justify-center p-6 text-sm font-semibold text-[#6c584c]" role="status">
            正在加载 LLM 调试记录...
          </div>
        )}

        {!loading && error && (
          <div className="flex flex-1 flex-col items-center justify-center gap-3 p-6 text-center">
            <div className="text-sm font-bold text-[#b94b4b]">加载失败</div>
            <div className="max-w-xl break-words text-xs text-[#6c584c]">{error}</div>
            <button
              type="button"
              onClick={loadCalls}
              className="rounded border border-[#25251c] bg-[#f8f5d8] px-3 py-1.5 text-xs font-semibold text-[#243225] hover:bg-white"
            >
              重试
            </button>
          </div>
        )}

        {!loading && !error && data && data.count === 0 && (
          <div className="flex flex-1 items-center justify-center p-6 text-sm text-[#6c584c]">
            该智能体在 tick {displayTick} 没有 LLM 调用
          </div>
        )}

        {!error && data && data.count > 0 && (
          <div className="flex min-h-0 flex-1 flex-col md:grid md:grid-cols-[310px_minmax(0,1fr)]">
            <nav
              aria-label="LLM 调用列表"
              className="max-h-[34vh] flex-shrink-0 overflow-y-auto border-b border-[#25251c] md:max-h-none md:border-b-0 md:border-r"
            >
              {data.calls.map((call) => (
                <CallListItem
                  key={call.sequence}
                  call={call}
                  selected={call.sequence === selectedCall?.sequence}
                  onSelect={() => setSelectedSequence(call.sequence)}
                />
              ))}
            </nav>
            <main className="min-h-0 min-w-0 flex-1 overflow-y-auto bg-[#f8f5d8]">
              {selectedCall && <CallDetail call={selectedCall} />}
            </main>
          </div>
        )}
      </section>
    </div>
  )

  return createPortal(modal, document.body)
}
