export type LLMTraceStatus = 'completed' | 'empty_response' | 'failed' | 'invalid_response'

export interface AgentLLMDebugCall {
  schema_version: number
  sequence: number
  call_id: string
  agent_id: string
  tick: number
  stage: string
  attempt: number
  system_prompt: string
  user_prompt: string
  response: string
  response_format: Record<string, unknown> | null
  status: LLMTraceStatus
  error_type: string
  error: string
  validation_error: string
  duration_ms: number
  metadata: Record<string, unknown>
}

export interface AgentLLMDebugResponse {
  agent_id: string
  tick: number
  count: number
  calls: AgentLLMDebugCall[]
}

interface ApiErrorPayload {
  error?: unknown
}

// 调试数据只在面板打开时按需请求，不写入全局仿真状态。
export async function fetchAgentLLMDebug(
  agentId: string,
  tick: number,
  signal: AbortSignal,
): Promise<AgentLLMDebugResponse> {
  const params = new URLSearchParams({ tick: String(tick) })
  const response = await fetch(`/api/agents/${encodeURIComponent(agentId)}/llm-debug?${params}`, { signal })
  const payload: unknown = await response.json()

  if (!response.ok) {
    const errorPayload = payload as ApiErrorPayload
    const message = typeof errorPayload.error === 'string' ? errorPayload.error : `HTTP ${response.status}`
    throw new Error(message)
  }

  return payload as AgentLLMDebugResponse
}
