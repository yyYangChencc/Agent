export const STRUCTURED_MEMORY_GROUP_KEYS = [
  'events',
  'entity_states',
  'person_profiles',
  'entity_relations',
  'social_posts',
  'scene_snapshots',
  'derived_memories',
  'memory_conflicts',
  'memory_access_log',
] as const

export type StructuredMemoryGroupKey = (typeof STRUCTURED_MEMORY_GROUP_KEYS)[number]

export type StructuredMemoryRow = Record<string, unknown>

export interface VectorMemory {
  id: string
  document: string
  metadata: Record<string, unknown>
}

export interface AgentMemoryResponse {
  agent_id: string
  count: number
  memories: VectorMemory[]
  structured: Partial<Record<StructuredMemoryGroupKey, StructuredMemoryRow[]>>
  structured_counts: Partial<Record<StructuredMemoryGroupKey, number>>
  structured_count: number
  total_count: number
}

interface ApiErrorPayload {
  error?: unknown
}

// 单独封装 HTTP 请求，避免把一次性面板数据写入仿真状态仓库。
export async function fetchAgentMemories(
  agentId: string,
  signal: AbortSignal,
): Promise<AgentMemoryResponse> {
  const response = await fetch(`/api/agents/${encodeURIComponent(agentId)}/memories`, { signal })
  const payload: unknown = await response.json()

  if (!response.ok) {
    const errorPayload = payload as ApiErrorPayload
    const message = typeof errorPayload.error === 'string' ? errorPayload.error : `HTTP ${response.status}`
    throw new Error(message)
  }

  return payload as AgentMemoryResponse
}
