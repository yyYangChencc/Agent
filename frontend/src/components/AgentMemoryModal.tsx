import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import {
  AgentMemoryResponse,
  fetchAgentMemories,
  StructuredMemoryGroupKey,
  StructuredMemoryRow,
  VectorMemory,
} from '../api/agentMemory'

type MemoryGroupKey = 'vector_memories' | StructuredMemoryGroupKey
type MemoryRow = VectorMemory | StructuredMemoryRow

interface AgentMemoryModalProps {
  agentId: string
  onClose: () => void
}

interface MemoryGroup {
  key: MemoryGroupKey
  label: string
  source: 'Chroma' | 'SQLite'
}

interface RowPresentation {
  title: string
  body: string
  facts: Array<[string, string]>
  valid: boolean | null
}

const MEMORY_GROUPS: readonly MemoryGroup[] = [
  { key: 'vector_memories', label: '向量记忆', source: 'Chroma' },
  { key: 'events', label: '经历事件', source: 'SQLite' },
  { key: 'entity_states', label: '实体状态', source: 'SQLite' },
  { key: 'person_profiles', label: '人物档案', source: 'SQLite' },
  { key: 'entity_relations', label: '关系状态', source: 'SQLite' },
  { key: 'social_posts', label: '平台暴露', source: 'SQLite' },
  { key: 'scene_snapshots', label: '场景快照', source: 'SQLite' },
  { key: 'derived_memories', label: '派生记忆', source: 'SQLite' },
  { key: 'memory_conflicts', label: '记忆冲突', source: 'SQLite' },
  { key: 'memory_access_log', label: '召回记录', source: 'SQLite' },
]

function valueText(value: unknown): string {
  if (value === null || value === undefined || value === '') return ''
  if (typeof value === 'string') return value
  if (typeof value === 'number' || typeof value === 'boolean') return String(value)
  return JSON.stringify(value)
}

function fieldText(row: StructuredMemoryRow, key: string): string {
  return valueText(row[key])
}

function compactFacts(entries: Array<[string, unknown]>): Array<[string, string]> {
  return entries
    .map(([label, value]) => [label, valueText(value)] as [string, string])
    .filter(([, value]) => value !== '')
}

// 各结构化分组严格使用后端表中的字段生成摘要。
function presentStructuredRow(groupKey: StructuredMemoryGroupKey, row: StructuredMemoryRow): RowPresentation {
  const validValue = row.valid
  const valid = validValue === 1 || validValue === true ? true : validValue === 0 || validValue === false ? false : null

  switch (groupKey) {
    case 'events':
      return {
        title: fieldText(row, 'summary') || fieldText(row, 'event_type') || `事件 ${fieldText(row, 'id')}`,
        body: fieldText(row, 'event_type'),
        facts: compactFacts([
          ['tick', row.world_time],
          ['记忆类型', row.memory_type],
          ['来源', row.source_type],
          ['主题', row.topic],
        ]),
        valid,
      }
    case 'entity_states':
      return {
        title: fieldText(row, 'name') || fieldText(row, 'entity_id'),
        body: fieldText(row, 'entity_type'),
        facts: compactFacts([
          ['最后见到', row.last_seen_at],
          ['区域', row.region_id],
          ['来源', row.source_type],
          ['更新次数', row.update_count],
        ]),
        valid,
      }
    case 'person_profiles':
      return {
        title: fieldText(row, 'name') || fieldText(row, 'target_agent_id'),
        body:
          fieldText(row, 'recent_post_summary') ||
          fieldText(row, 'relationship_impression') ||
          fieldText(row, 'opinion_impression') ||
          fieldText(row, 'actions_impression'),
        facts: compactFacts([
          ['对象', row.target_agent_id],
          ['线下见到', row.last_seen_at],
          ['线上见到', row.last_social_seen_at],
          ['置信度', row.confidence],
        ]),
        valid: null,
      }
    case 'entity_relations':
      return {
        title: `${fieldText(row, 'subject_id')} ${fieldText(row, 'relation_type')} ${fieldText(row, 'object_id')}`.trim(),
        body: '',
        facts: compactFacts([
          ['最后见到', row.last_seen_at],
          ['来源', row.source_type],
          ['重要度', row.importance],
          ['置信度', row.confidence],
        ]),
        valid,
      }
    case 'social_posts':
      return {
        title: fieldText(row, 'content') || fieldText(row, 'post_id'),
        body: fieldText(row, 'topic'),
        facts: compactFacts([
          ['作者', row.author_id],
          ['最后见到', row.last_seen_at],
          ['转发', row.reposts],
          ['评论', row.comments_count],
        ]),
        valid,
      }
    case 'scene_snapshots':
      return {
        title: `场景快照 · tick ${fieldText(row, 'world_time')}`,
        body: '',
        facts: compactFacts([
          ['人物', row.people_count],
          ['物品', row.objects_count],
          ['动作', row.actions_count],
          ['提醒', row.notifications_count],
        ]),
        valid: null,
      }
    case 'derived_memories':
      return {
        title: fieldText(row, 'summary') || fieldText(row, 'memory_id'),
        body: fieldText(row, 'task'),
        facts: compactFacts([
          ['tick', row.world_time],
          ['记忆类型', row.memory_type],
          ['来源', row.source_type],
          ['访问次数', row.access_count],
        ]),
        valid,
      }
    case 'memory_conflicts':
      return {
        title: fieldText(row, 'conflict_type') || `冲突 ${fieldText(row, 'id')}`,
        body: fieldText(row, 'resolution'),
        facts: compactFacts([
          ['对象', row.subject_id],
          ['tick', row.world_time],
          ['已解决', row.resolved],
          ['置信度', row.confidence],
        ]),
        valid: null,
      }
    case 'memory_access_log':
      return {
        title: fieldText(row, 'context') || `召回 ${fieldText(row, 'id')}`,
        body: fieldText(row, 'query_text'),
        facts: compactFacts([
          ['tick', row.world_time],
          ['记忆种类', row.memory_kind],
          ['记忆引用', row.memory_ref],
        ]),
        valid: null,
      }
  }
}

function MemoryCard({ groupKey, row }: { groupKey: MemoryGroupKey; row: MemoryRow }) {
  const presentation: RowPresentation =
    groupKey === 'vector_memories'
      ? {
          title: (row as VectorMemory).document || (row as VectorMemory).id,
          body: '',
          facts: compactFacts([
            ['ID', (row as VectorMemory).id],
            ['记忆类型', (row as VectorMemory).metadata.memory_type],
            ['来源', (row as VectorMemory).metadata.source_type],
            ['保存时间', (row as VectorMemory).metadata.saved_at],
          ]),
          valid: null,
        }
      : presentStructuredRow(groupKey, row as StructuredMemoryRow)

  return (
    <article className="rounded border border-[#9ca879] bg-[#f8f5d8] p-3 shadow-sm">
      <div className="flex min-w-0 items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="max-h-24 overflow-hidden whitespace-pre-wrap break-words text-sm font-semibold leading-5 text-[#243225]">
            {presentation.title || '无摘要'}
          </div>
          {presentation.body && presentation.body !== presentation.title && (
            <div className="mt-1 max-h-16 overflow-hidden whitespace-pre-wrap break-words text-xs leading-5 text-[#6c584c]">
              {presentation.body}
            </div>
          )}
        </div>
        {presentation.valid !== null && (
          <span
            className={`flex-shrink-0 rounded border px-1.5 py-0.5 text-[11px] font-semibold ${
              presentation.valid
                ? 'border-[#62813f] bg-[#e1ebc2] text-[#3e5f25]'
                : 'border-[#b94b4b] bg-[#f6dddd] text-[#8c3030]'
            }`}
          >
            {presentation.valid ? '有效' : '失效'}
          </span>
        )}
      </div>

      {presentation.facts.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1.5">
          {presentation.facts.map(([label, value]) => (
            <span key={`${label}:${value}`} className="rounded border border-[#c9d39f] bg-[#eef0cf] px-1.5 py-0.5 text-[11px] text-[#4d5b39]">
              {label}: {value}
            </span>
          ))}
        </div>
      )}

      <details className="mt-2 border-t border-[#c9d39f] pt-2">
        <summary className="cursor-pointer select-none text-xs font-semibold text-[#4f6f84] hover:text-[#31576e]">
          查看完整 JSON
        </summary>
        <pre className="mt-2 max-h-80 overflow-auto whitespace-pre-wrap break-words rounded bg-[#20251f] p-3 text-[11px] leading-5 text-[#e8ecd5]">
          {JSON.stringify(row, null, 2)}
        </pre>
      </details>
    </article>
  )
}

function rowsForGroup(data: AgentMemoryResponse, groupKey: MemoryGroupKey): MemoryRow[] {
  if (groupKey === 'vector_memories') return data.memories
  return data.structured[groupKey] ?? []
}

function countForGroup(data: AgentMemoryResponse, groupKey: MemoryGroupKey): number {
  if (groupKey === 'vector_memories') return data.count
  return data.structured_counts[groupKey] ?? (data.structured[groupKey]?.length ?? 0)
}

export function AgentMemoryModal({ agentId, onClose }: AgentMemoryModalProps) {
  const [data, setData] = useState<AgentMemoryResponse | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const [activeGroup, setActiveGroup] = useState<MemoryGroupKey>('vector_memories')
  const [searchText, setSearchText] = useState('')
  const requestRef = useRef<AbortController | null>(null)

  const loadMemories = useCallback(() => {
    requestRef.current?.abort()
    const controller = new AbortController()
    requestRef.current = controller
    setData(null)
    setError('')
    setLoading(true)

    void fetchAgentMemories(agentId, controller.signal)
      .then((payload) => {
        if (!controller.signal.aborted) setData(payload)
      })
      .catch((requestError: unknown) => {
        if (controller.signal.aborted) return
        setError(requestError instanceof Error ? requestError.message : '未知错误')
      })
      .finally(() => {
        if (!controller.signal.aborted && requestRef.current === controller) setLoading(false)
      })
  }, [agentId])

  useEffect(() => {
    loadMemories()
    return () => requestRef.current?.abort()
  }, [loadMemories])

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [onClose])

  const activeRows = useMemo(() => (data ? rowsForGroup(data, activeGroup) : []), [activeGroup, data])
  const normalizedSearch = searchText.trim().toLocaleLowerCase()
  const visibleRows = useMemo(() => {
    if (!normalizedSearch) return activeRows
    return activeRows.filter((row) => JSON.stringify(row).toLocaleLowerCase().includes(normalizedSearch))
  }, [activeRows, normalizedSearch])
  const activeGroupMeta = MEMORY_GROUPS.find((group) => group.key === activeGroup) ?? MEMORY_GROUPS[0]

  const modal = (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-2 sm:p-4"
      onMouseDown={(event) => {
        if (event.currentTarget === event.target) onClose()
      }}
    >
      <section
        role="dialog"
        aria-modal="true"
        aria-labelledby="agent-memory-title"
        className="flex h-[min(92vh,800px)] w-[min(96vw,1120px)] min-w-0 flex-col overflow-hidden rounded border-2 border-[#25251c] bg-[#dde3c3] shadow-2xl"
      >
        <header className="flex min-w-0 items-center justify-between gap-3 border-b-2 border-[#25251c] bg-[#c9d39f] px-3 py-2 sm:px-4">
          <div className="min-w-0">
            <h2 id="agent-memory-title" className="truncate text-base font-bold text-[#243225] sm:text-lg">
              全部记忆
            </h2>
            <div className="truncate font-mono text-xs text-[#6c584c]">{agentId}</div>
          </div>
          <div className="flex flex-shrink-0 items-center gap-2">
            <button
              type="button"
              title="刷新记忆"
              disabled={loading}
              onClick={loadMemories}
              className="rounded border border-[#25251c] bg-[#f8f5d8] px-2.5 py-1.5 text-xs font-semibold text-[#243225] hover:bg-white disabled:cursor-not-allowed disabled:opacity-50"
            >
              刷新
            </button>
            <button
              type="button"
              aria-label="关闭记忆面板"
              title="关闭"
              onClick={onClose}
              className="flex h-8 w-8 items-center justify-center rounded border border-[#25251c] bg-[#f8f5d8] text-xl leading-none text-[#243225] hover:bg-white"
            >
              ×
            </button>
          </div>
        </header>

        {loading && (
          <div className="flex flex-1 items-center justify-center p-6 text-sm font-semibold text-[#6c584c]" role="status">
            正在加载记忆...
          </div>
        )}

        {!loading && error && (
          <div className="flex flex-1 flex-col items-center justify-center gap-3 p-6 text-center">
            <div className="text-sm font-bold text-[#b94b4b]">加载失败</div>
            <div className="max-w-xl break-words text-xs text-[#6c584c]">{error}</div>
            <button
              type="button"
              onClick={loadMemories}
              className="rounded border border-[#25251c] bg-[#f8f5d8] px-3 py-1.5 text-xs font-semibold text-[#243225] hover:bg-white"
            >
              重试
            </button>
          </div>
        )}

        {!loading && !error && data && data.total_count === 0 && (
          <div className="flex flex-1 items-center justify-center p-6 text-sm text-[#6c584c]">该智能体暂无记忆</div>
        )}

        {!loading && !error && data && data.total_count > 0 && (
          <div className="flex min-h-0 flex-1 flex-col md:grid md:grid-cols-[220px_minmax(0,1fr)]">
            <nav
              className="flex flex-shrink-0 gap-1 overflow-x-auto border-b border-[#25251c] bg-[#eef0cf] p-2 md:block md:overflow-y-auto md:border-b-0 md:border-r"
              aria-label="记忆分类"
              role="tablist"
            >
              <div className="hidden px-2 pb-2 text-xs font-semibold text-[#6c584c] md:block">
                共 {data.total_count} 条
              </div>
              {MEMORY_GROUPS.map((group) => {
                const count = countForGroup(data, group.key)
                const selected = activeGroup === group.key
                return (
                  <button
                    key={group.key}
                    type="button"
                    role="tab"
                    aria-selected={selected}
                    onClick={() => {
                      setActiveGroup(group.key)
                      setSearchText('')
                    }}
                    className={`mb-0 flex min-w-[132px] flex-shrink-0 items-center justify-between gap-2 rounded border px-2 py-2 text-left text-xs md:mb-1 md:w-full md:min-w-0 ${
                      selected
                        ? 'border-[#25251c] bg-[#c9d39f] font-bold text-[#243225]'
                        : 'border-transparent bg-transparent text-[#4d5b39] hover:border-[#9ca879] hover:bg-[#f8f5d8]'
                    }`}
                  >
                    <span className="min-w-0 truncate">
                      <span className="block truncate">{group.label}</span>
                      <span className="block text-[10px] font-normal opacity-70">{group.source}</span>
                    </span>
                    <span className="flex-shrink-0 rounded bg-[#f8f5d8] px-1.5 py-0.5 font-mono text-[11px]">{count}</span>
                  </button>
                )
              })}
            </nav>

            <div className="flex min-h-0 min-w-0 flex-col bg-[#dde3c3]">
              <div className="flex flex-col gap-2 border-b border-[#9ca879] px-3 py-2 sm:flex-row sm:items-center sm:justify-between">
                <div className="min-w-0 text-sm font-bold text-[#243225]">
                  {activeGroupMeta.label}
                  <span className="ml-2 font-mono text-xs font-normal text-[#6c584c]">
                    {visibleRows.length === activeRows.length ? `${activeRows.length} 条` : `${visibleRows.length} / ${activeRows.length} 条`}
                  </span>
                </div>
                <input
                  type="search"
                  value={searchText}
                  onChange={(event) => setSearchText(event.target.value)}
                  placeholder="搜索当前分类"
                  aria-label="搜索当前记忆分类"
                  className="h-8 w-full rounded border border-[#6c584c] bg-[#f8f5d8] px-2 text-xs text-[#243225] outline-none placeholder:text-[#8b8078] focus:border-[#4f8fc0] sm:w-64"
                />
              </div>

              <div className="min-h-0 flex-1 overflow-y-auto p-3">
                {activeRows.length === 0 && (
                  <div className="flex h-full min-h-32 items-center justify-center text-sm text-[#6c584c]">该分类暂无记忆</div>
                )}
                {activeRows.length > 0 && visibleRows.length === 0 && (
                  <div className="flex h-full min-h-32 items-center justify-center text-sm text-[#6c584c]">没有匹配的记忆</div>
                )}
                {visibleRows.length > 0 && (
                  <div className="grid min-w-0 gap-2">
                    {visibleRows.map((row, index) => (
                      <MemoryCard key={`${activeGroup}:${index}`} groupKey={activeGroup} row={row} />
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </section>
    </div>
  )

  return createPortal(modal, document.body)
}
