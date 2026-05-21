import { useState, useMemo } from 'react'
import { useSimStore, PostState } from '../store/simStore'

const AGENT_COLORS: Record<string, string> = {
  agent_1: 'text-blue-400',
  agent_2: 'text-pink-400',
  agent_3: 'text-green-400',
  agent_4: 'text-yellow-400',
  agent_5: 'text-purple-400',
}

function agentColor(id: string): string {
  return AGENT_COLORS[id] ?? 'text-gray-300'
}

function OpinionBar({ value }: { value: number }) {
  const pct = Math.round(value * 100)
  const color =
    value < 0.35 ? 'bg-red-500' : value > 0.65 ? 'bg-green-500' : 'bg-gray-500'
  return (
    <div className="flex items-center gap-1 mt-0.5">
      <div className="relative h-1.5 w-16 bg-gray-700 rounded overflow-hidden">
        <div className={`h-full rounded ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-gray-500 text-xs">{value.toFixed(2)}</span>
    </div>
  )
}

function PostCard({ post }: { post: PostState }) {
  const [open, setOpen] = useState(false)

  return (
    <div className="border border-gray-700 rounded p-2 space-y-1">
      <div className="flex items-center justify-between">
        <span className={`text-xs font-semibold ${agentColor(post.author_id)}`}>
          {post.author_id}
        </span>
        <span className="text-xs text-gray-500">t={post.time}</span>
      </div>
      <div className="text-xs text-gray-200 leading-relaxed break-words">{post.content}</div>
      <OpinionBar value={post.opinion_index} />
      <div className="flex items-center gap-3 text-xs text-gray-400">
        <span>👍 {post.likes}</span>
        <span>👎 {post.dislikes}</span>
        {post.comments.length > 0 && (
          <button
            className="underline hover:text-gray-200"
            onClick={() => setOpen((v) => !v)}
          >
            💬 {post.comments.length} {open ? '收起' : '展开'}
          </button>
        )}
        {post.comments.length === 0 && <span>💬 0</span>}
      </div>
      {open && post.comments.length > 0 && (
        <div className="mt-1 space-y-1 pl-2 border-l border-gray-600">
          {post.comments.map((c) => (
            <div key={c.id} className="text-xs">
              <span className={`font-semibold ${agentColor(c.author_id)}`}>{c.author_id}</span>
              {c.time !== null && (
                <span className="text-gray-500 ml-1">t={c.time}</span>
              )}
              <span className="text-gray-300 ml-1">{c.content}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

const EMPTY_POSTS: PostState[] = []

export function SocialPanel() {
  const posts = useSimStore((s) => s.worldState?.posts ?? EMPTY_POSTS)
  const sorted = useMemo(() => [...posts].sort((a, b) => b.id - a.id), [posts])

  return (
    <div className="flex flex-col w-72 border-l border-gray-700 bg-gray-900 overflow-hidden">
      <div className="px-3 py-2 border-b border-gray-700 text-sm font-semibold text-gray-300 flex-shrink-0">
        社交平台
        <span className="ml-2 text-xs text-gray-500 font-normal">{posts.length} 条帖子</span>
      </div>
      <div className="flex-1 overflow-y-auto p-2 space-y-2">
        {sorted.length === 0 ? (
          <div className="text-xs text-gray-500 text-center mt-4">暂无帖子</div>
        ) : (
          sorted.map((p) => <PostCard key={p.id} post={p} />)
        )}
      </div>
    </div>
  )
}
