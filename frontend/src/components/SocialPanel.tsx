import { useState, useMemo } from 'react'
import { useSimStore, PostState } from '../store/simStore'

const AGENT_COLORS: Record<string, string> = {
  agent_1: 'text-[#2f6fbd]',
  agent_2: 'text-[#a33a5f]',
  agent_3: 'text-[#2f7d4f]',
  agent_4: 'text-[#8a5a33]',
  agent_5: 'text-[#6b4fa3]',
}

function agentColor(id: string): string {
  return AGENT_COLORS[id] ?? 'text-[#243225]'
}

function OpinionBar({ value }: { value: number }) {
  const clamped = Math.max(-1, Math.min(1, value))
  const pct = Math.abs(clamped) * 50
  const left = clamped >= 0 ? 50 : 50 - pct
  const color =
    clamped < -0.35 ? 'bg-[#b94b4b]' : clamped > 0.35 ? 'bg-[#4f8f4d]' : 'bg-[#8a7356]'
  return (
    <div className="flex items-center gap-1.5 mt-1">
      <div className="relative h-2 w-20 bg-[#d7b36a] border border-[#6c584c] rounded overflow-hidden">
        <div className="absolute left-1/2 top-0 h-full w-px bg-[#25251c]/60" />
        <div
          className={`absolute top-0 h-full rounded ${color}`}
          style={{ left: `${left}%`, width: `${pct}%` }}
        />
      </div>
      <span className="text-[#6c584c] text-xs font-mono">{value.toFixed(2)}</span>
    </div>
  )
}

function PostCard({ post }: { post: PostState }) {
  const [open, setOpen] = useState(false)
  const sourceLabel =
    post.source_type === 'official_news'
      ? '官方新闻'
      : post.source_type === 'influencer'
        ? '投放者'
        : '智能体'
  const sourceClass =
    post.source_type === 'official_news'
      ? 'bg-[#4f8fc0] text-[#f8f5d8]'
      : post.source_type === 'influencer'
        ? 'bg-[#9b5b43] text-[#f8f5d8]'
        : 'bg-[#d7b36a] text-[#243225]'

  return (
    <div className="border-2 border-[#25251c] bg-[#f8f5d8] rounded p-2.5 space-y-1.5 shadow-[2px_2px_0_#25251c]">
      <div className="flex items-center justify-between">
        <span className={`text-xs font-semibold ${agentColor(post.author_id)}`}>
          {post.author_id}
        </span>
        <div className="flex items-center gap-1">
          <span className={`px-1.5 py-0.5 rounded text-[10px] font-semibold ${sourceClass}`}>
            {sourceLabel}
          </span>
          {post.is_rumor && (
            <span className="px-1.5 py-0.5 rounded text-[10px] font-semibold bg-[#b94b4b] text-[#f8f5d8]">
              极化投放
            </span>
          )}
          <span className="text-xs text-[#6c584c] font-mono">t={post.time}</span>
        </div>
      </div>
      {post.topic && <div className="text-[11px] text-[#6c584c]">主题：{post.topic}</div>}
      <div className="text-xs text-[#243225] leading-relaxed break-words">{post.content}</div>
      <OpinionBar value={post.opinion_index} />
      <div className="flex items-center gap-3 text-xs text-[#4b3f2f]">
        <span>👍 {post.likes}</span>
        <span>👎 {post.dislikes}</span>
        {post.comments.length > 0 && (
          <button
            className="underline text-[#2f6fbd] hover:text-[#1f4f8d]"
            onClick={() => setOpen((v) => !v)}
          >
            💬 {post.comments.length} {open ? '收起' : '展开'}
          </button>
        )}
        {post.comments.length === 0 && <span>💬 0</span>}
      </div>
      {open && post.comments.length > 0 && (
        <div className="mt-2 space-y-1.5 pl-2 border-l-2 border-[#d7b36a]">
          {post.comments.map((c) => (
            <div key={c.id} className="text-xs bg-[#efe2b9] border border-[#d7b36a] rounded px-2 py-1">
              <span className={`font-semibold ${agentColor(c.author_id)}`}>{c.author_id}</span>
              {c.time !== null && (
                <span className="text-[#6c584c] ml-1 font-mono">t={c.time}</span>
              )}
              <span className="ml-2 text-[#6c584c]">认同原帖</span>
              <OpinionBar value={c.agreement_to_post} />
              <span className="text-[#243225] ml-1">{c.content}</span>
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
    <div className="flex flex-col w-72 border-l-4 border-[#25251c] bg-[#dde3c3] text-[#243225] overflow-hidden">
      <div className="px-3 py-2 border-b-4 border-[#25251c] bg-[#c9d39f] text-sm font-extrabold text-[#243225] flex-shrink-0">
        社交平台
        <span className="ml-2 text-xs text-[#6c584c] font-normal">{posts.length} 条帖子</span>
      </div>
      <div className="flex-1 overflow-y-auto p-3 space-y-3">
        {sorted.length === 0 ? (
          <div className="text-xs text-[#6c584c] text-center mt-4">暂无帖子</div>
        ) : (
          sorted.map((p) => <PostCard key={p.id} post={p} />)
        )}
      </div>
    </div>
  )
}
