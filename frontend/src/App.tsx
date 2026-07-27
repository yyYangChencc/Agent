import { useWebSocket } from './hooks/useWebSocket'
import { ControlBar } from './components/ControlBar'
import { WorldCanvas } from './components/WorldCanvas'
import { AgentPanel } from './components/AgentPanel'
import { SocialPanel } from './components/SocialPanel'

// useWebSocket 在顶层调用一次，生命周期与页面一致
// 放在 App 而非子组件内，避免路由切换或组件卸载时重连
export default function App() {
  useWebSocket()

  return (
    <div className="flex flex-col h-screen bg-[#20251f] text-[#243225]">
      <ControlBar />
      {/* 只让地图区域双向滚动，智能体与社交面板始终保留在视口内。 */}
      <div className="flex flex-1 min-h-0 overflow-hidden bg-[#141714]">
        <div className="flex-1 min-w-0 overflow-auto">
          <WorldCanvas />
        </div>
        <AgentPanel />
        <SocialPanel />
      </div>
    </div>
  )
}
