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
      {/* 画布固定 600px 宽，智能体面板和社交面板占剩余空间 */}
      <div className="flex flex-1 overflow-hidden bg-[#141714]">
        <WorldCanvas />
        <AgentPanel />
        <SocialPanel />
      </div>
    </div>
  )
}
