import { useWebSocket } from './hooks/useWebSocket'
import { ControlBar } from './components/ControlBar'
import { WorldCanvas } from './components/WorldCanvas'
import { AgentPanel } from './components/AgentPanel'

// useWebSocket 在顶层调用一次，生命周期与页面一致
// 放在 App 而非子组件内，避免路由切换或组件卸载时重连
export default function App() {
  useWebSocket()

  return (
    <div className="flex flex-col h-screen bg-gray-900 text-gray-100">
      <ControlBar />
      {/* 画布固定 600px 宽，面板占剩余空间 */}
      <div className="flex flex-1 overflow-hidden">
        <WorldCanvas />
        <AgentPanel />
      </div>
    </div>
  )
}
