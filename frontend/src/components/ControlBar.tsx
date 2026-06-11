import { useSimStore } from '../store/simStore'

// 支持的仿真速度倍率，对应后端 sim_state["speed"]
const SPEEDS = [1, 2, 5]

// 顶部控制栏：仿真控制按钮 + 速度选择 + 状态指示
export function ControlBar() {
  const running = useSimStore((s) => s.running)
  const speed = useSimStore((s) => s.speed)
  const connected = useSimStore((s) => s.connected)
  const worldState = useSimStore((s) => s.worldState)
  const showMapRegions = useSimStore((s) => s.showMapRegions)
  const toggleMapRegions = useSimStore((s) => s.toggleMapRegions)
  const sendCmd = useSimStore((s) => s.sendCmd)

  // 未收到任何 tick 时显示 0
  const tick = worldState?.time ?? 0

  return (
    <div className="flex items-center gap-3 px-4 py-2 bg-[#dde3c3] border-b-4 border-[#25251c] text-sm flex-shrink-0 text-[#243225]">
      <span className="font-extrabold text-[#243225]">Agent Sim</span>
      <span className="text-[#6c584c]">|</span>
      {/* 当前仿真时刻，与后端 world.time 同步 */}
      <span className="text-[#3f4f37] font-mono">t={tick}</span>
      {/* 连接状态指示灯：绿色=已连接，红色=断开 */}
      <span
        className={`w-2 h-2 rounded-full ${connected ? 'bg-green-400' : 'bg-red-500'}`}
        title={connected ? '已连接' : '未连接'}
      />

      {/* 弹性空白，将操作按钮推到右侧 */}
      <div className="flex-1" />

      <button
        onClick={toggleMapRegions}
        className={`px-3 py-1 rounded text-xs ${
          showMapRegions
            ? 'bg-[#f8c86b] hover:bg-[#ffd77f] text-[#243225]'
            : 'bg-[#f8f5d8] hover:bg-[#f8c86b] text-[#243225]'
        }`}
      >
        {showMapRegions ? '隐藏区域' : '显示区域'}
      </button>

      {/* 单步：仅在暂停状态可用，每次触发后端执行一个 world.step() */}
      <button
        onClick={() => sendCmd({ cmd: 'step' })}
        disabled={running || !connected}
        className="px-3 py-1 rounded bg-[#f8f5d8] hover:bg-[#f8c86b] disabled:opacity-40 disabled:cursor-not-allowed text-[#243225] text-xs"
      >
        单步
      </button>

      {/* 继续/暂停：切换自动步进循环，颜色随状态变化提供反馈 */}
      <button
        onClick={() => sendCmd({ cmd: running ? 'pause' : 'resume' })}
        disabled={!connected}
        className={`px-3 py-1 rounded text-xs disabled:opacity-40 disabled:cursor-not-allowed ${
          running
            ? 'bg-[#f8c86b] hover:bg-[#ffd77f] text-[#243225]'
            : 'bg-[#6aa96f] hover:bg-[#7fbf84] text-[#f8f5d8]'
        }`}
      >
        {running ? '暂停' : '继续'}
      </button>

      {/* 重置：后端重建整个 runtime，t 归零，智能体和记忆全部重新初始化 */}
      <button
        onClick={() => sendCmd({ cmd: 'reset' })}
        disabled={!connected}
        className="px-3 py-1 rounded bg-[#9b5b43] hover:bg-[#b66b4f] text-[#f8f5d8] text-xs disabled:opacity-40 disabled:cursor-not-allowed"
      >
        重置
      </button>

      <span className="text-[#6c584c]">|</span>

      {/* 速度按钮：当前倍率高亮，点击后后端调整 asyncio.sleep 间隔 */}
      {SPEEDS.map((s) => (
        <button
          key={s}
          onClick={() => sendCmd({ cmd: 'set_speed', value: s })}
          disabled={!connected}
          className={`px-2 py-1 rounded text-xs disabled:opacity-40 disabled:cursor-not-allowed ${
            speed === s
              ? 'bg-[#4f8fc0] text-[#f8f5d8]'
              : 'bg-[#f8f5d8] hover:bg-[#f8c86b] text-[#243225]'
          }`}
        >
          {s}×
        </button>
      ))}
    </div>
  )
}
