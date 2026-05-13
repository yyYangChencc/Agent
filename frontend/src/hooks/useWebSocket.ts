import { useEffect } from 'react'
import { useSimStore } from '../store/simStore'

export function useWebSocket() {
  const { setWorldState, setStatus, setWs, setConnected } = useSimStore()

  useEffect(() => {
    // 根据当前页面协议自动选择 ws/wss，生产环境 https 下也能正常连接
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsUrl = `${protocol}//${window.location.host}/ws`
    const ws = new WebSocket(wsUrl)

    ws.onopen = () => {
      setWs(ws)
      setConnected(true)
    }

    ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data)
        // tick 和 init 都携带完整世界快照，统一更新 worldState
        if (msg.type === 'tick' || msg.type === 'init') {
          setWorldState(msg.state)
        }
        // init 同时携带运行状态（新连接时对齐前端显示）
        if (msg.type === 'init') {
          setStatus(msg.running ?? false, msg.speed ?? 1.0)
        } else if (msg.type === 'status') {
          // status 由 resume/pause/set_speed 触发，仅更新控制状态
          setStatus(msg.running, msg.speed)
        }
      } catch {
        // 忽略格式错误的消息，防止单条异常中断整个消息处理
      }
    }

    ws.onclose = () => {
      setWs(null)
      setConnected(false)
    }

    ws.onerror = () => {
      setConnected(false)
    }

    return () => {
      ws.close()
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []) // 仅在挂载时建立连接，setXxx 函数引用稳定无需列入依赖
}
