import { useEffect, useRef } from 'react'
import { useSimStore } from '../store/simStore'

const RECONNECT_DELAY = 2000

export function useWebSocket() {
  const { setWorldState, setStatus, setWs, setConnected, resetHistory } = useSimStore()
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const activeRef = useRef(true)

  useEffect(() => {
    activeRef.current = true

    function connect() {
      if (!activeRef.current) return
      // 前端和后端同源部署；协议根据当前页面自动选择 ws/wss。
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
          if (msg.type === 'tick' || msg.type === 'init') {
            setWorldState(msg.state)
          }
          if (msg.type === 'init') {
            // init 表示后端给出一份完整初始快照；旧图表历史不再适用。
            resetHistory()
            setStatus(msg.running ?? false, msg.speed ?? 1.0, msg.scenario_name, msg.scenarios, msg.archived_run ?? null)
          } else if (msg.type === 'status') {
            setStatus(msg.running, msg.speed, msg.scenario_name, msg.scenarios, msg.archived_run ?? undefined)
          }
        } catch {
          // 忽略格式错误的消息
        }
      }

      ws.onclose = () => {
        setWs(null)
        setConnected(false)
        if (activeRef.current) {
          timerRef.current = setTimeout(connect, RECONNECT_DELAY)
        }
      }

      ws.onerror = () => {
        setConnected(false)
        // onclose 会在 onerror 后触发，重连逻辑在 onclose 中处理
      }

      setWs(ws)
    }

    connect()

    return () => {
      activeRef.current = false
      if (timerRef.current) clearTimeout(timerRef.current)
      const ws = useSimStore.getState().ws
      if (ws) ws.close()
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])
}
