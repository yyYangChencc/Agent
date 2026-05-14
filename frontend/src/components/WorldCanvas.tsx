import { useEffect, useRef } from 'react'
import * as PIXI from 'pixi.js'
import { useSimStore } from '../store/simStore'

// 每格像素大小，地图为 25×25，画布固定 600px
const CELL = 24
// 智能体圆形颜色池，按 ID 哈希取色，保证同一智能体颜色稳定
const AGENT_COLORS = [0x4f8ef7, 0xe74c6f, 0x2ecc71, 0xf39c12, 0x9b59b6]
// 与后端 observer.py 保持一致的观测半径（格数）
const OBSERVATION_RADIUS = 5

// 将智能体 ID 字符串映射到固定颜色，避免每次渲染随机变色
function agentColor(id: string): number {
  let h = 0
  for (let i = 0; i < id.length; i++) h = (h * 31 + id.charCodeAt(i)) >>> 0
  return AGENT_COLORS[h % AGENT_COLORS.length]
}

export function WorldCanvas() {
  const containerRef = useRef<HTMLDivElement>(null)
  // 用 ref 持有 Pixi app 实例，避免重渲染时重复初始化
  const appRef = useRef<PIXI.Application | null>(null)
  // 按智能体 ID 缓存 Graphics/Text 对象，tick 时只更新坐标，不重建
  const agentGfxRef = useRef<Map<string, { body: PIXI.Graphics; label: PIXI.Text }>>(new Map())
  // 按对象 ID 缓存食物等场景物体的 Graphics
  const objectGfxRef = useRef<Map<string, PIXI.Graphics>>(new Map())
  // 选中高亮层（观测范围圆 + 黄色描边），独立于智能体图层便于整体清除
  const selectionGfxRef = useRef<PIXI.Graphics | null>(null)

  const worldState = useSimStore((s) => s.worldState)
  const selectedAgentId = useSimStore((s) => s.selectedAgentId)
  const selectedObjectId = useSimStore((s) => s.selectedObjectId)
  const selectAgent = useSimStore((s) => s.selectAgent)
  const selectObject = useSimStore((s) => s.selectObject)

  // 仅在组件挂载时初始化 Pixi，销毁时清理，不依赖任何状态
  useEffect(() => {
    const el = containerRef.current
    if (!el) return

    // React StrictMode 会 mount→unmount→mount，用两个标志避免在 init 未完成时
    // 调用 destroy（ResizePlugin 尚未就绪，destroy 会抛 "not a function" 异常）
    let cancelled = false
    let initialized = false
    const app = new PIXI.Application()
    appRef.current = app

    // Pixi v8 采用异步 init，需要 await 后才能操作 canvas
    app.init({
      width: 600,
      height: 600,
      background: 0x1a1a2e,
      antialias: true,
    }).then(() => {
      if (cancelled) {
        // cleanup 先于 init 完成时，在这里安全销毁
        app.destroy(true)
        return
      }
      initialized = true
      el.appendChild(app.canvas)

      // 绘制静态网格，只创建一次，不随 tick 刷新
      const grid = new PIXI.Graphics()
      for (let x = 0; x <= 25; x++) {
        grid.moveTo(x * CELL, 0).lineTo(x * CELL, 600)
      }
      for (let y = 0; y <= 25; y++) {
        grid.moveTo(0, y * CELL).lineTo(600, y * CELL)
      }
      grid.stroke({ color: 0x2a2a4a, width: 1 })
      app.stage.addChild(grid)

      // 选中高亮层在网格之上、智能体图层之下，保证不遮挡标签
      const selGfx = new PIXI.Graphics()
      selectionGfxRef.current = selGfx
      app.stage.addChild(selGfx)
    })

    return () => {
      cancelled = true
      // 只有 init 已完成（插件全部就绪）时才能安全调用 destroy
      if (initialized) {
        app.destroy(true)
      }
      appRef.current = null
      agentGfxRef.current.clear()
      objectGfxRef.current.clear()
    }
  }, [])

  // 每次收到新的世界状态时，更新所有智能体和场景物体的位置与外观
  useEffect(() => {
    const app = appRef.current
    if (!app || !worldState) return

    const stage = app.stage
    const agentGfx = agentGfxRef.current
    const objectGfx = objectGfxRef.current

    // 更新场景物体（食物等）
    const seenObjects = new Set<string>()
    for (const obj of worldState.objects) {
      seenObjects.add(obj.id)
      const sx = obj.pos[0] * CELL
      const sy = obj.pos[1] * CELL

      // 首次出现时创建 Graphics，后续只重绘
      if (!objectGfx.has(obj.id)) {
        const g = new PIXI.Graphics()
        // 物品支持点击选中，与智能体圆圈行为一致
        g.eventMode = 'static'
        g.cursor = 'pointer'
        g.on('pointerdown', () => {
          const { selectedObjectId: curId, selectObject: sel } = useSimStore.getState()
          sel(obj.id === curId ? null : obj.id)
        })
        stage.addChild(g)
        objectGfx.set(obj.id, g)
      }
      const g = objectGfx.get(obj.id)!
      g.clear()
      // num <= 0 表示物品已耗尽，隐藏方块而非移除，保留对象引用
      const hidden = obj.num !== null && obj.num <= 0
      if (!hidden) {
        if (obj.kind === 'building') {
          // 建筑：灰色填充整格，区别于食物的小方块
          g.rect(sx, sy, CELL, CELL).fill(0x6b7280)
        } else {
          // 默认（含 food）：绿色小方块
          g.rect(sx + 6, sy + 6, 12, 12).fill(0x2ecc71)
        }
      }
    }
    // 清除服务端已不存在的物体
    for (const [id, g] of objectGfx) {
      if (!seenObjects.has(id)) {
        stage.removeChild(g)
        g.destroy()
        objectGfx.delete(id)
      }
    }

    // 更新智能体
    const seenAgents = new Set<string>()
    for (const agent of worldState.agents) {
      seenAgents.add(agent.id)
      // pos[0] → X（列），pos[1] → Y（行），与 map.grid 索引一致
      const sx = agent.pos[0] * CELL + CELL / 2
      const sy = agent.pos[1] * CELL + CELL / 2
      const color = agentColor(agent.id)

      if (!agentGfx.has(agent.id)) {
        const body = new PIXI.Graphics()
        // Pixi v8 必须显式设置 eventMode 才能接收指针事件
        body.eventMode = 'static'
        body.cursor = 'pointer'
        body.on('pointerdown', () => {
          // 再次点击已选中智能体则取消选中
          selectAgent(agent.id === useSimStore.getState().selectedAgentId ? null : agent.id)
        })
        const label = new PIXI.Text({ text: agent.id, style: { fontSize: 8, fill: 0xffffff } })
        // anchor(0.5, 1) 使文字底部中心对齐到目标坐标，便于放在圆圈正上方
        label.anchor.set(0.5, 1)
        stage.addChild(body)
        stage.addChild(label)
        agentGfx.set(agent.id, { body, label })
      }

      const { body, label } = agentGfx.get(agent.id)!
      body.clear()
      body.circle(sx, sy, 9).fill(color)
      // 标签放在圆圈上方 11px，留出圆形半径 + 2px 间距
      label.position.set(sx, sy - 11)
    }
    // 清除服务端已不存在的智能体
    for (const [id, { body, label }] of agentGfx) {
      if (!seenAgents.has(id)) {
        stage.removeChild(body)
        stage.removeChild(label)
        body.destroy()
        label.destroy()
        agentGfx.delete(id)
      }
    }
  }, [worldState, selectAgent])

  // 选中状态变化时单独刷新高亮层，避免重绘所有智能体
  useEffect(() => {
    const selGfx = selectionGfxRef.current
    if (!selGfx || !worldState) return

    selGfx.clear()

    // 智能体选中：半透明观测范围圆 + 黄色描边
    if (selectedAgentId) {
      const agent = worldState.agents.find((a) => a.id === selectedAgentId)
      if (agent) {
        const sx = agent.pos[0] * CELL + CELL / 2
        const sy = agent.pos[1] * CELL + CELL / 2
        const r = OBSERVATION_RADIUS * CELL

        selGfx.circle(sx, sy, r).fill({ color: 0xffffff, alpha: 0.05 })
        selGfx.circle(sx, sy, r).stroke({ color: 0xffff00, width: 1, alpha: 0.4 })
        selGfx.circle(sx, sy, 9).stroke({ color: 0xffff00, width: 2 })
      }
    }

    // 物品选中：黄色描边矩形，比物品方块略大以便视觉区分
    if (selectedObjectId) {
      const obj = worldState.objects.find((o) => o.id === selectedObjectId)
      if (obj) {
        const sx = obj.pos[0] * CELL
        const sy = obj.pos[1] * CELL
        selGfx.rect(sx + 4, sy + 4, 16, 16).stroke({ color: 0xffff00, width: 2 })
      }
    }
  }, [worldState, selectedAgentId, selectedObjectId])

  return (
    <div
      ref={containerRef}
      className="flex-shrink-0 w-[600px] h-[600px] overflow-hidden"
    />
  )
}
