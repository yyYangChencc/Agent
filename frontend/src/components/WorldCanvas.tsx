import { useEffect, useRef } from 'react'
import * as PIXI from 'pixi.js'
import { useSimStore } from '../store/simStore'
import type { MapBounds, WorldState } from '../store/simStore'

// 每格像素大小，画布尺寸由后端 map_size 决定
const CELL = 24
// 智能体圆形颜色池，按 ID 哈希取色，保证同一智能体颜色稳定
const AGENT_COLORS = [0x4f8ef7, 0xe74c6f, 0x2ecc71, 0xf39c12, 0x9b59b6]
// 与后端 observer.py 保持一致的观测半径（格数）
const OBSERVATION_RADIUS = 5

const BUILDING_KINDS = new Set(['building', 'bed', 'food_shop', 'playground', 'company'])
const BUILDING_COLORS: Record<string, number> = {
  bed: 0x8b5cf6,
  company: 0x3b82f6,
  food_shop: 0xf59e0b,
  playground: 0x10b981,
  building: 0x6b7280,
}
const BUILDING_LABELS: Record<string, string> = {
  bed: '床',
  company: '公',
  food_shop: '店',
  playground: '乐',
  building: '筑',
}

function colorFromHex(value: string, fallback: number): number {
  if (/^#[0-9a-fA-F]{6}$/.test(value)) {
    return Number(`0x${value.slice(1)}`)
  }
  return fallback
}

function boundsToRect(bounds: MapBounds) {
  const [rowStart, colStart, rowEnd, colEnd] = bounds
  return {
    x: colStart * CELL,
    y: rowStart * CELL,
    width: (colEnd - colStart + 1) * CELL,
    height: (rowEnd - rowStart + 1) * CELL,
  }
}

function mapSignature(worldState: WorldState, showMapRegions: boolean): string {
  return JSON.stringify({
    map_size: worldState.map_size,
    map_design: worldState.map_design,
    showMapRegions,
  })
}

function drawMapLayers(
  worldState: WorldState,
  showMapRegions: boolean,
  app: PIXI.Application,
  mapGfx: PIXI.Graphics,
  grid: PIXI.Graphics,
  labels: PIXI.Container,
) {
  const [mapCols, mapRows] = worldState.map_size
  const pixelWidth = mapCols * CELL
  const pixelHeight = mapRows * CELL

  if (app.canvas.width !== pixelWidth || app.canvas.height !== pixelHeight) {
    app.renderer.resize(pixelWidth, pixelHeight)
  }

  mapGfx.clear()
  grid.clear()
  labels.removeChildren().forEach((child) => child.destroy())

  if (worldState.map_design) {
    for (const terrain of worldState.map_design.terrain) {
      const rect = boundsToRect(terrain.bounds)
      mapGfx
        .rect(rect.x, rect.y, rect.width, rect.height)
        .fill({ color: colorFromHex(terrain.color, 0x203a2f), alpha: terrain.alpha })
    }

    if (showMapRegions) {
      for (const region of worldState.map_design.regions) {
        const rect = boundsToRect(region.bounds)
        const color = colorFromHex(region.color, 0x6b7280)
        mapGfx
          .rect(rect.x, rect.y, rect.width, rect.height)
          .fill({ color, alpha: 0.18 })
        mapGfx
          .rect(rect.x + 1, rect.y + 1, rect.width - 2, rect.height - 2)
          .stroke({ color, width: 1, alpha: 0.45 })

        const label = new PIXI.Text({
          text: region.name,
          style: { fontSize: 10, fill: 0xe5e7eb, fontWeight: 'bold' },
        })
        label.position.set(region.label_pos[1] * CELL + 4, region.label_pos[0] * CELL + 4)
        labels.addChild(label)
      }
    }

    for (const road of worldState.map_design.roads) {
      const color = colorFromHex(road.color, 0x8a7356)
      for (const [row, col] of road.cells) {
        mapGfx
          .rect(col * CELL, row * CELL, CELL, CELL)
          .fill({ color, alpha: 0.75 })
      }
    }
  } else {
    mapGfx.rect(0, 0, pixelWidth, pixelHeight).fill(0x1a1a2e)
  }

  for (let col = 0; col <= mapCols; col++) {
    grid.moveTo(col * CELL, 0).lineTo(col * CELL, pixelHeight)
  }
  for (let row = 0; row <= mapRows; row++) {
    grid.moveTo(0, row * CELL).lineTo(pixelWidth, row * CELL)
  }
  grid.stroke({ color: 0x2a2a4a, width: 1 })
}

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
  // 按对象 ID 缓存场景物体的 Graphics 和可选的占用数量标签
  const objectGfxRef = useRef<Map<string, { g: PIXI.Graphics; badge: PIXI.Text | null; kindLabel: PIXI.Text | null }>>(new Map())
  const mapGfxRef = useRef<PIXI.Graphics | null>(null)
  const gridGfxRef = useRef<PIXI.Graphics | null>(null)
  const mapLabelContainerRef = useRef<PIXI.Container | null>(null)
  const objectLayerRef = useRef<PIXI.Container | null>(null)
  const agentLayerRef = useRef<PIXI.Container | null>(null)
  const mapSignatureRef = useRef<string>('')
  // 选中高亮层（观测范围圆 + 黄色描边），独立于智能体图层便于整体清除
  const selectionGfxRef = useRef<PIXI.Graphics | null>(null)

  const worldState = useSimStore((s) => s.worldState)
  const selectedAgentId = useSimStore((s) => s.selectedAgentId)
  const selectedObjectId = useSimStore((s) => s.selectedObjectId)
  const showMapRegions = useSimStore((s) => s.showMapRegions)
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

      app.stage.sortableChildren = true
      const mapGfx = new PIXI.Graphics()
      const grid = new PIXI.Graphics()
      const mapLabels = new PIXI.Container()
      const objectLayer = new PIXI.Container()
      const agentLayer = new PIXI.Container()
      mapGfx.zIndex = 0
      grid.zIndex = 10
      mapLabels.zIndex = 20
      objectLayer.zIndex = 100
      agentLayer.zIndex = 200
      mapGfxRef.current = mapGfx
      gridGfxRef.current = grid
      mapLabelContainerRef.current = mapLabels
      objectLayerRef.current = objectLayer
      agentLayerRef.current = agentLayer
      app.stage.addChild(mapGfx)
      app.stage.addChild(grid)
      app.stage.addChild(mapLabels)
      app.stage.addChild(objectLayer)
      app.stage.addChild(agentLayer)
      const currentWorldState = useSimStore.getState().worldState
      if (currentWorldState) {
        const currentShowMapRegions = useSimStore.getState().showMapRegions
        drawMapLayers(currentWorldState, currentShowMapRegions, app, mapGfx, grid, mapLabels)
        mapSignatureRef.current = mapSignature(currentWorldState, currentShowMapRegions)
      }

      // 选中高亮层在最上方，保证描边和观测范围不会被地图或对象遮挡
      const selGfx = new PIXI.Graphics()
      selGfx.zIndex = 300
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
      mapGfxRef.current = null
      gridGfxRef.current = null
      mapLabelContainerRef.current = null
      objectLayerRef.current = null
      agentLayerRef.current = null
      mapSignatureRef.current = ''
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
    const objectLayer = objectLayerRef.current ?? stage
    const agentLayer = agentLayerRef.current ?? stage

    const mapGfx = mapGfxRef.current
    const gridGfx = gridGfxRef.current
    const mapLabels = mapLabelContainerRef.current
    if (mapGfx && gridGfx && mapLabels) {
      const signature = mapSignature(worldState, showMapRegions)
      if (signature !== mapSignatureRef.current) {
        drawMapLayers(worldState, showMapRegions, app, mapGfx, gridGfx, mapLabels)
        mapSignatureRef.current = signature
      }
    }

    // 更新场景物体（食物等）
    const seenObjects = new Set<string>()
    for (const obj of worldState.objects) {
      seenObjects.add(obj.id)
      const sx = obj.pos[1] * CELL
      const sy = obj.pos[0] * CELL

      // 首次出现时创建 Graphics（和可选的占用数量标签），后续只重绘
      if (!objectGfx.has(obj.id)) {
        const g = new PIXI.Graphics()
        // 物品支持点击选中，与智能体圆圈行为一致
        g.eventMode = 'static'
        g.cursor = 'pointer'
        g.on('pointerdown', () => {
          const { selectedObjectId: curId, selectObject: sel } = useSimStore.getState()
          sel(obj.id === curId ? null : obj.id)
        })
        objectLayer.addChild(g)
        // 建筑才需要占用数量标签和种类标签
        let badge: PIXI.Text | null = null
        let kindLabel: PIXI.Text | null = null
        if (BUILDING_KINDS.has(obj.kind)) {
          badge = new PIXI.Text({ text: '', style: { fontSize: 8, fill: 0xffffff } })
          badge.anchor.set(1, 0)
          objectLayer.addChild(badge)
          kindLabel = new PIXI.Text({
            text: BUILDING_LABELS[obj.kind] ?? obj.kind,
            style: { fontSize: 10, fill: 0xffffff, fontWeight: 'bold' },
          })
          kindLabel.anchor.set(0.5, 0.5)
          objectLayer.addChild(kindLabel)
        }
        objectGfx.set(obj.id, { g, badge, kindLabel })
      }
      const { g, badge, kindLabel } = objectGfx.get(obj.id)!
      g.clear()
      // num <= 0 表示物品已耗尽，隐藏方块而非移除，保留对象引用
      const hidden = obj.num !== null && obj.num <= 0
      if (!hidden) {
        if (BUILDING_KINDS.has(obj.kind)) {
          // 建筑：按 kind 填充不同颜色的整格
          const color = BUILDING_COLORS[obj.kind] ?? 0x6b7280
          g.rect(sx, sy, CELL, CELL).fill(color)
        } else {
          // 默认（含 food）：绿色小方块
          g.rect(sx + 6, sy + 6, 12, 12).fill(0x2ecc71)
        }
      }
      // 更新种类标签（格子中央）
      if (kindLabel) {
        kindLabel.visible = !hidden
        kindLabel.position.set(sx + CELL / 2, sy + CELL / 2)
      }
      // 更新占用数量标签（右上角）
      if (badge) {
        const count = obj.occupant_count ?? 0
        if (count > 0 && !hidden) {
          badge.text = String(count)
          badge.position.set(sx + CELL - 1, sy + 1)
          badge.visible = true
        } else {
          badge.visible = false
        }
      }
    }
    // 清除服务端已不存在的物体
    for (const [id, { g, badge, kindLabel }] of objectGfx) {
      if (!seenObjects.has(id)) {
        objectLayer.removeChild(g)
        g.destroy()
        if (badge) {
          objectLayer.removeChild(badge)
          badge.destroy()
        }
        if (kindLabel) {
          objectLayer.removeChild(kindLabel)
          kindLabel.destroy()
        }
        objectGfx.delete(id)
      }
    }

    // 更新智能体
    const seenAgents = new Set<string>()
    for (const agent of worldState.agents) {
      seenAgents.add(agent.id)
      // pos[0] → 行（垂直/Y），pos[1] → 列（水平/X），与后端 map.grid[x][y] 一致
      const sx = agent.pos[1] * CELL + CELL / 2
      const sy = agent.pos[0] * CELL + CELL / 2
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
        agentLayer.addChild(body)
        agentLayer.addChild(label)
        agentGfx.set(agent.id, { body, label })
      }

      const { body, label } = agentGfx.get(agent.id)!
      // 在建筑内或睡觉中的智能体不在画布上单独渲染
      const hidden = !!agent.inside_building_id || agent.sleeping
      body.visible = !hidden
      label.visible = !hidden
      if (!hidden) {
        body.clear()
        body.circle(sx, sy, 9).fill(color)
        // 标签放在圆圈上方 11px，留出圆形半径 + 2px 间距
        label.position.set(sx, sy - 11)
      }
    }
    // 清除服务端已不存在的智能体
    for (const [id, { body, label }] of agentGfx) {
      if (!seenAgents.has(id)) {
        agentLayer.removeChild(body)
        agentLayer.removeChild(label)
        body.destroy()
        label.destroy()
        agentGfx.delete(id)
      }
    }
  }, [worldState, selectAgent, showMapRegions])

  // 选中状态变化时单独刷新高亮层，避免重绘所有智能体
  useEffect(() => {
    const selGfx = selectionGfxRef.current
    if (!selGfx || !worldState) return

    selGfx.clear()

    // 智能体选中：半透明观测范围圆 + 黄色描边
    if (selectedAgentId) {
      const agent = worldState.agents.find((a) => a.id === selectedAgentId)
      if (agent) {
        const sx = agent.pos[1] * CELL + CELL / 2
        const sy = agent.pos[0] * CELL + CELL / 2
        const r = OBSERVATION_RADIUS * CELL

        selGfx.circle(sx, sy, r).fill({ color: 0xffffff, alpha: 0.05 })
        selGfx.circle(sx, sy, r).stroke({ color: 0xffff00, width: 1, alpha: 0.4 })
        selGfx.circle(sx, sy, 9).stroke({ color: 0xffff00, width: 2 })
      }
    }

    // 物品选中：建筑用全格描边，食物用小方块描边
    if (selectedObjectId) {
      const obj = worldState.objects.find((o) => o.id === selectedObjectId)
      if (obj) {
        const sx = obj.pos[1] * CELL
        const sy = obj.pos[0] * CELL
        if (BUILDING_KINDS.has(obj.kind)) {
          selGfx.rect(sx, sy, CELL, CELL).stroke({ color: 0xffff00, width: 2 })
        } else {
          selGfx.rect(sx + 4, sy + 4, 16, 16).stroke({ color: 0xffff00, width: 2 })
        }
      }
    }
  }, [worldState, selectedAgentId, selectedObjectId])

  const canvasWidth = (worldState?.map_size[0] ?? 25) * CELL
  const canvasHeight = (worldState?.map_size[1] ?? 25) * CELL

  return (
    <div
      ref={containerRef}
      className="flex-shrink-0 overflow-hidden"
      style={{ width: canvasWidth, height: canvasHeight }}
    />
  )
}
