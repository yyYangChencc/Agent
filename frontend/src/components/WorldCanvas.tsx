import { useEffect, useRef, useState } from 'react'
import * as PIXI from 'pixi.js'
import { useSimStore } from '../store/simStore'
import type { MapBounds, WorldState } from '../store/simStore'

// 每格像素大小，沿用 D:/2d 的 32px 像素瓦片风格，画布尺寸由后端 map_size 决定
const CELL = 32
// 智能体圆形颜色池，按 ID 哈希取色，保证同一智能体颜色稳定
const AGENT_COLORS = [0xf6d365, 0xff8fa3, 0x76e4f7, 0xc3f584, 0xb69cff]
// 与后端 observer.py 保持一致的观测半径（格数）
const OBSERVATION_RADIUS = 5
const MOVE_STEP_MS = 180
const LAYER_Z = {
  map: 0,
  grid: 10,
  mapLabels: 20,
  objects: 100,
  agents: 200,
  selection: 300,
}

type PixelTileKind = 'grass' | 'path' | 'water' | 'floor' | 'wall' | 'garden'

const TILE_COLORS: Record<PixelTileKind, number> = {
  grass: 0x7fb069,
  path: 0xd7b36a,
  water: 0x4f8fc0,
  floor: 0xbd9560,
  wall: 0x6c584c,
  garden: 0x609b57,
}

const BUILDING_KINDS = new Set(['building', 'bed', 'food_shop', 'playground', 'company'])
const BUILDING_LABELS: Record<string, string> = {
  bed: '床',
  company: '公',
  food_shop: '店',
  playground: '乐',
  building: '筑',
}

type AgentGfx = {
  body: PIXI.Sprite
  label: PIXI.Text
  x: number
  y: number
  animationId: number
}

let agentTexture: PIXI.Texture | null = null

const LABEL_STYLE = {
  fill: '#fff7cf',
  fontFamily: 'monospace',
  fontSize: 10,
  stroke: { color: '#25251c', width: 3 },
}

const BADGE_STYLE = {
  fontFamily: 'monospace',
  fontSize: 9,
  fill: '#fff7cf',
  fontWeight: 'bold' as const,
  stroke: { color: '#25251c', width: 3 },
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

function cellCenter(pos: [number, number]) {
  return {
    x: pos[1] * CELL + CELL / 2,
    y: pos[0] * CELL + CELL / 2,
  }
}

function inferTileKind(row: number, col: number, worldState: WorldState): PixelTileKind {
  const design = worldState.map_design
  if (design) {
    for (const road of design.roads) {
      if (road.cells.some(([roadRow, roadCol]) => roadRow === row && roadCol === col)) {
        return 'path'
      }
    }

    const terrain = [...design.terrain].reverse().find((item) => {
      const [rowStart, colStart, rowEnd, colEnd] = item.bounds
      return row >= rowStart && row <= rowEnd && col >= colStart && col <= colEnd
    })
    if (terrain?.kind === 'plaza' || terrain?.kind === 'office_ground') return 'floor'
    if (terrain?.kind === 'park') return 'garden'
    if (terrain?.kind === 'yard') return 'grass'
  }

  return 'grass'
}

function drawPixelTile(
  gfx: PIXI.Graphics,
  row: number,
  col: number,
  kind: PixelTileKind,
) {
  const x = col * CELL
  const y = row * CELL
  const color = TILE_COLORS[kind]
  gfx.rect(x, y, CELL, CELL).fill(color)
  gfx.rect(x, y, CELL, 2).fill({ color: 0x000000, alpha: 0.08 })
  gfx.rect(x, y, 2, CELL).fill({ color: 0x000000, alpha: 0.06 })

  if (kind === 'grass' && (row * 7 + col * 13) % 5 === 0) {
    gfx.rect(x + 20, y + 8, 4, 10).fill(0x5b8f4b)
  }
  if (kind === 'garden') {
    gfx.rect(x + 6, y + 6, 20, 4).fill(0x4d7d45)
    gfx.rect(x + 6, y + 16, 20, 4).fill(0x4d7d45)
  }
  if (kind === 'path' || kind === 'floor') {
    gfx.rect(x + 4, y + 4, 4, 4).fill({ color: 0xffffff, alpha: 0.08 })
    gfx.rect(x + 22, y + 20, 5, 4).fill({ color: 0x000000, alpha: 0.06 })
  }
  if (kind === 'water') {
    gfx.rect(x + 4, y + 10, 22, 3).fill({ color: 0xa6d4e8, alpha: 0.55 })
    gfx.rect(x + 10, y + 21, 16, 3).fill({ color: 0xa6d4e8, alpha: 0.35 })
  }
}

function makeAgentTexture(): PIXI.Texture {
  if (agentTexture) return agentTexture

  const canvas = document.createElement('canvas')
  canvas.width = 16
  canvas.height = 20
  const context = canvas.getContext('2d')
  if (!context) {
    throw new Error('canvas_context_unavailable')
  }

  context.imageSmoothingEnabled = false
  context.fillStyle = '#25251c'
  context.fillRect(5, 1, 6, 4)
  context.fillStyle = '#f2c39a'
  context.fillRect(4, 4, 8, 7)
  context.fillStyle = '#ffffff'
  context.fillRect(5, 7, 2, 2)
  context.fillRect(9, 7, 2, 2)
  context.fillStyle = '#25251c'
  context.fillRect(5, 8, 2, 1)
  context.fillRect(9, 8, 2, 1)
  context.fillStyle = '#ffffff'
  context.fillRect(4, 11, 8, 7)
  context.fillStyle = '#4b5f9f'
  context.fillRect(4, 18, 3, 2)
  context.fillRect(9, 18, 3, 2)

  agentTexture = PIXI.Texture.from(canvas)
  agentTexture.source.scaleMode = 'nearest'
  return agentTexture
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
    for (let row = 0; row < mapRows; row++) {
      for (let col = 0; col < mapCols; col++) {
        drawPixelTile(mapGfx, row, col, inferTileKind(row, col, worldState))
      }
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
          style: { fontSize: 10, fill: 0xe5e7eb, fontWeight: 'bold' as const },
        })
        label.position.set(region.label_pos[1] * CELL + 4, region.label_pos[0] * CELL + 4)
        labels.addChild(label)
      }
    }
  } else {
    for (let row = 0; row < mapRows; row++) {
      for (let col = 0; col < mapCols; col++) {
        drawPixelTile(mapGfx, row, col, 'grass')
      }
    }
  }

  for (let col = 0; col <= mapCols; col++) {
    grid.moveTo(col * CELL, 0).lineTo(col * CELL, pixelHeight)
  }
  for (let row = 0; row <= mapRows; row++) {
    grid.moveTo(0, row * CELL).lineTo(pixelWidth, row * CELL)
  }
  grid.stroke({ color: 0x25251c, width: 1, alpha: 0.18 })
}

function drawAgentAt(gfx: AgentGfx, x: number, y: number, color: number) {
  gfx.x = x
  gfx.y = y
  gfx.body.position.set(x, y)
  gfx.body.tint = color
  gfx.label.position.set(x, y - 24)
}

function drawPixelObject(
  gfx: PIXI.Graphics,
  kind: string,
  x: number,
  y: number,
) {
  if (kind === 'bed') {
    gfx.rect(x + 5, y + 6, 22, 20).fill(0x5d6ab1)
    gfx.rect(x + 8, y + 8, 16, 7).fill(0xf6e6c8)
    gfx.rect(x + 5, y + 22, 22, 4).fill(0x3e477c)
    return
  }

  if (kind === 'food_shop') {
    gfx.rect(x + 4, y + 10, 24, 18).fill(0xc97b45)
    gfx.rect(x + 3, y + 6, 26, 7).fill(0xf0c15f)
    gfx.rect(x + 8, y + 17, 7, 11).fill(0x5b3a29)
    gfx.rect(x + 18, y + 15, 6, 5).fill(0xf8f5d8)
    return
  }

  if (kind === 'company') {
    gfx.rect(x + 5, y + 5, 22, 23).fill(0x637c9b)
    gfx.rect(x + 9, y + 9, 5, 5).fill(0xdce6ef)
    gfx.rect(x + 18, y + 9, 5, 5).fill(0xdce6ef)
    gfx.rect(x + 13, y + 20, 6, 8).fill(0x344354)
    return
  }

  if (kind === 'playground') {
    gfx.rect(x + 5, y + 21, 22, 4).fill(0x6c584c)
    gfx.rect(x + 8, y + 9, 4, 13).fill(0x4f9d72)
    gfx.rect(x + 20, y + 9, 4, 13).fill(0x4f9d72)
    gfx.rect(x + 7, y + 8, 18, 3).fill(0xf8c86b)
    return
  }

  if (kind === 'building') {
    gfx.rect(x + 5, y + 8, 22, 20).fill(0x8a5a33)
    gfx.rect(x + 4, y + 5, 24, 6).fill(0xc28f5c)
    gfx.rect(x + 13, y + 18, 6, 10).fill(0x4b3223)
    return
  }

  if (kind === 'food') {
    gfx.rect(x + 10, y + 12, 12, 12).fill(0x4d9f45)
    gfx.rect(x + 14, y + 8, 4, 5).fill(0x2d6d33)
    gfx.rect(x + 13, y + 15, 4, 4).fill(0x8fd46a)
    return
  }

  gfx.rect(x + 6, y + 8, 20, 18).fill(0x9b6b43)
  gfx.rect(x + 8, y + 10, 16, 4).fill(0xc28f5c)
}

function animateAgentAlongPath(
  gfx: AgentGfx,
  path: [number, number][],
  color: number,
  onFrame?: () => void,
) {
  if (path.length < 2) return

  const animationId = gfx.animationId + 1
  gfx.animationId = animationId
  const points = path.map(cellCenter)
  let segmentIndex = 0
  let segmentStartTime: number | null = null

  const step = (timestamp: number) => {
    if (gfx.animationId !== animationId) return
    if (segmentStartTime === null) {
      segmentStartTime = timestamp
    }

    const from = points[segmentIndex]
    const to = points[segmentIndex + 1]
    const t = Math.min(1, (timestamp - segmentStartTime) / MOVE_STEP_MS)
    const x = from.x + (to.x - from.x) * t
    const y = from.y + (to.y - from.y) * t
    drawAgentAt(gfx, x, y, color)
    onFrame?.()

    if (t >= 1) {
      segmentIndex += 1
      segmentStartTime = timestamp
      if (segmentIndex >= points.length - 1) {
        const end = points[points.length - 1]
        drawAgentAt(gfx, end.x, end.y, color)
        onFrame?.()
        return
      }
    }

    requestAnimationFrame(step)
  }

  requestAnimationFrame(step)
}

function drawSelection(
  selGfx: PIXI.Graphics,
  worldState: WorldState,
  selectedAgentId: string | null,
  selectedObjectId: string | null,
  agentGfx: Map<string, AgentGfx>,
) {
  selGfx.clear()

  if (selectedAgentId) {
    const agent = worldState.agents.find((a) => a.id === selectedAgentId)
    if (agent) {
      const gfx = agentGfx.get(agent.id)
      const finalPoint = cellCenter(agent.pos)
      const sx = gfx?.x ?? finalPoint.x
      const sy = gfx?.y ?? finalPoint.y
      const r = OBSERVATION_RADIUS * CELL

      selGfx.circle(sx, sy, r).fill({ color: 0xffffff, alpha: 0.05 })
      selGfx.circle(sx, sy, r).stroke({ color: 0xffff00, width: 1, alpha: 0.4 })
      selGfx.rect(sx - 12, sy - 24, 24, 30).stroke({ color: 0xf8c86b, width: 2 })
    }
  }

  if (selectedObjectId) {
    const obj = worldState.objects.find((o) => o.id === selectedObjectId)
    if (obj) {
      const sx = obj.pos[1] * CELL
      const sy = obj.pos[0] * CELL
      if (BUILDING_KINDS.has(obj.kind)) {
        selGfx.rect(sx + 3, sy + 3, CELL - 6, CELL - 6).stroke({ color: 0xf8c86b, width: 2 })
      } else {
        selGfx.rect(sx + 7, sy + 7, 18, 18).stroke({ color: 0xf8c86b, width: 2 })
      }
    }
  }
}

// 将智能体 ID 字符串映射到固定颜色，避免每次渲染随机变色
function agentColor(id: string): number {
  let h = 0
  for (let i = 0; i < id.length; i++) h = (h * 31 + id.charCodeAt(i)) >>> 0
  return AGENT_COLORS[h % AGENT_COLORS.length]
}

export function WorldCanvas() {
  const containerRef = useRef<HTMLDivElement>(null)
  const [pixiReadyVersion, setPixiReadyVersion] = useState(0)
  // 用 ref 持有 Pixi app 实例，避免重渲染时重复初始化
  const appRef = useRef<PIXI.Application | null>(null)
  // 按智能体 ID 缓存 Graphics/Text 对象，tick 时只更新坐标，不重建
  const agentGfxRef = useRef<Map<string, AgentGfx>>(new Map())
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

    // Pixi v8 采用异步 init，需要 await 后才能操作 canvas
    app.init({
      width: 600,
      height: 600,
      background: 0x141714,
      antialias: false,
      resolution: window.devicePixelRatio || 1,
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
      mapGfx.zIndex = LAYER_Z.map
      grid.zIndex = LAYER_Z.grid
      mapLabels.zIndex = LAYER_Z.mapLabels
      objectLayer.zIndex = LAYER_Z.objects
      agentLayer.zIndex = LAYER_Z.agents
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
      selGfx.zIndex = LAYER_Z.selection
      selectionGfxRef.current = selGfx
      app.stage.addChild(selGfx)
      app.stage.sortChildren()
      appRef.current = app
      setPixiReadyVersion((version) => version + 1)
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

    const agentGfx = agentGfxRef.current
    const objectGfx = objectGfxRef.current
    const objectLayer = objectLayerRef.current
    const agentLayer = agentLayerRef.current

    const mapGfx = mapGfxRef.current
    const gridGfx = gridGfxRef.current
    const mapLabels = mapLabelContainerRef.current
    if (!objectLayer || !agentLayer || !mapGfx || !gridGfx || !mapLabels) return

    const signature = mapSignature(worldState, showMapRegions)
    if (signature !== mapSignatureRef.current) {
      drawMapLayers(worldState, showMapRegions, app, mapGfx, gridGfx, mapLabels)
      mapSignatureRef.current = signature
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
          badge = new PIXI.Text({ text: '', style: BADGE_STYLE })
          badge.anchor.set(1, 0)
          objectLayer.addChild(badge)
          kindLabel = new PIXI.Text({
            text: BUILDING_LABELS[obj.kind] ?? obj.kind,
            style: BADGE_STYLE,
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
        drawPixelObject(g, obj.kind, sx, sy)
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
    const movementByAgent = new Map((worldState.movements ?? []).map((movement) => [movement.agent_id, movement.path]))
    for (const agent of worldState.agents) {
      seenAgents.add(agent.id)
      // pos[0] → 行（垂直/Y），pos[1] → 列（水平/X），与后端 map.grid[x][y] 一致
      const finalPoint = cellCenter(agent.pos)
      const color = agentColor(agent.id)

      if (!agentGfx.has(agent.id)) {
        const body = new PIXI.Sprite(makeAgentTexture())
        body.anchor.set(0.5, 0.72)
        body.scale.set(2)
        // Pixi v8 必须显式设置 eventMode 才能接收指针事件
        body.eventMode = 'static'
        body.cursor = 'pointer'
        body.on('pointerdown', () => {
          // 再次点击已选中智能体则取消选中
          selectAgent(agent.id === useSimStore.getState().selectedAgentId ? null : agent.id)
        })
        const label = new PIXI.Text({ text: agent.id, style: LABEL_STYLE })
        // anchor(0.5, 1) 使文字底部中心对齐到目标坐标，便于放在圆圈正上方
        label.anchor.set(0.5, 1)
        agentLayer.addChild(body)
        agentLayer.addChild(label)
        agentGfx.set(agent.id, {
          body,
          label,
          x: finalPoint.x,
          y: finalPoint.y,
          animationId: 0,
        })
      }

      const gfx = agentGfx.get(agent.id)!
      // 在建筑内或睡觉中的智能体不在画布上单独渲染
      const hidden = !!agent.inside_building_id || agent.sleeping
      gfx.body.visible = !hidden
      gfx.label.visible = !hidden
      if (!hidden) {
        const movementPath = movementByAgent.get(agent.id)
        if (movementPath && movementPath.length > 1) {
          animateAgentAlongPath(gfx, movementPath, color, () => {
            const selGfx = selectionGfxRef.current
            const currentWorldState = useSimStore.getState().worldState
            const currentSelectedAgentId = useSimStore.getState().selectedAgentId
            const currentSelectedObjectId = useSimStore.getState().selectedObjectId
            if (selGfx && currentWorldState && currentSelectedAgentId === agent.id) {
              drawSelection(
                selGfx,
                currentWorldState,
                currentSelectedAgentId,
                currentSelectedObjectId,
                agentGfxRef.current,
              )
            }
          })
        } else {
          gfx.animationId += 1
          drawAgentAt(gfx, finalPoint.x, finalPoint.y, color)
        }
      } else {
        gfx.animationId += 1
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
    app.stage.sortChildren()
  }, [worldState, selectAgent, showMapRegions, pixiReadyVersion])

  // 选中状态变化时单独刷新高亮层，避免重绘所有智能体
  useEffect(() => {
    const selGfx = selectionGfxRef.current
    if (!selGfx || !worldState) return

    drawSelection(selGfx, worldState, selectedAgentId, selectedObjectId, agentGfxRef.current)
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
