import { useEffect, useRef, useState } from 'react'
import * as PIXI from 'pixi.js'
import { useSimStore } from '../store/simStore'
import type { MapBounds, MapPosition, ObjectState, WorldState } from '../store/simStore'

// 每格像素大小，沿用 D:/2d 的 32px 像素瓦片风格，画布尺寸由后端 map_size 决定
const CELL = 32
// 智能体圆形颜色池，按 ID 哈希取色，保证同一智能体颜色稳定
const AGENT_COLORS = [0xf6d365, 0xff8fa3, 0x76e4f7, 0xc3f584, 0xb69cff]
// 与后端 observer.py 保持一致的观测半径（格数）
const OBSERVATION_RADIUS = 5
const MOVE_STEP_MS = 180
const LAYER_Z = {
  mapTiles: 0,
  map: 5,
  roads: 10,
  mapLabels: 20,
  decorations: 30,
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

type SpriteCrop = { x: number; y: number; width: number; height: number }

// 裁切框与已验收测试地图使用的素材有效像素范围完全一致。
const WORLD_ASSET_URLS = {
  homes: new URL('../assets/world/agent-homes.png', import.meta.url).href,
  companies: new URL('../assets/world/company-buildings.png', import.meta.url).href,
  shops: new URL('../assets/world/shop-buildings.png', import.meta.url).href,
  playgrounds: new URL('../assets/world/recreation-facilities.png', import.meta.url).href,
  terrain: new URL('../assets/world/terrain-and-nature.png', import.meta.url).href,
}

const HOUSE_SPRITES: SpriteCrop[] = [
  { x: 171, y: 189, width: 364, height: 263 },
  { x: 575, y: 189, width: 399, height: 264 },
  { x: 1023, y: 203, width: 398, height: 251 },
  { x: 147, y: 551, width: 422, height: 237 },
  { x: 587, y: 537, width: 411, height: 251 },
  { x: 1035, y: 539, width: 387, height: 249 },
]

const COMPANY_SPRITES: SpriteCrop[] = [
  { x: 168, y: 92, width: 364, height: 302 },
  { x: 586, y: 54, width: 392, height: 340 },
  { x: 1031, y: 89, width: 396, height: 305 },
  { x: 172, y: 464, width: 361, height: 280 },
  { x: 591, y: 475, width: 399, height: 269 },
  { x: 1042, y: 474, width: 382, height: 270 },
]

const SHOP_SPRITES: SpriteCrop[] = [
  { x: 131, y: 114, width: 401, height: 318 },
  { x: 576, y: 131, width: 403, height: 302 },
  { x: 1029, y: 120, width: 398, height: 313 },
  { x: 131, y: 563, width: 401, height: 285 },
  { x: 577, y: 555, width: 403, height: 292 },
  { x: 1027, y: 559, width: 401, height: 288 },
]

const PLAYGROUND_SPRITES: SpriteCrop[] = [
  { x: 48, y: 48, width: 432, height: 394 },
  { x: 537, y: 49, width: 424, height: 394 },
  { x: 54, y: 559, width: 426, height: 337 },
  { x: 536, y: 552, width: 425, height: 336 },
]

const TERRAIN_SPRITES = {
  grass: { x: 84, y: 72, width: 167, height: 136 },
  stone_road: { x: 304, y: 296, width: 176, height: 144 },
  trees: [
    { x: 54, y: 496, width: 234, height: 226 },
    { x: 319, y: 487, width: 159, height: 234 },
    { x: 504, y: 496, width: 224, height: 224 },
    { x: 744, y: 496, width: 231, height: 224 },
  ],
  shrubs: [
    { x: 72, y: 791, width: 188, height: 120 },
    { x: 296, y: 768, width: 202, height: 144 },
  ],
}

type WorldTextureMap = Map<string, PIXI.Texture>

let worldTexturesPromise: Promise<WorldTextureMap> | null = null

function croppedTexture(base: PIXI.Texture, crop: SpriteCrop): PIXI.Texture {
  const texture = new PIXI.Texture({
    source: base.source,
    frame: new PIXI.Rectangle(crop.x, crop.y, crop.width, crop.height),
  })
  texture.source.scaleMode = 'nearest'
  return texture
}

function registerTextureSeries(
  target: WorldTextureMap,
  prefix: string,
  base: PIXI.Texture,
  crops: SpriteCrop[],
) {
  crops.forEach((crop, index) => target.set(`${prefix}_${index}`, croppedTexture(base, crop)))
}

function loadWorldTextures(): Promise<WorldTextureMap> {
  if (worldTexturesPromise) return worldTexturesPromise
  worldTexturesPromise = Promise.all([
    PIXI.Assets.load<PIXI.Texture>(WORLD_ASSET_URLS.homes),
    PIXI.Assets.load<PIXI.Texture>(WORLD_ASSET_URLS.companies),
    PIXI.Assets.load<PIXI.Texture>(WORLD_ASSET_URLS.shops),
    PIXI.Assets.load<PIXI.Texture>(WORLD_ASSET_URLS.playgrounds),
    PIXI.Assets.load<PIXI.Texture>(WORLD_ASSET_URLS.terrain),
  ]).then(([homes, companies, shops, playgrounds, terrain]) => {
    const textures: WorldTextureMap = new Map()
    registerTextureSeries(textures, 'home', homes, HOUSE_SPRITES)
    registerTextureSeries(textures, 'company', companies, COMPANY_SPRITES)
    registerTextureSeries(textures, 'shop', shops, SHOP_SPRITES)
    registerTextureSeries(textures, 'playground', playgrounds, PLAYGROUND_SPRITES)
    textures.set('grass', croppedTexture(terrain, TERRAIN_SPRITES.grass))
    textures.set('stone_road', croppedTexture(terrain, TERRAIN_SPRITES.stone_road))
    registerTextureSeries(textures, 'tree', terrain, TERRAIN_SPRITES.trees)
    registerTextureSeries(textures, 'shrub', terrain, TERRAIN_SPRITES.shrubs)
    return textures
  })
  return worldTexturesPromise
}

type AgentGfx = {
  body: PIXI.Sprite
  label: PIXI.Text
  x: number
  y: number
  animationId: number
}

type ObjectGfx = {
  container: PIXI.Container
  shape: PIXI.Graphics
  sprite: PIXI.Sprite
  badge: PIXI.Text | null
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

function objectFootprint(obj: ObjectState): MapPosition[] {
  return obj.footprint.length > 0 ? obj.footprint : [obj.pos]
}

function objectBounds(obj: ObjectState) {
  const footprint = objectFootprint(obj)
  const rows = footprint.map(([row]) => row)
  const cols = footprint.map(([, col]) => col)
  const rowStart = Math.min(...rows)
  const rowEnd = Math.max(...rows)
  const colStart = Math.min(...cols)
  const colEnd = Math.max(...cols)
  return {
    rowStart,
    rowEnd,
    colStart,
    colEnd,
    width: (colEnd - colStart + 1) * CELL,
    height: (rowEnd - rowStart + 1) * CELL,
  }
}

function inferTileKind(
  row: number,
  col: number,
  worldState: WorldState,
  roadCells?: Set<string>,
): PixelTileKind {
  const design = worldState.map_design
  if (design) {
    if (roadCells) {
      if (roadCells.has(`${row},${col}`)) return 'path'
    } else {
      for (const road of design.roads) {
        if (road.cells.some(([roadRow, roadCol]) => roadRow === row && roadCol === col)) {
          return 'path'
        }
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

function destroyContainerChildren(container: PIXI.Container) {
  container.removeChildren().forEach((child) => child.destroy())
}

function addTileSprite(
  layer: PIXI.Container,
  texture: PIXI.Texture,
  row: number,
  col: number,
) {
  const sprite = new PIXI.Sprite(texture)
  sprite.position.set(col * CELL, row * CELL)
  sprite.width = CELL
  sprite.height = CELL
  layer.addChild(sprite)
}

function drawDecorations(
  worldState: WorldState,
  layer: PIXI.Container,
  textures: WorldTextureMap,
) {
  for (const decoration of worldState.map_design?.decorations ?? []) {
    const texture = textures.get(decoration.sprite_key)
    if (!texture) continue
    const sprite = new PIXI.Sprite(texture)
    const maxWidth = decoration.kind === 'tree' ? 30 : 29
    const maxHeight = decoration.kind === 'tree' ? 31 : 19
    const scale = Math.min(maxWidth / texture.width, maxHeight / texture.height)
    sprite.width = Math.round(texture.width * scale)
    sprite.height = Math.round(texture.height * scale)
    sprite.position.set(
      decoration.pos[1] * CELL + Math.round((CELL - sprite.width) / 2),
      decoration.pos[0] * CELL + CELL - sprite.height,
    )
    sprite.zIndex = decoration.pos[0]
    layer.addChild(sprite)
  }
  layer.sortChildren()
}

function drawMapLayers(
  worldState: WorldState,
  showMapRegions: boolean,
  app: PIXI.Application,
  mapGfx: PIXI.Graphics,
  tileLayer: PIXI.Container,
  roadLayer: PIXI.Container,
  labels: PIXI.Container,
  decorationLayer: PIXI.Container,
  textures: WorldTextureMap,
) {
  const [mapCols, mapRows] = worldState.map_size
  const pixelWidth = mapCols * CELL
  const pixelHeight = mapRows * CELL

  if (app.canvas.width !== pixelWidth || app.canvas.height !== pixelHeight) {
    app.renderer.resize(pixelWidth, pixelHeight)
  }

  mapGfx.clear()
  destroyContainerChildren(tileLayer)
  destroyContainerChildren(roadLayer)
  destroyContainerChildren(labels)
  destroyContainerChildren(decorationLayer)

  if (worldState.map_design) {
    const roadCells = new Set(
      worldState.map_design.roads.flatMap((road) => (
        road.cells.map(([row, col]) => `${row},${col}`)
      )),
    )
    const grassSpriteKey = worldState.map_design.tile_sprites?.grass
    const grassTexture = grassSpriteKey ? textures.get(grassSpriteKey) : undefined
    if (grassTexture) {
      // 大地图只创建一个草地重复纹理层，避免为每个草地格创建 Sprite。
      const grassLayer = new PIXI.TilingSprite({
        texture: grassTexture,
        width: pixelWidth,
        height: pixelHeight,
        tileScale: {
          x: CELL / grassTexture.width,
          y: CELL / grassTexture.height,
        },
      })
      tileLayer.addChild(grassLayer)
    }
    for (let row = 0; row < mapRows; row++) {
      for (let col = 0; col < mapCols; col++) {
        const kind = inferTileKind(row, col, worldState, roadCells)
        const spriteKey = kind === 'grass'
          ? worldState.map_design.tile_sprites?.grass
          : kind === 'path'
            ? worldState.map_design.tile_sprites?.road
            : undefined
        const texture = spriteKey ? textures.get(spriteKey) : undefined
        if (kind === 'grass' && grassTexture) {
          continue
        } else if (texture) {
          addTileSprite(kind === 'path' ? roadLayer : tileLayer, texture, row, col)
        } else {
          drawPixelTile(mapGfx, row, col, kind)
        }
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
  drawDecorations(worldState, decorationLayer, textures)
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

function drawObject(
  gfx: ObjectGfx,
  obj: ObjectState,
  textures: WorldTextureMap,
) {
  const bounds = objectBounds(obj)
  const hidden = obj.num !== null && obj.num <= 0
  gfx.container.position.set(bounds.colStart * CELL, bounds.rowStart * CELL)
  gfx.container.zIndex = bounds.rowEnd
  gfx.container.hitArea = new PIXI.Rectangle(0, 0, bounds.width, bounds.height)
  gfx.container.visible = !hidden
  gfx.shape.clear()

  const texture = obj.sprite_key ? textures.get(obj.sprite_key) : undefined
  if (texture && BUILDING_KINDS.has(obj.kind)) {
    gfx.sprite.texture = texture
    const maxWidth = Math.max(1, bounds.width - 4)
    const scale = obj.kind === 'bed'
      ? maxWidth / texture.width
      : Math.min(maxWidth / texture.width, Math.max(1, bounds.height - 6) / texture.height)
    gfx.sprite.width = Math.round(texture.width * scale)
    gfx.sprite.height = Math.round(texture.height * scale)
    gfx.sprite.position.set(
      Math.round((bounds.width - gfx.sprite.width) / 2),
      Math.round(bounds.height - gfx.sprite.height - (obj.kind === 'bed' ? 1 : 3)),
    )
    gfx.sprite.visible = true
  } else {
    gfx.sprite.visible = false
    drawPixelObject(gfx.shape, obj.kind, 0, 0)
  }

  if (gfx.badge) {
    const count = obj.occupant_count ?? 0
    gfx.badge.text = String(count)
    gfx.badge.position.set(bounds.width - 1, 1)
    gfx.badge.visible = count > 0
  }
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
      const bounds = objectBounds(obj)
      const sx = bounds.colStart * CELL
      const sy = bounds.rowStart * CELL
      if (BUILDING_KINDS.has(obj.kind)) {
        selGfx
          .rect(sx + 3, sy + 3, bounds.width - 6, bounds.height - 6)
          .stroke({ color: 0xf8c86b, width: 2 })
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
  // 按对象 ID 缓存素材容器和占用数量标签。
  const objectGfxRef = useRef<Map<string, ObjectGfx>>(new Map())
  const worldTexturesRef = useRef<WorldTextureMap>(new Map())
  const mapGfxRef = useRef<PIXI.Graphics | null>(null)
  const mapTileLayerRef = useRef<PIXI.Container | null>(null)
  const roadLayerRef = useRef<PIXI.Container | null>(null)
  const mapLabelContainerRef = useRef<PIXI.Container | null>(null)
  const decorationLayerRef = useRef<PIXI.Container | null>(null)
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
    }).then(async () => {
      let textures: WorldTextureMap = new Map()
      try {
        textures = await loadWorldTextures()
      } catch (error) {
        // 素材加载失败时保留原像素图形，避免画布整体不可用。
        console.error('world_assets_load_failed', error)
      }
      if (cancelled) {
        // cleanup 先于 init 完成时，在这里安全销毁
        app.destroy(true)
        return
      }
      initialized = true
      el.appendChild(app.canvas)

      app.stage.sortableChildren = true
      const mapTiles = new PIXI.Container()
      const mapGfx = new PIXI.Graphics()
      const roads = new PIXI.Container()
      const mapLabels = new PIXI.Container()
      const decorations = new PIXI.Container()
      const objectLayer = new PIXI.Container()
      const agentLayer = new PIXI.Container()
      mapTiles.zIndex = LAYER_Z.mapTiles
      mapGfx.zIndex = LAYER_Z.map
      roads.zIndex = LAYER_Z.roads
      mapLabels.zIndex = LAYER_Z.mapLabels
      decorations.zIndex = LAYER_Z.decorations
      objectLayer.zIndex = LAYER_Z.objects
      agentLayer.zIndex = LAYER_Z.agents
      decorations.sortableChildren = true
      objectLayer.sortableChildren = true
      worldTexturesRef.current = textures
      mapGfxRef.current = mapGfx
      mapTileLayerRef.current = mapTiles
      roadLayerRef.current = roads
      mapLabelContainerRef.current = mapLabels
      decorationLayerRef.current = decorations
      objectLayerRef.current = objectLayer
      agentLayerRef.current = agentLayer
      app.stage.addChild(mapTiles)
      app.stage.addChild(mapGfx)
      app.stage.addChild(roads)
      app.stage.addChild(mapLabels)
      app.stage.addChild(decorations)
      app.stage.addChild(objectLayer)
      app.stage.addChild(agentLayer)
      const currentWorldState = useSimStore.getState().worldState
      if (currentWorldState) {
        const currentShowMapRegions = useSimStore.getState().showMapRegions
        drawMapLayers(
          currentWorldState,
          currentShowMapRegions,
          app,
          mapGfx,
          mapTiles,
          roads,
          mapLabels,
          decorations,
          textures,
        )
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
      mapTileLayerRef.current = null
      roadLayerRef.current = null
      mapLabelContainerRef.current = null
      decorationLayerRef.current = null
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
    const mapTiles = mapTileLayerRef.current
    const roads = roadLayerRef.current
    const mapLabels = mapLabelContainerRef.current
    const decorations = decorationLayerRef.current
    if (!objectLayer || !agentLayer || !mapGfx || !mapTiles || !roads || !mapLabels || !decorations) return

    const signature = mapSignature(worldState, showMapRegions)
    if (signature !== mapSignatureRef.current) {
      drawMapLayers(
        worldState,
        showMapRegions,
        app,
        mapGfx,
        mapTiles,
        roads,
        mapLabels,
        decorations,
        worldTexturesRef.current,
      )
      mapSignatureRef.current = signature
    }

    // 更新场景物体（食物等）
    const seenObjects = new Set<string>()
    for (const obj of worldState.objects) {
      seenObjects.add(obj.id)

      // 首次出现时创建一个可点击容器，后续只更新素材和位置。
      if (!objectGfx.has(obj.id)) {
        const container = new PIXI.Container()
        const shape = new PIXI.Graphics()
        const sprite = new PIXI.Sprite(PIXI.Texture.EMPTY)
        // 物品支持点击选中，与智能体圆圈行为一致
        container.eventMode = 'static'
        container.cursor = 'pointer'
        container.on('pointerdown', () => {
          const { selectedObjectId: curId, selectObject: sel } = useSimStore.getState()
          sel(obj.id === curId ? null : obj.id)
        })
        container.addChild(shape)
        container.addChild(sprite)
        // 建筑人数徽标与素材使用同一容器，跟随完整占地移动。
        let badge: PIXI.Text | null = null
        if (BUILDING_KINDS.has(obj.kind)) {
          badge = new PIXI.Text({ text: '', style: BADGE_STYLE })
          badge.anchor.set(1, 0)
          container.addChild(badge)
        }
        objectLayer.addChild(container)
        objectGfx.set(obj.id, { container, shape, sprite, badge })
      }
      drawObject(objectGfx.get(obj.id)!, obj, worldTexturesRef.current)
    }
    // 清除服务端已不存在的物体
    for (const [id, { container }] of objectGfx) {
      if (!seenObjects.has(id)) {
        objectLayer.removeChild(container)
        container.destroy({ children: true })
        objectGfx.delete(id)
      }
    }
    objectLayer.sortChildren()

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
