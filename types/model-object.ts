import type React from "react"
export type ModelObject = {
  id: string
  name?: string // Optional display name
  type: "box" | "sphere" | "cylinder" | "cone" | "torus" | "plane" | "wedge"
  position: [number, number, number]
  rotation: [number, number, number]
  scale: [number, number, number]
  color: string
  materialType?: "standard" | "phong" | "toon" | "lambert" | "physical"
  wireframe?: boolean
  opacity?: number
  visible?: boolean
  locked?: boolean
  parentId?: string | null
  createdAt: number
  updatedAt: number
}

export type BoxParams = {
  width: number
  height: number
  depth: number
}

export type SphereParams = {
  radius: number
  widthSegments?: number
  heightSegments?: number
}

export type CylinderParams = {
  radiusTop: number
  radiusBottom: number
  height: number
  radialSegments?: number
}

export type TorusParams = {
  radius: number
  tubeRadius: number
  arc?: number
  segments?: number
}

export type GeometryParams =
  | { type: "box"; params: BoxParams }
  | { type: "sphere"; params: SphereParams }
  | { type: "cylinder"; params: CylinderParams }
  | { type: "torus"; params: TorusParams }

export interface Tool {
  name: string
  icon: React.ReactNode
  action: string
}

export interface ToolCategory {
  name: string
  icon: React.ReactNode
  expanded: boolean
  tools: Tool[]
}
