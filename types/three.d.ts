import { extend } from "@react-three/fiber"
import * as THREE from "three"

// Extend the fiber namespace with Three.js objects
extend(THREE)

declare global {
  namespace JSX {
    interface IntrinsicElements {
      // Geometry
      boxGeometry: any
      sphereGeometry: any
      cylinderGeometry: any
      planeGeometry: any

      // Materials
      meshStandardMaterial: any
      meshBasicMaterial: any
      meshPhongMaterial: any

      // Objects
      mesh: any
      group: any

      // Lights
      ambientLight: any
      directionalLight: any
      pointLight: any
      spotLight: any

      // Helpers
      gridHelper: any
      axesHelper: any
    }
  }
}

export interface ModelObject {
  id: string
  type: "box" | "sphere" | "cylinder"
  position: [number, number, number]
  rotation: [number, number, number]
  scale: [number, number, number]
  color: string
  material?: THREE.Material
  geometry?: THREE.BufferGeometry
}

export interface Tool {
  name: string
  icon: string
  action: string
}

export interface Measurement {
  id: string
  type: "distance" | "angle" | "area" | "volume"
  value: number
  unit: string
  points: THREE.Vector3[]
}
