"use client"

import { useRef } from "react"
import { Canvas, useFrame, extend } from "@react-three/fiber"
import * as THREE from "three"
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls"

// Extend the fiber namespace
extend({ OrbitControls, ...THREE })

function SpinningCube() {
  const meshRef = useRef<THREE.Mesh>(null)

  useFrame((state, delta) => {
    if (meshRef.current) {
      meshRef.current.rotation.x += delta * 0.5
      meshRef.current.rotation.y += delta * 0.8
    }
  })

  return (
    <mesh ref={meshRef} className="animate-orbit">
      <boxGeometry args={[1, 1, 1]} />
      <meshStandardMaterial
        color={new THREE.Color().setHSL(270 / 360, 0.6, 0.35)} // nebula-purple
        wireframe
        transparent
        opacity={0.8}
      />
    </mesh>
  )
}

export function ModelLoadingSpinner() {
  return (
    <div
      className="flex flex-col items-center justify-center h-full w-full bg-cosmic-gradient min-h-[60px] p-6"
      role="status"
      aria-live="polite"
      aria-label="Loading 3D environment, please wait"
    >
      <div className="w-32 h-32 mb-4">
        <Canvas camera={{ position: [2, 2, 2], fov: 60 }}>
          <ambientLight intensity={0.5} />
          <directionalLight
            position={[10, 10, 5]}
            color={new THREE.Color().setHSL(30 / 360, 0.7, 0.6)} // star-yellow
            intensity={1}
          />
          <SpinningCube />
          <OrbitControls enablePan={false} enableZoom={false} enableRotate={true} />
        </Canvas>
      </div>
      <div className="text-center animate-shimmer">
        <div className="flex items-center gap-2 mb-2">
          <div
            className="w-2 h-2 bg-nebula-purple rounded-full animate-bounce"
            style={{ animationDelay: "0.1s" }}
          />
          <div
            className="w-2 h-2 bg-star-yellow rounded-full animate-bounce"
            style={{ animationDelay: "0.2s" }}
          />
          <div
            className="w-2 h-2 bg-nebula-purple rounded-full animate-bounce"
            style={{ animationDelay: "0.3s" }}
          />
        </div>
        <p className="text-sm text-muted-foreground animate-pulse">Loading 3D Environment...</p>
      </div>
    </div>
  )
}