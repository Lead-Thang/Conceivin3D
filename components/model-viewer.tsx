"use client"

import { TOOL_CATEGORIES } from '@/types/tool-category'

import type React from "react"
import { useState, useRef, useCallback, useEffect, Suspense } from "react"
import { Canvas, useFrame, extend } from "@react-three/fiber"
import { OrbitControls, Grid, GizmoHelper, GizmoViewport, Environment } from "@react-three/drei"
import { useModelViewer } from "@/hooks/use-model-viewer"
import { PropertiesPanel } from "@/components/PropertiesPanel"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader } from "@/components/ui/card"
import { Box, GripVertical, X, ChevronDown, ChevronRight } from "lucide-react"
import type { ModelObject, ToolCategory } from "@/types/model-object"
import * as THREE from "three"
// extend(THREE)

function Scene({
  objects,
  selectedObject,
  onSelectObject,
  demoMode,
  prompt,
}: {
  objects: ModelObject[]
  selectedObject: string | null
  onSelectObject: (id: string | null) => void
  demoMode?: boolean
  prompt?: string
}) {
  const meshRefs = useRef<Record<string, THREE.Mesh>>({})
  const nebulaGlowRef = useRef<THREE.PointLight>(null)

  useEffect(() => {
    if (nebulaGlowRef.current && prompt?.includes("nebula")) {
      nebulaGlowRef.current.intensity = 0.7
    } else if (nebulaGlowRef.current) {
      nebulaGlowRef.current.intensity = 0.5
    }
  }, [prompt])

  useFrame((state) => {
    if (demoMode && objects.length > 0) {
      objects.forEach((obj) => {
        const mesh = meshRefs.current[obj.id]
        if (mesh) {
          mesh.rotation.y += 0.005
        }
      })
    }
  })

  return (
    <>
      <Environment preset="studio" />
      <ambientLight intensity={0.5} />
      <directionalLight position={[10, 10, 5]} intensity={1} />
      <Grid
        args={[20, 20]}
        cellSize={1}
        cellThickness={0.5}
        cellColor="#6366f1"
        sectionSize={5}
        sectionThickness={1}
        sectionColor="#8b5cf6"
        fadeDistance={25}
        fadeStrength={1}
        followCamera={false}
        infiniteGrid={true}
      />
      <pointLight
        ref={nebulaGlowRef}
        color={new THREE.Color().setHSL(270 / 360, 0.6, 0.35)} // nebula-purple
        intensity={0.5}
        distance={50}
        position={[-5, -5, 5]}
      />
      {objects.map((obj) => (
        <ModelPrimitive
          key={obj.id}
          object={obj}
          isSelected={selectedObject === obj.id}
          onSelect={() => onSelectObject(obj.id)}
          meshRef={(ref) => {
            if (ref) {
              meshRefs.current[obj.id] = ref
            }
          }}
          prompt={prompt}
        />
      ))}
      <OrbitControls enablePan={true} enableZoom={true} enableRotate={true} />
      <GizmoHelper alignment="bottom-right" margin={[80, 80]}>
        <GizmoViewport axisColors={["#9d4b4b", "#2f7f32", "#3b5b9d"]} labelColor="white" />
      </GizmoHelper>
    </>
  )
}

function ModelPrimitive({
  object,
  isSelected,
  onSelect,
  meshRef,
  prompt,
}: {
  object: ModelObject
  isSelected: boolean
  onSelect: () => void
  meshRef: (ref: THREE.Mesh | null) => void
  prompt?: string
}) {
  const renderGeometry = () => {
    switch (object.type) {
      case "box":
        return <boxGeometry args={[1, 1, 1]} />
      case "sphere":
        return <sphereGeometry args={[0.5, 32, 32]} />
      case "cylinder":
        return <cylinderGeometry args={[0.5, 0.5, 1, 32]} />
      case "cone":
        return <coneGeometry args={[0.5, 1, 32]} />
      case "torus":
        return <torusGeometry args={[0.5, 0.2, 16, 100]} />
      case "plane":
        return <planeGeometry args={[1, 1]} />
      case "wedge":
        return <boxGeometry args={[1, 0.5, 1]} />
      default:
        return <boxGeometry args={[1, 1, 1]} />
    }
  }

  const getMaterialColor = () => {
    if (prompt?.includes("star")) {
      return new THREE.Color().setHSL(30 / 360, 0.7, 0.6) // star-yellow
    }
    return new THREE.Color(object.color)
  }

  return (
    <mesh
      name={object.id}
      position={object.position}
      rotation={object.rotation}
      scale={object.scale}
      onClick={onSelect}
      ref={meshRef}
    >
      {renderGeometry()}
      <meshStandardMaterial
        color={getMaterialColor()}
        wireframe={isSelected}
        transparent={isSelected}
        opacity={isSelected ? 0.7 : 1}
      />
      {isSelected && (
        <mesh position={[0, 0, 0]} scale={[1.05, 1.05, 1.05]}>
          {renderGeometry()}
          <meshBasicMaterial color="#00ffff" wireframe transparent opacity={0.3} />
        </mesh>
      )}
    </mesh>
  )
}

function MovableToolbar({
  categories,
  onToolAction,
  onToggleCategory,
  isVisible,
  onToggleVisibility,
  position,
  onPositionChange,
}: {
  categories: ToolCategory[]
  onToolAction: (action: string) => void
  onToggleCategory: (categoryName: string) => void
  isVisible: boolean
  onToggleVisibility: () => void
  position: { x: number; y: number }
  onPositionChange: (position: { x: number; y: number }) => void
}) {
  const [isDragging, setIsDragging] = useState(false)
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 })
  const toolbarRef = useRef<HTMLDivElement>(null)

  const handleMouseDown = (e: React.MouseEvent) => {
    const rect = toolbarRef.current?.getBoundingClientRect()
    if (rect) {
      setDragOffset({
        x: e.clientX - rect.left,
        y: e.clientY - rect.top,
      })
      setIsDragging(true)
    }
  }

  const handleMouseMove = useCallback(
    (e: MouseEvent) => {
      if (isDragging) {
        onPositionChange({
          x: e.clientX - dragOffset.x,
          y: e.clientY - dragOffset.y,
        })
      }
    },
    [isDragging, dragOffset, onPositionChange],
  )

  const handleMouseUp = useCallback(() => {
    setIsDragging(false)
  }, [])

  useEffect(() => {
    if (isDragging) {
      document.addEventListener("mousemove", handleMouseMove)
      document.addEventListener("mouseup", handleMouseUp)
      return () => {
        document.removeEventListener("mousemove", handleMouseMove)
        document.removeEventListener("mouseup", handleMouseUp)
      }
    }
  }, [isDragging, handleMouseMove, handleMouseUp])

  if (!isVisible) {
    return (
      <Button
        onClick={onToggleVisibility}
        className="fixed top-20 left-4 z-50 rounded-full w-12 h-12 btn-logo-gradient text-white shadow-lg border-0"
        style={{ left: `${position.x}px`, top: `${position.y}px` }}
        aria-label="Open Toolbar"
      >
        <Box className="h-6 w-6" />
      </Button>
    )
  }

  return (
    <Card
      ref={toolbarRef}
      className="fixed z-50 bg-slate-800/95 backdrop-blur-sm border border-slate-700 rounded-lg shadow-2xl max-h-[80vh] overflow-y-auto"
      style={{ left: `${position.x}px`, top: `${position.y}px`, minWidth: "200px" }}
    >
      <CardHeader
        className="flex items-center justify-between space-y-0 p-2 border-b border-slate-700 cursor-move bg-logo-gradient"
        onMouseDown={handleMouseDown}
      >
        <div className="flex items-center space-x-2">
          <GripVertical className="h-4 w-4 text-white cursor-grab" />
          <span className="text-sm font-medium text-white">Tools</span>
        </div>
        <Button
          variant="ghost"
          size="icon"
          onClick={onToggleVisibility}
          className="h-6 w-6 p-0 text-white hover:bg-white/20"
          aria-label="Close Toolbar"
        >
          <X className="h-3 w-3" />
        </Button>
      </CardHeader>
      <CardContent className="p-2 space-y-1">
        {categories.map((category) => (
          <div key={category.name}>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => onToggleCategory(category.name)}
              className="w-full justify-start text-gray-300 hover:text-white hover:bg-slate-700"
            >
              {category.expanded ? <ChevronDown className="h-3 w-3 mr-2" /> : <ChevronRight className="h-3 w-3 mr-2" />}
              {category.icon}
              <span className="ml-2 text-sm">{category.name}</span>
            </Button>
            {category.expanded && (
              <div className="ml-4 mt-1 space-y-1">
                {category.tools.map((tool) => (
                  <Button
                    key={tool.action}
                    variant="ghost"
                    size="sm"
                    onClick={() => onToolAction(tool.action)}
                    className="w-full justify-start text-gray-400 hover:text-white hover:bg-slate-700 text-xs"
                  >
                    {tool.icon}
                    <span className="ml-2">{tool.name}</span>
                  </Button>
                ))}
              </div>
            )}
          </div>
        ))}
      </CardContent>
    </Card>
  )
}

export function ModelViewer({
  demoMode = false,
  darkMode = false,
  prompt = "A rotating asteroid in space",
}: {
  demoMode?: boolean
  darkMode?: boolean
  prompt?: string
}) {
  const {
    objects,
    selectedObject,
    setSelectedObject,
    addObject,
    deleteSelected,
    updatePosition,
    updateScale,
    updateColor,
    resetCamera,
    toolCategories,
    setToolCategories,
    addToHistory,
    undo,
    redo,
  } = useModelViewer()

  const [isToolbarVisible, setIsToolbarVisible] = useState(!demoMode)
  const [toolbarPosition, setToolbarPosition] = useState({ x: 20, y: 100 })

  const handleToolAction = useCallback(
    (action: string) => {
      if (action.startsWith("add-")) {
        const type = action.replace("add-", "") as ModelObject["type"]
        addObject(type)
      } else {
        console.log("Tool action:", action)
      }
    },
    [addObject],
  )

  const toggleCategory = useCallback(
    (categoryName: string) => {
      setToolCategories((prev) =>
        prev.map((cat) => (cat.name === categoryName ? { ...cat, expanded: !cat.expanded } : cat)),
      )
    },
    [setToolCategories],
  )

  const selected = selectedObject ? objects.find((obj) => obj.id === selectedObject) : null

  const handleObjectUpdate = useCallback(
    (update: Partial<ModelObject>) => {
      if (!selectedObject) return

      const index = objects.findIndex((obj) => obj.id === selectedObject)
      if (index !== -1) {
        const newObj = { ...objects[index], ...update, updatedAt: Date.now() }
        const newObjs = [...objects]
        newObjs[index] = newObj
        addToHistory(newObjs)
      }
    },
    [selectedObject, objects, addToHistory],
  )

  return (
    <div className="h-full flex relative bg-cosmic-gradient">
      {!demoMode && (
        <MovableToolbar
          categories={toolCategories}
          onToolAction={handleToolAction}
          onToggleCategory={toggleCategory}
          isVisible={isToolbarVisible}
          onToggleVisibility={() => setIsToolbarVisible(!isToolbarVisible)}
          position={toolbarPosition}
          onPositionChange={setToolbarPosition}
        />
      )}
      <div className="flex-1 relative">
        <Canvas
          camera={{ position: [5, 5, 5], fov: 60 }}
          style={{
            background: darkMode ? "#000000" : "#000000",
          }}
        >
          <Suspense fallback={null}>
            <Scene
              objects={objects}
              selectedObject={selectedObject}
              onSelectObject={setSelectedObject}
              demoMode={demoMode}
              prompt={prompt}
            />
          </Suspense>
        </Canvas>
        {!demoMode && (
          <div className="absolute bottom-4 left-4 bg-slate-800/90 backdrop-blur-sm rounded-lg px-4 py-2 text-sm text-gray-300">
            <div className="flex items-center space-x-4">
              <span>Objects: {objects.length}</span>
              {selected && <span>Selected: {selected.type}</span>}
            </div>
          </div>
        )}
        {selected && !demoMode && (
          <div className="absolute top-4 left-1/2 transform -translate-x-1/2 bg-logo-gradient text-white px-3 py-1 rounded-full text-sm font-medium">
            Selected: {selected.type} ({selected.id})
          </div>
        )}
      </div>
      {!demoMode && (
        <div className="w-64 bg-slate-800 border-l border-slate-700 p-4 overflow-y-auto">
          <h3 className="text-lg font-semibold text-white mb-4">Properties</h3>
          <PropertiesPanel object={selected} onUpdate={handleObjectUpdate} onDelete={deleteSelected} />
          <div className="mt-8 pt-4 border-t border-slate-700">
            <h4 className="text-md font-medium text-white mb-2">Scene Info</h4>
            <div className="text-sm text-gray-300 space-y-1">
              <p>Objects: {objects.length}</p>
              <p>Triangles: {objects.length * 12}</p>
              <p>Vertices: {objects.length * 8}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}