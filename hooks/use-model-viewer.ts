"use client"

import type React from "react"
import { useState, useCallback } from "react"
import {
  Box,
  Cylinder,
  Move3D,
  RotateCcw,
  Scale,
  Ruler,
  Square,
  Circle,
  Triangle,
  Copy,
  CornerDownRight,
  Layers,
  Palette,
} from "lucide-react"
import type { ModelObject, ToolCategory } from "@/types/model-object"
import { TOOL_CATEGORIES } from "@/types/tool-category"

const defaultToolCategories: ToolCategory[] = [
  {
    name: TOOL_CATEGORIES.CREATION,
    icon: <Box className="h-4 w-4" />,
    expanded: true,
    tools: [
      { name: "Box", icon: <Box className="h-3 w-3" />, action: "add-box" },
      { name: "Sphere", icon: <Circle className="h-3 w-3" />, action: "add-sphere" },
      { name: "Cylinder", icon: <Cylinder className="h-3 w-3" />, action: "add-cylinder" },
      { name: "Cone", icon: <Triangle className="h-3 w-3" />, action: "add-cone" },
      { name: "Torus", icon: <Circle className="h-3 w-3" />, action: "add-torus" },
      { name: "Plane", icon: <Square className="h-3 w-3" />, action: "add-plane" },
    ],
  },
  {
    name: TOOL_CATEGORIES.TRANSFORMATION,
    icon: <Move3D className="h-4 w-4" />,
    expanded: false,
    tools: [
      { name: "Move", icon: <Move3D className="h-3 w-3" />, action: "move" },
      { name: "Rotate", icon: <RotateCcw className="h-3 w-3" />, action: "rotate" },
      { name: "Scale", icon: <Scale className="h-3 w-3" />, action: "scale" },
    ],
  },
  {
    name: TOOL_CATEGORIES.MEASUREMENT,
    icon: <Ruler className="h-4 w-4" />,
    expanded: false,
    tools: [
      { name: "Distance", icon: <Ruler className="h-3 w-3" />, action: "measure-distance" },
      { name: "Area", icon: <Square className="h-3 w-3" />, action: "measure-area" },
      { name: "Volume", icon: <Layers className="h-3 w-3" />, action: "measure-volume" },
    ],
  },
  {
    name: TOOL_CATEGORIES.HISTORY,
    icon: <Copy className="h-4 w-4" />,
    expanded: false,
    tools: [
      { name: "Undo", icon: <CornerDownRight className="h-3 w-3" />, action: "undo" },
      { name: "Redo", icon: <CornerDownRight className="h-3 w-3" />, action: "redo" },
      { name: "Delete", icon: <Copy className="h-3 w-3" />, action: "delete-selected" },
    ],
  },
]

export interface UseModelViewerReturn {
  objects: ModelObject[]
  selectedObject: string | null
  setSelectedObject: (id: string | null) => void
  addObject: (type: ModelObject["type"]) => void
  deleteSelected: () => void
  updatePosition: (axis: "x" | "y" | "z", value: number) => void
  updateScale: (axis: "x" | "y" | "z", value: number) => void
  updateColor: (color: string) => void
  resetCamera: () => void
  undo: () => void
  redo: () => void
  toolCategories: ToolCategory[]
  setToolCategories: React.Dispatch<React.SetStateAction<ToolCategory[]>>
  addToHistory: (newState: ModelObject[]) => void
  history: ModelObject[][]
  historyIndex: number
}

export function useModelViewer(): UseModelViewerReturn {
  const [objects, setObjects] = useState<ModelObject[]>([
    {
      id: "1",
      type: "box",
      position: [0, 0.5, 0],
      rotation: [0, 0, 0],
      scale: [1, 1, 1],
      color: "#8b5cf6",
      createdAt: Date.now(),
      updatedAt: Date.now(),
    },
  ])

  const [history, setHistory] = useState<ModelObject[][]>([])
  const [historyIndex, setHistoryIndex] = useState(-1)
  const [selectedObject, setSelectedObject] = useState<string | null>(null)
  const [toolCategories, setToolCategories] = useState(defaultToolCategories)

  // History helpers
  const addToHistory = useCallback(
    (newState: ModelObject[]) => {
      setHistory((prev) => [...prev.slice(0, historyIndex + 1), newState])
      setHistoryIndex((idx) => idx + 1)
    },
    [historyIndex],
  )

  const undo = useCallback(() => {
    if (historyIndex >= 0) {
      setHistoryIndex((idx) => idx - 1)
      setObjects([...history[historyIndex]])
    }
  }, [historyIndex, history])

  const redo = useCallback(() => {
    if (historyIndex < history.length - 1) {
      setHistoryIndex((idx) => idx + 1)
      setObjects([...history[historyIndex + 1]])
    }
  }, [historyIndex, history])

  // Add new object
  const addObject = useCallback(
    (type: ModelObject["type"]) => {
      const newObject: ModelObject = {
        id: Date.now().toString(),
        type,
        position: [Math.random() * 4 - 2, 0.5, Math.random() * 4 - 2],
        rotation: [0, 0, 0],
        scale: [1, 1, 1],
        color: `hsl(${Math.random() * 360}, 70%, 60%)`,
        createdAt: Date.now(),
        updatedAt: Date.now(),
      }

      setObjects((prev) => [...prev, newObject])
      addToHistory([...objects, newObject])
      setSelectedObject(newObject.id)
    },
    [objects, addToHistory],
  )

  // Delete selected object
  const deleteSelected = useCallback(() => {
    if (selectedObject) {
      const newObjects = objects.filter((obj) => obj.id !== selectedObject)
      setObjects(newObjects)
      addToHistory(newObjects)
      setSelectedObject(null)
    }
  }, [selectedObject, objects, addToHistory])

  // Update position
  const updatePosition = useCallback(
    (axis: "x" | "y" | "z", value: number) => {
      if (!selectedObject) return

      const index = objects.findIndex((obj) => obj.id === selectedObject)
      if (index !== -1) {
        const newObj = { ...objects[index] }
        newObj.position = newObj.position.map((v, i) => (i === ["x", "y", "z"].indexOf(axis) ? value : v)) as [
          number,
          number,
          number,
        ]
        newObj.updatedAt = Date.now()

        const newObjs = [...objects]
        newObjs[index] = newObj
        setObjects(newObjs)
        addToHistory(newObjs)
      }
    },
    [selectedObject, objects, addToHistory],
  )

  // Update scale
  const updateScale = useCallback(
    (axis: "x" | "y" | "z", value: number) => {
      if (!selectedObject) return

      const index = objects.findIndex((obj) => obj.id === selectedObject)
      if (index !== -1) {
        const newObj = { ...objects[index] }
        newObj.scale = newObj.scale.map((v, i) => (i === ["x", "y", "z"].indexOf(axis) ? Math.max(0.1, value) : v)) as [
          number,
          number,
          number,
        ]
        newObj.updatedAt = Date.now()

        const newObjs = [...objects]
        newObjs[index] = newObj
        setObjects(newObjs)
        addToHistory(newObjs)
      }
    },
    [selectedObject, objects, addToHistory],
  )

  // Update color
  const updateColor = useCallback(
    (color: string) => {
      if (!selectedObject) return

      const index = objects.findIndex((obj) => obj.id === selectedObject)
      if (index !== -1) {
        const newObj = { ...objects[index], color, updatedAt: Date.now() }
        const newObjs = [...objects]
        newObjs[index] = newObj
        setObjects(newObjs)
        addToHistory(newObjs)
      }
    },
    [selectedObject, objects, addToHistory],
  )

  // Reset camera view
  const resetCamera = useCallback(() => {
    console.log("Resetting camera...")
  }, [])

  return {
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
    history,
    historyIndex,
    addToHistory,
    undo,
    redo,
  }
}
