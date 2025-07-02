"use client"

import { InputWithHelper } from "@/components/ui/input-with-helper"
import { ColorInput } from "@/components/ui/color-input"
import { Button } from "@/components/ui/button"
import type { ModelObject } from "@/types/model-object"

interface PropertiesPanelProps {
  object?: ModelObject | null
  onUpdate?: (update: Partial<ModelObject>) => void
  onDelete?: () => void
}

// Helper function to update array values
function updateArray(arr: [number, number, number], axis: "x" | "y" | "z", value: number): [number, number, number] {
  const index = axis === "x" ? 0 : axis === "y" ? 1 : 2
  const newArr = [...arr] as [number, number, number]
  newArr[index] = value
  return newArr
}

export function PropertiesPanel({ object, onUpdate, onDelete }: PropertiesPanelProps) {
  if (!object || !onUpdate) {
    return (
      <div className="p-4 text-gray-500 text-sm" role="status" aria-label="No object selected for editing">
        Select an object to edit properties
      </div>
    )
  }

  const handlePositionChange = (axis: "x" | "y" | "z", value: number) => {
    onUpdate({ position: updateArray(object.position, axis, value) })
  }

  const handleScaleChange = (axis: "x" | "y" | "z", value: number) => {
    onUpdate({ scale: updateArray(object.scale, axis, value) })
  }

  const handleRotationChange = (axis: "x" | "y" | "z", value: number) => {
    onUpdate({ rotation: updateArray(object.rotation, axis, value) })
  }

  const handleColorChange = (color: string) => {
    onUpdate({ color })
  }

  const handleReset = () => {
    onUpdate({
      position: [0, 0.5, 0],
      rotation: [0, 0, 0],
      scale: [1, 1, 1],
      color: "#8b5cf6", // Default to a space-inspired purple
    })
  }

  return (
    <div className="space-y-6 h-full flex flex-col bg-nebula-purple/50 backdrop-blur-sm p-4 rounded-lg shadow-lg">
      {/* Object Type */}
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-1" htmlFor="object-type">
          Type
        </label>
        <p
          id="object-type"
          className="text-white capitalize bg-nebula-purple/70 px-3 py-1 rounded hover:bg-nebula-purple/80 transition-all duration-300"
        >
          {object.type}
        </p>
      </div>

      {/* Position */}
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">Position</label>
        <div className="grid grid-cols-3 gap-2">
          <InputWithHelper
            label="X"
            helperText="Move along X-axis"
            type="number"
            step="0.1"
            value={object.position[0].toFixed(2)}
            onChange={(e) => handlePositionChange("x", Number.parseFloat(e.target.value) || 0)}
            onKeyDown={(e) => {
              if (e.key === "Enter") e.currentTarget.blur()
            }}
            aria-label="Adjust X-axis position"
          />
          <InputWithHelper
            label="Y"
            helperText="Move along Y-axis"
            type="number"
            step="0.1"
            value={object.position[1].toFixed(2)}
            onChange={(e) => handlePositionChange("y", Number.parseFloat(e.target.value) || 0)}
            onKeyDown={(e) => {
              if (e.key === "Enter") e.currentTarget.blur()
            }}
            aria-label="Adjust Y-axis position"
          />
          <InputWithHelper
            label="Z"
            helperText="Move along Z-axis"
            type="number"
            step="0.1"
            value={object.position[2].toFixed(2)}
            onChange={(e) => handlePositionChange("z", Number.parseFloat(e.target.value) || 0)}
            onKeyDown={(e) => {
              if (e.key === "Enter") e.currentTarget.blur()
            }}
            aria-label="Adjust Z-axis position"
          />
        </div>
      </div>

      {/* Rotation */}
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">Rotation (degrees)</label>
        <div className="grid grid-cols-3 gap-2">
          <InputWithHelper
            label="X"
            helperText="Rotate around X-axis"
            type="number"
            step="1"
            value={Math.round((object.rotation[0] * 180) / Math.PI)}
            onChange={(e) => handleRotationChange("x", ((Number.parseFloat(e.target.value) || 0) * Math.PI) / 180)}
            aria-label="Adjust X-axis rotation"
          />
          <InputWithHelper
            label="Y"
            helperText="Rotate around Y-axis"
            type="number"
            step="1"
            value={Math.round((object.rotation[1] * 180) / Math.PI)}
            onChange={(e) => handleRotationChange("y", ((Number.parseFloat(e.target.value) || 0) * Math.PI) / 180)}
            aria-label="Adjust Y-axis rotation"
          />
          <InputWithHelper
            label="Z"
            helperText="Rotate around Z-axis"
            type="number"
            step="1"
            value={Math.round((object.rotation[2] * 180) / Math.PI)}
            onChange={(e) => handleRotationChange("z", ((Number.parseFloat(e.target.value) || 0) * Math.PI) / 180)}
            aria-label="Adjust Z-axis rotation"
          />
        </div>
      </div>

      {/* Scale */}
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">Scale</label>
        <div className="grid grid-cols-3 gap-2">
          <InputWithHelper
            label="X"
            helperText="Scale on X-axis"
            type="number"
            step="0.1"
            value={object.scale[0].toFixed(2)}
            onChange={(e) => handleScaleChange("x", Math.max(0.1, Number.parseFloat(e.target.value) || 1))}
            aria-label="Adjust X-axis scale"
          />
          <InputWithHelper
            label="Y"
            helperText="Scale on Y-axis"
            type="number"
            step="0.1"
            value={object.scale[1].toFixed(2)}
            onChange={(e) => handleScaleChange("y", Math.max(0.1, Number.parseFloat(e.target.value) || 1))}
            aria-label="Adjust Y-axis scale"
          />
          <InputWithHelper
            label="Z"
            helperText="Scale on Z-axis"
            type="number"
            step="0.1"
            value={object.scale[2].toFixed(2)}
            onChange={(e) => handleScaleChange("z", Math.max(0.1, Number.parseFloat(e.target.value) || 1))}
            aria-label="Adjust Z-axis scale"
          />
        </div>
      </div>

      {/* Color */}
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2" htmlFor="object-color">
          Color
        </label>
        <ColorInput
          id="object-color"
          value={object.color}
          onChange={(e) => handleColorChange(e.target.value)}
          helperText="Click to change object color"
          aria-label="Select object color"
        />
      </div>

      {/* Object Info */}
      <div className="text-xs text-gray-400 bg-nebula-purple/70 p-2 rounded hover:bg-nebula-purple/80 transition-all duration-300">
        <p>ID: {object.id}</p>
        <p>Created: {new Date(object.createdAt).toLocaleTimeString()}</p>
        {object.updatedAt !== object.createdAt && (
          <p>Modified: {new Date(object.updatedAt).toLocaleTimeString()}</p>
        )}
      </div>

      {/* Actions */}
      <div className="pt-4 border-t border-nebula-purple/20 flex space-x-2 mt-auto">
        <Button
          variant="outline"
          size="sm"
          onClick={handleReset}
          className="flex-1 border-nebula-purple/30 text-nebula-purple hover:bg-nebula-purple/10 hover:scale-105 transition-all duration-300"
          aria-label="Reset object properties to default"
        >
          Reset
        </Button>
        <Button
          variant="destructive"
          size="sm"
          onClick={onDelete}
          className="flex-1 hover:scale-105 transition-all duration-300"
          aria-label="Delete selected object"
        >
          Delete
        </Button>
      </div>
    </div>
  )
}