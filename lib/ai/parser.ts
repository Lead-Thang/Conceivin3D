import type { UseModelViewerReturn } from "@/hooks/use-model-viewer"
import type { ModelObject } from "@/types/model-object"

export interface ParsedResult {
  type: "text" | "command"
  text?: string
  action?: string
  params?: Record<string, any>
}

export function parseAIResponse(text: string, viewerActions: UseModelViewerReturn): ParsedResult {
  const lowerText = text.toLowerCase()

  // Create/Add commands
  if (lowerText.includes("create") || lowerText.includes("add")) {
    const objectType = extractObjectType(lowerText)
    const position = extractPosition(lowerText)
    const color = extractColor(lowerText)

    return {
      type: "command",
      action: "add",
      params: {
        type: objectType || "box",
        position: position || [0, 0.5, 0],
        color: color || "#8b5cf6",
      },
    }
  }

  // Delete commands
  if (lowerText.includes("delete") || lowerText.includes("remove")) {
    return {
      type: "command",
      action: "delete",
    }
  }

  // Scale commands
  if (lowerText.includes("scale up") || lowerText.includes("bigger") || lowerText.includes("larger")) {
    const factor = extractScaleFactor(lowerText) || 1.2
    return {
      type: "command",
      action: "update",
      params: {
        scale: [factor, factor, factor],
      },
    }
  }

  if (lowerText.includes("scale down") || lowerText.includes("smaller")) {
    const factor = extractScaleFactor(lowerText) || 0.8
    return {
      type: "command",
      action: "update",
      params: {
        scale: [factor, factor, factor],
      },
    }
  }

  // Color commands
  const color = extractColor(lowerText)
  if (color) {
    return {
      type: "command",
      action: "update",
      params: { color },
    }
  }

  // Move commands
  if (lowerText.includes("move") || lowerText.includes("position")) {
    const position = extractPosition(lowerText)
    if (position) {
      return {
        type: "command",
        action: "update",
        params: { position },
      }
    }
  }

  // Fallback to text feedback
  return {
    type: "text",
    text,
  }
}

// Helper functions
function extractObjectType(text: string): ModelObject["type"] | undefined {
  if (text.includes("box") || text.includes("cube")) return "box"
  if (text.includes("sphere") || text.includes("ball")) return "sphere"
  if (text.includes("cylinder")) return "cylinder"
  if (text.includes("cone")) return "cone"
  if (text.includes("torus") || text.includes("donut")) return "torus"
  if (text.includes("plane") || text.includes("flat")) return "plane"
  if (text.includes("wedge")) return "wedge"
  return undefined
}

function extractPosition(text: string): [number, number, number] | null {
  // Look for patterns like "at [1, 2, 3]" or "position 1 2 3"
  const bracketMatch = text.match(/\[([0-9.,\s-]+)\]/i)
  if (bracketMatch) {
    const coords = bracketMatch[1].split(",").map((s) => Number.parseFloat(s.trim()))
    if (coords.length === 3 && coords.every((n) => !isNaN(n))) {
      return coords as [number, number, number]
    }
  }

  // Look for "at x y z" pattern
  const coordMatch = text.match(/(?:at|position)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)/i)
  if (coordMatch) {
    return [Number.parseFloat(coordMatch[1]), Number.parseFloat(coordMatch[2]), Number.parseFloat(coordMatch[3])]
  }

  return null
}

function extractColor(text: string): string | null {
  const colorMap: Record<string, string> = {
    red: "#ef4444",
    blue: "#3b82f6",
    green: "#10b981",
    yellow: "#f59e0b",
    purple: "#8b5cf6",
    orange: "#f97316",
    pink: "#ec4899",
    cyan: "#06b6d4",
    white: "#ffffff",
    black: "#000000",
    gray: "#6b7280",
    grey: "#6b7280",
  }

  for (const [colorName, colorValue] of Object.entries(colorMap)) {
    if (text.includes(colorName)) {
      return colorValue
    }
  }

  // Look for hex colors
  const hexMatch = text.match(/#[0-9a-fA-F]{6}/i)
  if (hexMatch) {
    return hexMatch[0]
  }

  return null
}

function extractScaleFactor(text: string): number | null {
  // Look for percentages like "20%" or "by 20%"
  const percentMatch = text.match(/(\d+)%/i)
  if (percentMatch) {
    return 1 + Number.parseFloat(percentMatch[1]) / 100
  }

  // Look for factors like "by 2" or "2x"
  const factorMatch = text.match(/(?:by\s+|x\s*)(\d+(?:\.\d+)?)/i)
  if (factorMatch) {
    return Number.parseFloat(factorMatch[1])
  }

  return null
}
