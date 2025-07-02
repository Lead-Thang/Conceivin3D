// lib/ai/elevenlabs.ts
import type { ToolCall } from "@/types/ai-tools"

export async function callElevenLabsAgent(text: string): Promise<{
  text: string
  audioUrl?: string
  command?: ToolCall | null
}> {
  try {
    const response = await fetch("/api/elevenlabs", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text }),
    })

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`)
    }

    const data = await response.json()

    return {
      text: data.text || "",
      audioUrl: data.audio_url || "",
      command: parseAICommand(data.text || text),
    }
  } catch (error) {
    console.error("Error calling ElevenLabs:", error)
    return {
      text: "I couldn't understand that. Try again.",
      command: null,
    }
  }
}

/**
 * Parses AI-generated text into structured commands
 */
function parseAICommand(text: string): ToolCall | null {
  const lowerText = text.toLowerCase()

  // Add/Create commands
  if (lowerText.includes("add") || lowerText.includes("create")) {
    if (lowerText.includes("box") || lowerText.includes("cube")) {
      return { name: "add", args: { type: "box" } }
    }
    if (lowerText.includes("sphere") || lowerText.includes("ball")) {
      return { name: "add", args: { type: "sphere" } }
    }
    if (lowerText.includes("cylinder")) {
      return { name: "add", args: { type: "cylinder" } }
    }
    if (lowerText.includes("cone")) {
      return { name: "add", args: { type: "cone" } }
    }
    if (lowerText.includes("torus") || lowerText.includes("donut")) {
      return { name: "add", args: { type: "torus" } }
    }
    if (lowerText.includes("plane")) {
      return { name: "add", args: { type: "plane" } }
    }
  }

  // Delete commands
  if (lowerText.includes("delete") || lowerText.includes("remove")) {
    return { name: "delete", args: {} }
  }

  // Scale commands
  if (lowerText.includes("scale up") || lowerText.includes("bigger")) {
    return { name: "scale", args: { factor: 1.2 } }
  }
  if (lowerText.includes("scale down") || lowerText.includes("smaller")) {
    return { name: "scale", args: { factor: 0.8 } }
  }

  // Color commands
  if (lowerText.includes("red")) {
    return { name: "color", args: { color: "#ef4444" } }
  }
  if (lowerText.includes("blue")) {
    return { name: "color", args: { color: "#3b82f6" } }
  }
  if (lowerText.includes("green")) {
    return { name: "color", args: { color: "#10b981" } }
  }
  if (lowerText.includes("yellow")) {
    return { name: "color", args: { color: "#f59e0b" } }
  }
  if (lowerText.includes("purple")) {
    return { name: "color", args: { color: "#8b5cf6" } }
  }

  // View commands
  if (lowerText.includes("wireframe")) {
    return { name: "view", args: { mode: "wireframe" } }
  }
  if (lowerText.includes("shaded")) {
    return { name: "view", args: { mode: "shaded" } }
  }

  // Camera commands
  if (lowerText.includes("reset camera") || lowerText.includes("reset view")) {
    return { name: "reset", args: {} }
  }

  // History commands
  if (lowerText.includes("undo")) {
    return { name: "undo", args: {} }
  }
  if (lowerText.includes("redo")) {
    return { name: "redo", args: {} }
  }

  return null
}
