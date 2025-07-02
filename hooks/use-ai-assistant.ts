"use client"

import type React from "react"
import { useState, useCallback } from "react"
import type { UseModelViewerReturn } from "@/hooks/use-model-viewer"
import { getActiveAIProvider } from "../lib/ai-adapters-config"

let activeAIProvider: AIProvider = getActiveAIProvider()

export function setAISource(source: string) {
  // Implementation to update the AI source
  // This would typically involve updating some state or configuration
  // For now, we'll just log the change and return true
  console.log(`Switching AI source to: ${source}`)
  return true
}

// Define AIProvider type - this should match what's in ai-adapters-config
export type AIProvider = {
  sendMessage(messages: Message[]): Promise<{ message: string; command?: any }>
}

export function useAIAssistant(viewerActions?: UseModelViewerReturn) {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      role: "assistant",
      content:
        "Hello! I'm your Conceivin3D assistant. I can help with 3D modeling, measurements, and design suggestions. Try asking me about creating objects, changing properties, or using the tools!",
    },
  ])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setInput(e.target.value)
    setError(null) // Clear error when user starts typing
  }, [])

  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault()
      if (!input.trim() || isLoading) return

      const userMessage: Message = {
        id: `user-${Date.now()}`,
        role: "user",
        content: input.trim(),
      }

      setMessages((prev) => [...prev, userMessage])
      setInput("")
      setIsLoading(true)
      setError(null)

      try {
        const aiProvider = getActiveAIProvider()
        const response = await aiProvider.sendMessage([...messages, userMessage])

        const assistantMessage: Message = {
          id: `assistant-${Date.now()}`,
          role: "assistant",
          content: response.message,
        }

        setMessages((prev) => [...prev, assistantMessage])

        // Process any commands if viewer actions are available
        if (viewerActions && response.command) {
          processCommand(response.command, viewerActions)
        }
      } catch (err) {
        console.error("AI Assistant error:", err)
        const errorMessage = err instanceof Error ? err.message : "Unknown error occurred"
        setError(errorMessage)

        const errorResponse: Message = {
          id: `error-${Date.now()}`,
          role: "assistant",
          content:
            "I'm sorry, I encountered an error processing your request. Please try again, or ask me something else about 3D modeling.",
        }
        setMessages((prev) => [...prev, errorResponse])
      } finally {
        setIsLoading(false)
      }
    },
    [input, isLoading, messages, viewerActions],
  )

  const sendCommand = useCallback((command: string) => {
    const commandMessage: Message = {
      id: `cmd-${Date.now()}`,
      role: "user",
      content: command,
    }
    setMessages((prev) => [...prev, commandMessage])

    // Trigger the submit with the command
    setInput(command)
    setTimeout(() => {
      const form = document.querySelector("form") as HTMLFormElement
      if (form) {
        form.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }))
      }
    }, 100)
  }, [])

  const resetConversation = useCallback(() => {
    setMessages([
      {
        id: "welcome",
        role: "assistant",
        content: "Hello! I'm your Conceivin3D assistant. How can I help you with 3D modeling today?",
      },
    ])
    setError(null)
    setInput("")
  }, [])

  const processAIMessage = useCallback((messageContent: string) => {
    // This can be used for additional processing if needed
    console.log("Processing AI message:", messageContent)
  }, [])

  return {
    messages,
    input,
    handleInputChange,
    handleSubmit,
    isLoading,
    error,
    sendCommand,
    resetConversation,
    processAIMessage,
  }
}

function processCommand(command: any, viewerActions: UseModelViewerReturn) {
  try {
    switch (command.action) {
      case "add":
        viewerActions.addObject(command.params?.type || "box")
        break
      case "delete":
        viewerActions.deleteSelected()
        break
      case "update":
        if (command.params?.color) {
          viewerActions.updateColor(command.params.color)
        }
        if (command.params?.scale) {
          const [x, y, z] = command.params.scale
          viewerActions.updateScale("x", x)
          viewerActions.updateScale("y", y)
          viewerActions.updateScale("z", z)
        }
        break
      default:
        console.log("Unknown command:", command)
    }
  } catch (error) {
    console.error("Error processing command:", error)
  }
}
