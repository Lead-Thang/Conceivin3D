"use client"

import type React from "react"
import { useState, useCallback } from "react"
import type { UseModelViewerReturn } from "@/hooks/use-model-viewer"
import axios from "axios"

export type AIProvider = {
  sendMessage(messages: Message[]): Promise<{ message: string; command?: any }>
}

export type Message = {
  id: string
  role: "user" | "assistant"
  content: string
}

export function setAISource(source: string) {
  console.log(`Switching AI source to: ${source}`)
  return true
}

export function useAIAssistant(viewerActions?: UseModelViewerReturn) {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      role: "assistant",
      content:
        "Hello! I'm your Conceivin3D assistant powered by PyTorch. I can help with 3D modeling and learn engineering knowledge. Try 'learn more' or ask about creating objects!",
    },
  ])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setInput(e.target.value)
    setError(null)
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

      setMessages((prev: Message[]) => [...prev, userMessage])
      setInput("")
      setIsLoading(true)
      setError(null)

      try {
        const response = await axios.post("http://localhost:8000/api/conceivo", {
          message: input,
          metrics: input.match(/\d+\.?\d*/g)?.map(Number) || [50000, 0.95, 150],
        })
        const { message, predicted_efficiency, command, new_knowledge } = response.data

        const assistantMessage: Message = {
          id: `assistant-${Date.now()}`,
          role: "assistant",
          content: predicted_efficiency
            ? `${message} (Predicted: ${predicted_efficiency})`
            : message + (new_knowledge ? `\nNew Knowledge: ${new_knowledge}` : ""),
        }

        setMessages((prev: Message[]) => [...prev, assistantMessage])

        if (viewerActions && command) {
          processCommand(command, viewerActions)
        }
      } catch (err) {
        console.error("AI Assistant error:", err)
        const errorMessage = err instanceof Error ? err.message : "Unknown error occurred"
        setError(errorMessage)

        const errorResponse: Message = {
          id: `error-${Date.now()}`,
          role: "assistant",
          content:
            "I'm sorry, I encountered an error. Please try again, or check the server with 'learn more' to refresh knowledge.",
        }
        setMessages((prev: Message[]) => [...prev, errorResponse])
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
    setMessages((prev: Message[]) => [...prev, commandMessage])

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