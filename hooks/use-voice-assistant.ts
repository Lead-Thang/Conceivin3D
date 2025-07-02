"use client"

import { useState, useCallback } from "react"

export function useVoiceAssistant() {
  const [audioUrl, setAudioUrl] = useState<string | null>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const speak = useCallback(async (text: string) => {
    if (!text.trim()) return

    setIsLoading(true)
    setError(null)

    try {
      const response = await fetch("/api/text-to-speech", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text }),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const audioBlob = await response.blob()
      const audioUrl = URL.createObjectURL(audioBlob)
      setAudioUrl(audioUrl)

      const audio = new Audio(audioUrl)
      setIsPlaying(true)

      audio.play()
      audio.onended = () => {
        setIsPlaying(false)
        URL.revokeObjectURL(audioUrl) // Clean up memory
      }
      audio.onerror = () => {
        setIsPlaying(false)
        setError("Failed to play audio")
      }
    } catch (err) {
      console.error("Error generating speech:", err)
      setError(err instanceof Error ? err.message : "Unknown error occurred")
    } finally {
      setIsLoading(false)
    }
  }, [])

  const stop = useCallback(() => {
    setIsPlaying(false)
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl)
      setAudioUrl(null)
    }
  }, [audioUrl])

  return {
    speak,
    stop,
    audioUrl,
    isPlaying,
    isLoading,
    error,
  }
}
