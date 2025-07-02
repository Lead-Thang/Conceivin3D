"use client"

import { useState, useRef, useEffect } from "react"
import { Button } from "./button"
import { Mic, Volume2 } from "lucide-react"

export function VoiceAssistantButton() {
  const [isListening, setIsListening] = useState(false)
  const recognition = useRef<any>(null)

  useEffect(() => {
    if ("webkitSpeechRecognition" in window) {
      recognition.current = new (window as any).webkitSpeechRecognition()
      recognition.current.continuous = true
      recognition.current.interimResults = true
      recognition.current.lang = "en-US"
    } else {
      console.warn("Web Speech API not supported")
    }
  }, [])

  const startListening = () => {
    if (!recognition.current) return
    recognition.current.start()
    setIsListening(true)
  }

  const stopListening = async () => {
    if (!recognition.current) return
    recognition.current.stop()
    setIsListening(false)

    recognition.current.onresult = (event: any) => {
      let finalTranscript = ""
      for (let i = event.resultIndex; i < event.results.length; ++i) {
        finalTranscript += event.results[i][0].transcript
      }
      handleAICommand(finalTranscript)
    }
  }

  const handleAICommand = async (command: string) => {
    try {
      const result = await fetch("/api/elevenlabs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: command }),
      })

      const data = await result.json()
      console.log("ElevenLabs response:", data)
      
      if (data.command) {
        applyToolCommand(data.command)
      }
    } catch (err) {
      console.error("Voice assistant error:", err)
    }
  }

  const applyToolCommand = (command: ToolCall) => {
    switch (command.name) {
      case "add":
        addObject(command.args.type)
        break
      case "delete":
        deleteSelected()
        break
      case "scale":
        updateScale(command.args.axis, command.args.value)
        break
      case "position":
        updatePosition(command.args.axis, command.args.value)
        break
      case "color":
        updateColor(command.args.color)
        break
      default:
        console.log("Unknown command:", command)
    }
  }

  return (
    <Button
      variant="outline"
      size="icon"
      className={`ml-2 ${isListening ? "bg-red-500/20 ring-2 ring-red-400" : ""}`}
      onClick={isListening ? stopListening : startListening}
      aria-label={isListening ? "Stop Listening" : "Start Voice Assistant"}
    >
      {isListening ? <Volume2 className="h-4 w-4 animate-pulse" /> : <Mic className="h-4 w-4" />}
    </Button>
  )
}
