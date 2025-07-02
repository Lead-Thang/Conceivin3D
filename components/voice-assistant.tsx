"use client"

import { useState, useEffect, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Mic, MicOff, Loader2, Volume2 } from "lucide-react"
import { useModelViewer } from "@/hooks/use-model-viewer"
import { callElevenLabsAgent } from "@/lib/ai/elevenlabs"
import { useToast } from "@/hooks/use-toast"
import type { ToolCall } from "@/types/ai-tools"

export function VoiceAssistant() {
  const [isListening, setIsListening] = useState(false)
  const [transcript, setTranscript] = useState("")
  const [isProcessing, setIsProcessing] = useState(false)
  const [isSupported, setIsSupported] = useState(false)
  const recognitionRef = useRef<any>(null)
  const { toast } = useToast()

  const viewerActions = useModelViewer()
  const { addObject, deleteSelected, updateColor, updateScale } = viewerActions

  // Check for Web Speech API support
  useEffect(() => {
    const supported = "webkitSpeechRecognition" in window || "SpeechRecognition" in window
    setIsSupported(supported)

    if (!supported) {
      console.warn("Web Speech API not supported in this browser")
      return
    }

    // Initialize Web Speech API
    const SpeechRecognition = (window as any).webkitSpeechRecognition || (window as any).SpeechRecognition
    recognitionRef.current = new SpeechRecognition()
    recognitionRef.current.continuous = false
    recognitionRef.current.interimResults = true
    recognitionRef.current.lang = "en-US"

    recognitionRef.current.onresult = (event: any) => {
      let current = ""
      for (let i = event.resultIndex; i < event.results.length; ++i) {
        current += event.results[i][0].transcript
      }
      setTranscript(current)
    }

    recognitionRef.current.onend = () => {
      setIsListening(false)
      if (transcript.trim()) {
        processVoiceCommand(transcript)
      }
    }

    recognitionRef.current.onerror = (event: any) => {
      console.error("Speech recognition error:", event.error)
      setIsListening(false)
      toast({
        title: "Voice Recognition Error",
        description: `Error: ${event.error}`,
        variant: "destructive",
      })
    }
  }, [transcript, toast])

  // Add a status check for ElevenLabs configuration
  useEffect(() => {
    // Test ElevenLabs connection on component mount
    const testConnection = async () => {
      try {
        const response = await fetch("/api/elevenlabs", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: "test" }),
        })

        if (response.ok) {
          console.log("✅ ElevenLabs connection successful")
        } else {
          console.warn("⚠️ ElevenLabs connection issue:", response.status)
        }
      } catch (error) {
        console.warn("⚠️ ElevenLabs test failed:", error)
      }
    }

    if (isSupported) {
      testConnection()
    }
  }, [isSupported])

  const processVoiceCommand = async (text: string) => {
    setIsProcessing(true)

    try {
      console.log("🎤 Processing voice command:", text)

      // Call ElevenLabs agent
      const result = await callElevenLabsAgent(text)

      // Execute command if available
      if (result.command) {
        dispatchCommand(result.command)
        toast({
          title: "✅ Voice Command Executed",
          description: `"${text}" → ${result.command.name}`,
        })
      } else {
        // Still show success even without a command
        toast({
          title: "🎤 Voice Processed",
          description: `Heard: "${text}"`,
        })
      }

      // Play audio response if available
      if (result.audioUrl) {
        const audio = new Audio(result.audioUrl)
        audio.play().catch(console.error)
      }
    } catch (error) {
      console.error("Voice command processing error:", error)
      toast({
        title: "❌ Voice Error",
        description: "Failed to process voice command",
        variant: "destructive",
      })
    } finally {
      setIsProcessing(false)
      setTranscript("")
    }
  }

  const dispatchCommand = (command: ToolCall) => {
    try {
      switch (command.name) {
        case "add":
          addObject(command.args?.type || "box")
          break
        case "delete":
          deleteSelected()
          break
        case "color":
          updateColor(command.args?.color || "#8b5cf6")
          break
        case "scale":
          const factor = command.args?.factor || 1.2
          updateScale("x", factor)
          updateScale("y", factor)
          updateScale("z", factor)
          break
        case "undo":
          viewerActions.undo()
          break
        case "redo":
          viewerActions.redo()
          break
        case "reset":
          viewerActions.resetCamera()
          break
        default:
          console.log("Unknown command:", command)
      }
    } catch (error) {
      console.error("Command execution error:", error)
      toast({
        title: "Command Error",
        description: "Failed to execute command",
        variant: "destructive",
      })
    }
  }

  const startListening = () => {
    if (!recognitionRef.current || !isSupported) return
    try {
      recognitionRef.current.start()
      setIsListening(true)
      setTranscript("")
    } catch (error) {
      console.error("Failed to start recognition:", error)
    }
  }

  const stopListening = () => {
    if (!recognitionRef.current) return
    try {
      recognitionRef.current.stop()
      setIsListening(false)
    } catch (error) {
      console.error("Failed to stop recognition:", error)
    }
  }

  if (!isSupported) {
    return (
      <div className="flex flex-col items-center space-y-2">
        <Button variant="outline" disabled className="border-gray-600 text-gray-500">
          <Volume2 className="h-4 w-4 mr-2" />
          Voice Not Supported
        </Button>
        <div className="text-xs text-gray-500 text-center max-w-xs">
          Voice commands require a modern browser with Web Speech API support
        </div>
      </div>
    )
  }

  return (
    <div className="flex flex-col items-center space-y-2">
      <Button
        variant="outline"
        className={`flex items-center gap-2 transition-all duration-300 ${
          isListening
            ? "bg-red-500/20 text-red-400 border-red-400 animate-pulse"
            : "border-logo-purple/30 text-logo-purple hover:bg-logo-purple/10"
        }`}
        onClick={isListening ? stopListening : startListening}
        disabled={isProcessing}
      >
        {isProcessing ? (
          <>
            <Loader2 className="h-4 w-4 animate-spin" /> Processing...
          </>
        ) : isListening ? (
          <>
            <MicOff className="h-4 w-4" /> Stop Listening
          </>
        ) : (
          <>
            <Mic className="h-4 w-4" /> Voice Command
          </>
        )}
      </Button>

      {transcript && (
        <div className="text-xs text-gray-400 max-w-xs text-center bg-black/50 px-2 py-1 rounded">"{transcript}"</div>
      )}

      {isListening && (
        <div className="text-xs text-logo-cyan text-center">
          🎤 Listening... Try: "Add a red cube" or "Make this blue"
        </div>
      )}
    </div>
  )
}
