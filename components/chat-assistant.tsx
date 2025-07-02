"use client"

import type React from "react"
import type { AIPlugin } from "../types/ai-plugin"
import { useRef, useEffect, useState } from "react"
import { Button } from "./ui/button"
import { Input } from "./ui/input"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { MessageCircle, X, Send, Bot, User, Minimize2, Maximize2, Loader2, Zap } from "lucide-react"
import { cn } from "../lib/utils"
import { useAIAssistant, setAISource } from "../hooks/use-ai-assistant"
import { useModelViewer } from "../hooks/use-model-viewer"
import { PluginManager } from "../../lib/plugin-manager"

export function ChatAssistant({ id }: { id?: string }) {
  const [isOpen, setIsOpen] = useState(true)
  const [isMinimized, setIsMinimized] = useState(false)
  const [activeSource, setActiveSource] = useState("chatgpt")
  const [plugins, setPlugins] = useState<Array<{ plugin: AIPlugin; enabled: boolean }>>([])

  const pluginManagerRef = useRef(new PluginManager())

  const { messages, input, handleInputChange, handleSubmit, isLoading, error, processAIMessage } = useAIAssistant()
  const { addObject, deleteSelected, updateScale, updatePosition, updateColor } = useModelViewer()

  const aiProviders = {
    chatgpt: { name: 'ChatGPT' },
    claude: { name: 'Claude' },
    hunyuan: { name: 'Hunyuan' }
  }

  // Update AI provider when source changes
  useEffect(() => {
    setAISource(activeSource)
  }, [activeSource])

  // Initialize plugins based on selected AI source
  useEffect(() => {
    const availablePlugins: Array<AIPlugin> = [
      // Add plugins specific to the selected AI here
      // Example:
      // { name: "hunyuan3d", supportsCommands: ["generate-mesh"] }
    ]

    availablePlugins.forEach(plugin => pluginManagerRef.current.register(plugin))
    setPlugins(pluginManagerRef.current.getAll())
  }, [activeSource])

  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  useEffect(() => {
    if (isOpen && !isMinimized && inputRef.current) {
      inputRef.current.focus()
    }
  }, [isOpen, isMinimized])

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim()) return
    handleSubmit(e)

    // Optional: Process message after sending
    setTimeout(() => {
      processAIMessage?.(input)
    }, 500)
  }

  if (!isOpen) {
    return (
      <Button
        onClick={() => setIsOpen(true)}
        className="fixed top-20 right-4 z-50 rounded-full w-12 h-12 btn-logo-gradient text-white shadow-lg border-0"
        aria-label="Open Grok AI Assistant"
      >
        <MessageCircle className="h-6 w-6" />
      </Button>
    )
  }

  return (
    <Card
      id={id}
      className={cn(
        "fixed top-20 right-4 z-50 w-80 shadow-xl transition-all duration-300 border-logo-purple/30",
        isMinimized ? "h-12" : "h-96",
        "bg-black/80 backdrop-blur-md",
      )}
    >
      {/* Header */}
      <CardHeader className="p-3 border-b border-logo-purple/20 flex flex-row items-center justify-between space-y-0 bg-logo-gradient">
        <CardTitle className="text-sm font-medium flex items-center text-white">
          <Bot className="h-4 w-4 mr-2" />
          Conceivo
          <Zap className="h-3 w-3 ml-1 text-yellow-300" />
        </CardTitle>
        <div className="flex items-center space-x-1">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setIsMinimized(!isMinimized)}
            aria-label={isMinimized ? "Maximize" : "Minimize"}
            className="h-6 w-6 p-0 text-white hover:bg-white/20"
          >
            {isMinimized ? <Maximize2 className="h-3 w-3" /> : <Minimize2 className="h-3 w-3" />}
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setIsOpen(false)}
            aria-label="Close Grok AI Assistant"
            className="h-6 w-6 p-0 text-white hover:bg-white/20"
          >
            <X className="h-3 w-3" />
          </Button>
        </div>
      </CardHeader>

      {!isMinimized && (
        <CardContent className="p-0">
          {/* Messages */}
          <div className="relative h-64 overflow-y-auto p-3 space-y-3 chat-messages">
            {messages.length === 0 && !isLoading && (
              <div className="text-center text-gray-500 text-sm py-6">
                <Zap className="h-8 w-8 text-yellow-400 mx-auto mb-2" />
                <p className="font-medium">Powered by Grok AI</p>
                <p className="text-xs mt-1">Ask me anything about 3D modeling, CAD tools, or design suggestions.</p>
              </div>
            )}

            {messages.map((message) => (
              <div key={message.id} className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}>
                <div
                  className={cn(
                    "max-w-[80%] rounded-lg px-3 py-2",
                    message.role === "user"
                      ? "bg-logo-gradient text-white ml-auto"
                      : "bg-gray-800/70 text-gray-300 border border-logo-purple/20 mr-auto",
                  )}
                >
                  <div className="flex items-start space-x-2">
                    {message.role === "assistant" ? (
                      <div className="flex items-center">
                        <Bot className="h-4 w-4 mt-0.5 text-logo-purple flex-shrink-0" />
                        <Zap className="h-2 w-2 text-yellow-400 ml-0.5" />
                      </div>
                    ) : (
                      <User className="h-4 w-4 mt-0.5 text-logo-cyan flex-shrink-0" />
                    )}
                    <p className="text-sm whitespace-pre-line">{message.content}</p>
                  </div>
                </div>
              </div>
            ))}

            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-gray-800/70 text-gray-300 rounded-lg px-3 py-2 border border-logo-purple/20">
                  <div className="flex items-center space-x-2">
                    <Bot className="h-4 w-4 text-logo-purple" />
                    <Zap className="h-3 w-3 text-yellow-400 animate-pulse" />
                    <div className="flex space-x-1">
                      <div className="w-2 h-2 bg-logo-purple rounded-full animate-bounce" />
                      <div
                        className="w-2 h-2 bg-logo-cyan rounded-full animate-bounce"
                        style={{ animationDelay: "0.1s" }}
                      />
                      <div
                        className="w-2 h-2 bg-logo-purple rounded-full animate-bounce"
                        style={{ animationDelay: "0.2s" }}
                      />
                    </div>
                  </div>
                </div>
              </div>
            )}

            {error && (
              <div className="flex justify-start">
                <div className="bg-red-900/70 text-red-300 rounded-lg px-3 py-2 border border-red-500/20">
                  <div className="flex items-center space-x-2">
                    <Bot className="h-4 w-4 text-red-400" />
                    <p className="text-sm">
                      Error: {typeof error === 'string' ? error : JSON.stringify(error)}
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* Bottom of message list */}
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className="p-3 border-t border-logo-purple/20">
            <form onSubmit={onSubmit} className="flex items-center space-x-2">
              <Input
                ref={inputRef}
                value={input}
                onChange={handleInputChange}
                placeholder="Ask Grok about 3D modeling..."
                className="flex-1 border-logo-purple/30 focus:border-logo-cyan bg-slate-900/50 text-white"
                disabled={isLoading}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault()
                    onSubmit(e)
                  }
                }}
              />
              <Button
                type="submit"
                size="icon"
                disabled={isLoading || !input.trim()}
                className="btn-logo-gradient text-white border-0 hover:bg-logo-cyan/90"
              >
                {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
              </Button>
            </form>
            <div className="text-xs text-gray-500 mt-1 text-center">
              Powered by <span className="text-yellow-400">Grok AI</span>
            </div>
          </div>
          <div className="p-2 border-t border-logo-purple/20 bg-slate-900/50">
            <label className="text-xs text-gray-400 block mb-1">AI Source</label>
            <select
              value={activeSource}
              onChange={(e) => setActiveSource(e.target.value)}
              className="w-full bg-black text-white text-sm border border-logo-purple/30 rounded p-1.5"
            >
              {Object.entries(aiProviders).map(([value, provider]) => (
                <option key={value} value={value}>
                  {value.charAt(0).toUpperCase() + value.slice(1)}
                </option>
              ))}
            </select>
          </div>
          {plugins.length > 0 && (
            <Card className="bg-black/80 backdrop-blur-md border border-logo-purple/20 mt-2">
              <CardHeader className="p-2 border-b border-logo-purple/20">
                <CardTitle className="text-xs text-gray-300">Extensions</CardTitle>
              </CardHeader>
              <CardContent className="p-2 space-y-1">
                {plugins.map(({ plugin, enabled }) => (
                  <div key={plugin.name} className="flex items-center justify-between">
                    <span className="text-xs text-gray-400">{plugin.name}</span>
                    <button
                      onClick={() => {
                        const updatedPlugins = plugins.map(p =>
                          p.plugin.name === plugin.name ? { ...p, enabled: !p.enabled } : p
                        )
                        setPlugins(updatedPlugins)
                      }}
                      className={`h-6 w-6 p-0 ${enabled ? 'text-green-400' : 'text-gray-600'}`}
                    >
                      {enabled ? '✔' : '✘'}
                    </button>
                  </div>
                ))}
              </CardContent>
            </Card>
          )}
        </CardContent>
      )}
    </Card>
  )
}
