"use client"

import type React from "react"
import { useState, useRef, useEffect } from "react"
import { Navbar } from "../components/navbar"
import { Hero } from "../components/hero"
import { Button } from "../components/ui/button"
import { Input } from "../components/ui/input"
import { Card, CardContent } from "../components/ui/card"
import {
  ArrowRight,
  Sparkles,
  ImageIcon,
  CuboidIcon as Cube,
  Lightbulb,
  Loader2,
  ChevronLeft,
  Folder,
  History,
} from "lucide-react"
import Link from "next/link"
import { cn } from "../lib/utils"
import { FeatureCardSkeleton } from "../components/ui/feature-card-skeleton"
import * as THREE from "three"

// Predefined prompt options with space theme
const promptOptions = [
  "A futuristic space station with rotating modules",
  "A sleek lunar rover with solar panels",
  "A cosmic helmet with holographic visor",
  "A zero-gravity chair for astronauts",
  "A starship cockpit with interactive controls",
]

export default function Home() {
  const [prompt, setPrompt] = useState("")
  const [enhancedPrompt, setEnhancedPrompt] = useState("")
  const [isGenerating, setIsGenerating] = useState(false)
  const [generatedScene, setGeneratedScene] = useState<THREE.Scene | null>(null)
  const [generatedDescription, setGeneratedDescription] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<"prompt" | "image" | "description">("prompt")
  const [isSidebarOpen, setIsSidebarOpen] = useState(true)
  const inputRef = useRef<HTMLInputElement>(null)
  const canvasRef = useRef<HTMLDivElement>(null)

  // Focus input on load
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.focus()
    }
  }, [])

  // Initialize Three.js scene with space theme
  useEffect(() => {
    if (canvasRef.current && !generatedScene) {
      const scene = new THREE.Scene()
      const camera = new THREE.PerspectiveCamera(75, 1, 0.1, 1000)
      const renderer = new THREE.WebGLRenderer({ alpha: true })
      renderer.setSize(512, 512)
      renderer.setClearColor(0x000000, 0)

      const geometry = new THREE.BoxGeometry(1, 1, 1)
      const material = new THREE.MeshPhongMaterial({
        color: 0x808080,
        specular: 0x555555,
        shininess: 30,
      })
      const cube = new THREE.Mesh(geometry, material)
      cube.position.set(0, 0, 0)
      scene.add(cube)

      // Add ambient and point light for space effect
      const ambientLight = new THREE.AmbientLight(0x404040)
      scene.add(ambientLight)
      const pointLight = new THREE.PointLight(0xffffff, 1, 100)
      pointLight.position.set(5, 5, 5)
      scene.add(pointLight)

      camera.position.z = 5
      const animate = () => {
        requestAnimationFrame(animate)
        cube.rotation.x += 0.005
        cube.rotation.y += 0.005
        renderer.render(scene, camera)
      }
      animate()

      if (canvasRef.current) {
        canvasRef.current.appendChild(renderer.domElement)
        setGeneratedScene(scene)
      }

      return () => {
        renderer.dispose()
      }
    }
  }, [generatedScene])

  // Handle prompt submission
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!prompt.trim()) return
    setIsGenerating(true)

    try {
      await new Promise((resolve) => setTimeout(resolve, 3000))
      setGeneratedDescription(
        `A 3D model of ${prompt} with the following space-inspired features:
- Designed for zero-gravity environments
- Reflective metallic finish for cosmic durability
- Integrated with holographic interfaces
- Optimized for lunar or orbital use

This design blends futuristic aesthetics with functional space technology.`,
      )
      setActiveTab("image")
    } catch (error) {
      console.error("AI generation failed:", error)
      setGeneratedDescription("Failed to generate. Please try again.")
    } finally {
      setIsGenerating(false)
    }
  }

  // Enhance prompt with space-themed suggestions
  const enhancePrompt = () => {
    if (!prompt.trim()) return
    const enhanced = `${prompt} with a sleek space-grade alloy exterior, glowing neon accents, and compatibility for extraterrestrial conditions`
    setEnhancedPrompt(enhanced)
    setPrompt(enhanced)
  }

  // Handle random prompt selection
  const selectRandomPrompt = () => {
    const randomPrompt = promptOptions[Math.floor(Math.random() * promptOptions.length)]
    setPrompt(randomPrompt)
  }

  return (
    <div className="min-h-screen bg-cosmic-gradient/95 text-white flex flex-col">
      <Navbar />

      {/* Show Hero section only when no content is generated */}
      {!generatedScene && <Hero />}

      <div className="flex-1 flex">
        {/* Sidebar */}
        <aside
          className={`transition-all duration-300 ease-in-out ${
            isSidebarOpen ? "w-64" : "w-0"
          } overflow-hidden bg-nebula-purple/80 backdrop-blur-sm border-r border-nebula-purple/20`}
        >
          <div className="p-4">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setIsSidebarOpen(false)}
              className="w-full text-star-yellow hover:bg-nebula-purple/30 mb-4"
              aria-label={isSidebarOpen ? "Close sidebar" : "Open sidebar"}
            >
              <ChevronLeft className="h-4 w-4 mr-2" />
              {isSidebarOpen && "Close"}
            </Button>
            {isSidebarOpen && (
              <>
                <h3 className="text-sm font-semibold text-star-yellow mb-2">Projects</h3>
                <Button
                  variant="ghost"
                  size="sm"
                  className="w-full justify-start text-gray-300 hover:text-star-yellow hover:bg-nebula-purple/20 mb-2"
                >
                  <Folder className="h-4 w-4 mr-2" />
                  Project 1
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  className="w-full justify-start text-gray-300 hover:text-star-yellow hover:bg-nebula-purple/20"
                >
                  <Folder className="h-4 w-4 mr-2" />
                  Project 2
                </Button>
                <h3 className="text-sm font-semibold text-star-yellow mt-4 mb-2">History</h3>
                <Button
                  variant="ghost"
                  size="sm"
                  className="w-full justify-start text-gray-300 hover:text-star-yellow hover:bg-nebula-purple/20"
                >
                  <History className="h-4 w-4 mr-2" />
                  Session 1
                </Button>
              </>
            )}
          </div>
        </aside>

        {/* Main Content */}
        <main className="flex-1 flex flex-col">
          <section className="relative flex-1 flex items-center justify-center px-4 py-20">
            {/* Space-themed background effects */}
            <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,rgba(255,255,255,0.05) 0%,transparent 70%)] bg-[size:20px_20px]" />
            <div className="absolute top-20 left-10 w-72 h-72 bg-nebula-purple/20 rounded-full blur-3xl animate-pulse" />
            <div
              className="absolute bottom-20 right-10 w-96 h-96 bg-star-yellow/20 rounded-full blur-3xl animate-pulse"
              style={{ animationDelay: "1s" }}
            />

            <div className="max-w-3xl w-full mx-auto z-10">
              {!generatedScene ? (
                <div className="text-center mb-8">
                  <h1 className="text-4xl md:text-6xl font-bold mb-6 text-white">
                    Explore the <span className="text-nebula-gradient">cosmos</span> of creation!
                  </h1>
                  <p className="text-xl text-gray-300 mb-8">
                    Describe your space product idea and our AI will launch it into 3D
                  </p>
                </div>
              ) : (
                <div className="flex justify-center mb-8">
                  <div className="flex space-x-4 bg-gray-900/50 backdrop-blur-sm rounded-lg p-1">
                    <button
                      className={cn(
                        "px-4 py-2 rounded-md text-sm font-medium transition-all",
                        activeTab === "prompt" ? "bg-nebula-gradient text-white" : "text-gray-300 hover:text-white",
                      )}
                      onClick={() => setActiveTab("prompt")}
                      aria-label="Switch to prompt tab"
                    >
                      <Lightbulb className="h-4 w-4 inline mr-2" />
                      Prompt
                    </button>
                    <button
                      className={cn(
                        "px-4 py-2 rounded-md text-sm font-medium transition-all",
                        activeTab === "image" ? "bg-nebula-gradient text-white" : "text-gray-300 hover:text-white",
                      )}
                      onClick={() => setActiveTab("image")}
                      aria-label="Switch to 3D model tab"
                    >
                      <ImageIcon className="h-4 w-4 inline mr-2" />
                      3D Model
                    </button>
                    <button
                      className={cn(
                        "px-4 py-2 rounded-md text-sm font-medium transition-all",
                        activeTab === "description" ? "bg-nebula-gradient text-white" : "text-gray-300 hover:text-white",
                      )}
                      onClick={() => setActiveTab("description")}
                      aria-label="Switch to description tab"
                    >
                      <Sparkles className="h-4 w-4 inline mr-2" />
                      AI Description
                    </button>
                  </div>
                </div>
              )}

              <Card className="bg-gray-900/50 backdrop-blur-sm border border-nebula-purple/30 shadow-2xl">
                <CardContent className={cn("p-6", generatedScene && activeTab !== "prompt" ? "hidden" : "block")}>
                  <form onSubmit={handleSubmit} className="space-y-4">
                    <div className="relative">
                      <Input
                        ref={inputRef}
                        type="text"
                        placeholder="Describe your space product (e.g., 'A lunar rover with solar panels')"
                        className="bg-black/50 border-nebula-purple/30 focus:border-star-yellow text-white h-14 pl-4 pr-32 text-lg"
                        value={prompt}
                        onChange={(e) => setPrompt(e.target.value)}
                        disabled={isGenerating}
                        aria-label="Enter your space product idea"
                      />
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        className="absolute right-20 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-nebula-purple"
                        onClick={selectRandomPrompt}
                        disabled={isGenerating}
                        aria-label="Get a random space prompt suggestion"
                      >
                        <Sparkles className="h-4 w-4 mr-1" />
                        Inspire me
                      </Button>
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        className="absolute right-2 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-star-yellow"
                        onClick={enhancePrompt}
                        disabled={isGenerating || !prompt.trim()}
                        aria-label="Enhance your prompt with space features"
                      >
                        <Sparkles className="h-4 w-4 mr-1" />
                        Enhance Prompt
                      </Button>
                    </div>
                    <div className="flex justify-center">
                      <Button
                        type="submit"
                        className="btn-nebula-gradient text-white px-8 py-6 text-lg font-medium border-0 shadow-lg hover:shadow-nebula-purple/30"
                        disabled={isGenerating || !prompt.trim()}
                        aria-label="Generate 3D model with AI"
                      >
                        {isGenerating ? (
                          <>
                            <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                            Generating...
                          </>
                        ) : (
                          <>
                            Launch into 3D
                            <ArrowRight className="ml-2 h-5 w-5" />
                          </>
                        )}
                      </Button>
                    </div>
                  </form>
                </CardContent>

                {/* Generated 3D Model Tab */}
                {generatedScene && activeTab === "image" && (
                  <CardContent className="p-6">
                    <div className="flex flex-col items-center">
                      <div
                        ref={canvasRef}
                        className="relative w-full max-w-md aspect-square mb-4 border border-nebula-purple/30 rounded-lg overflow-hidden"
                      >
                        {isGenerating && (
                          <div className="absolute inset-0 bg-gray-900/50 flex items-center justify-center">
                            <Loader2 className="h-10 w-10 animate-spin text-star-yellow" />
                          </div>
                        )}
                      </div>
                      <p className="text-gray-300 text-center mb-6">
                        AI-generated 3D model for: <span className="text-star-yellow">{prompt}</span>
                      </p>
                      <Link href="/design">
                        <Button
                          className="btn-nebula-gradient text-white px-8 py-6 text-lg font-medium border-0 shadow-lg hover:shadow-nebula-purple/30"
                          aria-label="Start 3D design process"
                        >
                          Start 3D Design
                          <Cube className="ml-2 h-5 w-5" />
                        </Button>
                      </Link>
                    </div>
                  </CardContent>
                )}

                {/* Generated Description Tab */}
                {generatedDescription && activeTab === "description" && (
                  <CardContent className="p-6">
                    <div className="bg-black/50 border border-nebula-purple/20 rounded-lg p-4 mb-6">
                      <h3 className="text-nebula-gradient font-bold mb-3">AI Design Recommendations</h3>
                      <div className="text-gray-300 whitespace-pre-line">{generatedDescription}</div>
                    </div>
                    <div className="flex justify-center">
                      <Link href="/design">
                        <Button
                          className="btn-nebula-gradient text-white px-8 py-6 text-lg font-medium border-0 shadow-lg hover:shadow-nebula-purple/30"
                          aria-label="Start 3D design process"
                        >
                          Start 3D Design
                          <Cube className="ml-2 h-5 w-5" />
                        </Button>
                      </Link>
                    </div>
                  </CardContent>
                )}
              </Card>

              {!generatedScene && (
                <div className="mt-6 text-center">
                  <p className="text-gray-400 text-sm">
                    Not sure where to start? Try one of our{" "}
                    <button
                      className="text-nebula-purple hover:text-star-yellow underline transition-colors"
                      onClick={selectRandomPrompt}
                      aria-label="View example space prompts"
                    >
                      example prompts
                    </button>
                  </p>
                </div>
              )}
            </div>
          </section>

          {/* Features preview (only shown before generation) */}
          {!generatedScene && (
            <section className="py-16 px-6 border-t border-nebula-purple/20">
              <div className="max-w-6xl mx-auto">
                <h2 className="text-2xl font-bold text-center mb-12 text-nebula-gradient">
                  From Concept to Cosmic Creation
                </h2>
                <div className="grid md:grid-cols-3 gap-8">
                  {Array.from({ length: 3 }).map((_, i) => (
                    <FeatureCardSkeleton key={i} />
                  ))}
                </div>
              </div>
            </section>
          )}
        </main>
      </div>
    </div>
  )
}