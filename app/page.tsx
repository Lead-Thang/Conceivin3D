"use client"

import React, { useState, useRef, useEffect } from "react";
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
  Trash2,
  Edit,
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
  const [activeTab, setActiveTab] = useState<"prompt" | "image" | "description" | "design">("prompt")
  const [isSidebarOpen, setIsSidebarOpen] = useState(true)
  const [selectedPart, setSelectedPart] = useState<THREE.Object3D | null>(null)
  const [parts, setParts] = useState<Record<string, THREE.Object3D>>({})
  const [isCircling, setIsCircling] = useState(false)
  const [circlePath, setCirclePath] = useState<SVGPathElement | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const canvasRef = useRef<HTMLDivElement>(null)
  const svgRef = useRef<SVGSVGElement>(null)

  // Focus input on load
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.focus()
    }
  }, [])

  // Initialize Three.js scene with space theme and parts
  useEffect(() => {
    if (canvasRef.current && !generatedScene) {
      const scene = new THREE.Scene()
      const camera = new THREE.PerspectiveCamera(75, 512 / 512, 0.1, 1000)
      const renderer = new THREE.WebGLRenderer({ alpha: true })
      renderer.setSize(512, 512)
      renderer.setClearColor(0x000000, 0)

      const body = new THREE.Mesh(
        new THREE.BoxGeometry(2, 1, 1),
        new THREE.MeshPhongMaterial({ color: 0x808080, specular: 0x555555, shininess: 30 })
      )
      const windshield = new THREE.Mesh(
        new THREE.PlaneGeometry(0.8, 0.4),
        new THREE.MeshPhongMaterial({ color: 0x0000ff, transparent: true, opacity: 0.5 })
      )
      windshield.position.set(0, 0.3, 0.6)
      const door = new THREE.Mesh(
        new THREE.BoxGeometry(0.6, 0.8, 0.1),
        new THREE.MeshPhongMaterial({ color: 0x808080 })
      )
      door.position.set(-0.8, 0, 0)
      const handle = new THREE.Mesh(
        new THREE.BoxGeometry(0.2, 0.1, 0.1),
        new THREE.MeshPhongMaterial({ color: 0xffff00 })
      )
      handle.position.set(-0.8, 0.4, 0.1)

      scene.add(body, windshield, door, handle)
      const partsMap = { body, windshield, door, handle }
      setParts(partsMap)

      const ambientLight = new THREE.AmbientLight(0x404040)
      const pointLight = new THREE.PointLight(0xffffff, 1, 100)
      pointLight.position.set(5, 5, 5)
      scene.add(ambientLight, pointLight)

      camera.position.z = 5
      const animate = () => {
        requestAnimationFrame(animate)
        body.rotation.y += 0.005
        windshield.rotation.y += 0.005
        door.rotation.y += 0.005
        handle.rotation.y += 0.005
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

  // Handle canvas click or circling for part selection
  const handleCanvasInteraction = (event: React.MouseEvent<HTMLDivElement>) => {
    if (!generatedScene || !svgRef.current) return
    const rect = canvasRef.current!.getBoundingClientRect()
    if (isCircling) {
      const x = event.clientX - rect.left
      const y = event.clientY - rect.top
      if (!circlePath) {
        const path = document.createElementNS("http://www.w3.org/2000/svg", "path")
        path.setAttribute("fill", "none")
        path.setAttribute("stroke", "red")
        path.setAttribute("stroke-width", "2")
        svgRef.current.appendChild(path)
        setCirclePath(path)
      }
      const path = circlePath!
      const d = path.getAttribute("d") || `M ${x} ${y}`
      path.setAttribute("d", `${d} L ${x} ${y}`)
      if (event.type === "mouseup") {
        setIsCircling(false)
        // Improved raycast using multiple points along the circle
        const raycaster = new THREE.Raycaster()
        const camera = new THREE.PerspectiveCamera(75, 512 / 512, 0.1, 1000)
        const points = d.split("L").slice(1).map(p => {
          const [x, y] = p.trim().split(" ").map(Number)
          return new THREE.Vector2((x / rect.width) * 2 - 1, -(y / rect.height) * 2 + 1)
        })
        let selected = null
        for (const point of points) {
          raycaster.setFromCamera(point, camera)
          const intersects = raycaster.intersectObjects(Object.values(parts))
          if (intersects.length > 0 && !selected) {
            selected = intersects[0].object
            break
          }
        }
        if (selected && selected instanceof THREE.Mesh && selected.material) {
          setSelectedPart(selected)
          selected.material.color.set(0xff0000)
        }
        if (circlePath) svgRef.current.removeChild(circlePath)
        setCirclePath(null)
      }
    } else {
      const x = ((event.clientX - rect.left) / rect.width) * 2 - 1
      const y = -((event.clientY - rect.top) / rect.height) * 2 + 1
      const raycaster = new THREE.Raycaster()
      raycaster.setFromCamera(new THREE.Vector2(x, y), new THREE.PerspectiveCamera(75, 512 / 512, 0.1, 1000))
      const intersects = raycaster.intersectObjects(Object.values(parts))
      if (intersects.length > 0 && intersects[0].object instanceof THREE.Mesh && intersects[0].object.material) {
        setSelectedPart(intersects[0].object)
        intersects[0].object.material.color.set(0xff0000)
      }
    }
  }

  // Start circling on mousedown
  const startCircling = (event: React.MouseEvent<HTMLDivElement>) => {
    if (!generatedScene) return
    setIsCircling(true)
    handleCanvasInteraction(event)
  }

  // Process text commands with backend integration
  const processCommand = async () => {
    const commandInput = document.getElementById("commandInput") as HTMLInputElement
    const command = commandInput.value.toLowerCase()
    if (selectedPart) {
      const partName = Object.keys(parts).find(key => parts[key] === selectedPart)
      if (partName) {
        const response = await fetch("http://localhost:8000/api/conceivo", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: command, selected_part: partName }),
        })
        const data = await response.json()
        if (data.edit_result) {
          alert(data.edit_result)
          if (command.includes("delete")) {
            generatedScene!.remove(selectedPart)
            setParts(prev => {
              const newParts = { ...prev }
              delete newParts[partName]
              return newParts
            })
            setSelectedPart(null)
          } else if (command.includes("fix")) {
            if (selectedPart instanceof THREE.Mesh && selectedPart.material) selectedPart.material.color.set(0x00ff00) // Example fix
          } else if (command.includes("fill")) {
            if (selectedPart instanceof THREE.Mesh && selectedPart.material) selectedPart.material.color.set(0xaaaaaa) // Example fill
          }
        } else {
          alert("Command failed or not recognized.")
        }
      }
    }
    commandInput.value = ""
  }

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

      {!generatedScene && <Hero />}

      <div className="flex-1 flex">
        {/* Sidebar (Inventory Bar) */}
        <aside
          className={`transition-all duration-300 ease-in-out ${
            isSidebarOpen ? "w-64" : "w-0"
          } overflow-hidden bg-nebula-purple/80 backdrop-blur-sm border-r border-nebula-purple/20`}
        >
          <div className="p-4" role="region" aria-label="Sidebar content">
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
                <h3 className="text-sm font-semibold text-star-yellow mb-2">Inventory</h3>
                <ul>
                  {Object.keys(parts).map(part => (
                    <li
                      key={part}
                      className={cn(
                        "text-gray-300 hover:text-star-yellow hover:bg-nebula-purple/20 p-1 rounded flex items-center",
                        selectedPart === parts[part] && "text-red-500"
                      )}
                      onClick={() => {
                        setSelectedPart(parts[part])
                        if (parts[part] instanceof THREE.Mesh && parts[part].material) parts[part].material.color.set(0xff0000)
                      }}
                    >
                      {part} {selectedPart === parts[part] && "(selected)"}
                    </li>
                  ))}
                </ul>
                <h3 className="text-sm font-semibold text-star-yellow mt-4 mb-2">Projects</h3>
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
                    <button
                      className={cn(
                        "px-4 py-2 rounded-md text-sm font-medium transition-all",
                        activeTab === "design" ? "bg-nebula-gradient text-white" : "text-gray-300 hover:text-white",
                      )}
                      onClick={() => setActiveTab("design")}
                      aria-label="Switch to design tab"
                    >
                      <Edit className="h-4 w-4 inline mr-2" />
                      Design
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
                      <div className="relative w-full max-w-md aspect-square mb-4 border border-nebula-purple/30 rounded-lg overflow-hidden" itemType="https://schema.org/ThreeDimensionalModel">
                        <svg
                          ref={svgRef}
                          style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%" }}
                        />
                        <div
                          ref={canvasRef}
                          onMouseDown={startCircling}
                          onMouseMove={handleCanvasInteraction}
                          onMouseUp={handleCanvasInteraction}
                          className="w-full h-full"
                        >
                          {isGenerating && (
                            <div className="absolute inset-0 bg-gray-900/50 flex items-center justify-center">
                              <Loader2 className="h-10 w-10 animate-spin text-star-yellow" />
                            </div>
                          )}
                        </div>
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

                {/* Design Tab for Editing */}
                {generatedScene && activeTab === "design" && (
                  <CardContent className="p-6">
                    <div className="flex flex-col items-center">
                      <div className="relative w-full max-w-md aspect-square mb-4 border border-nebula-purple/30 rounded-lg overflow-hidden">
                        <svg
                          ref={svgRef}
                          style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%" }}
                        />
                        <div
                          ref={canvasRef}
                          onMouseDown={startCircling}
                          onMouseMove={handleCanvasInteraction}
                          onMouseUp={handleCanvasInteraction}
                          className="w-full h-full"
                        />
                      </div>
                      <p className="text-gray-300 text-center mb-6">
                        Edit the model: <span className="text-star-yellow">{prompt}</span>
                      </p>
                      <div className="flex space-x-4">
                        <Input
                          id="commandInput"
                          type="text"
                          placeholder="e.g., delete windshield, fix door, fill handle"
                          className="bg-black/50 border-nebula-purple/30 focus:border-star-yellow text-white h-10 pl-4 pr-10 text-sm"
                          onKeyPress={(e) => e.key === "Enter" && processCommand()}
                        />
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={processCommand}
                          className="text-nebula-purple border-nebula-purple hover:bg-nebula-purple/20"
                        >
                          <Edit className="h-4 w-4" />
                        </Button>
                      </div>
                      {selectedPart && (
                        <p className="mt-2 text-star-yellow">
                          Selected: {Object.keys(parts).find(key => parts[key] === selectedPart)}
                        </p>
                      )}
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