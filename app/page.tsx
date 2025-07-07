"use client"

import React, { useState, useRef, useEffect } from "react";
import { Navbar } from "../components/navbar";
import { Hero } from "../components/hero";
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";
import { Card, CardContent } from "../components/ui/card";
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
  Edit,
} from "lucide-react";
import Link from "next/link";
import { cn } from "../lib/utils";
import { FeatureCardSkeleton } from "../components/ui/feature-card-skeleton";
import { ModelViewerComponent, useModelViewer } from "../hooks/use-model-viewer";
import { TOOL_CATEGORIES } from "../types/tool-category";
import { ModelObject, ToolCategory, Tool } from "@/types";
import axios from "axios";

const promptOptions = [
  "A futuristic space station with rotating modules",
  "A sleek lunar rover with solar panels",
  "A cosmic helmet with holographic visor",
  "A zero-gravity chair for astronauts",
  "A starship cockpit with interactive controls",
];

// Polyfill for Array.prototype.includes
function arrayIncludes<T>(array: T[], searchElement: T): boolean {
  return array.some(element => element === searchElement);
}

export default function Home() {
  const {
    objects,
    selectedObject,
    setSelectedObject,
    addObject: rawAddObject,
    importSTL,
    deleteSelected,
    updatePosition,
    updateScale,
    updateRotation,
    updateColor,
    showStress,
    toolCategories,
    setToolCategories,
    addToHistory,
    history,
    historyIndex,
    undo,
    redo,
    selectAll,
    setViewMode,
    animate,
    applyMeshOperation,
    applyModelingOperation,
  } = useModelViewer();

  // Define valid object types
  type ValidObjectType = "box" | "sphere" | "cylinder" | "cone" | "torus" | "plane" | "wedge" | "stl";

  // Type guard for valid object types
  function isValidObjectType(type: string): type is ValidObjectType {
    return arrayIncludes(["box", "sphere", "cylinder", "cone", "torus", "plane", "wedge", "stl"], type);
  }

  // Type-safe addObject with single object parameter
  const addObject = (params: {
    type: ValidObjectType;
    id?: string;
    modelPath?: string;
    position?: number[];
    rotation?: number[];
    scale?: number[];
    color?: string;
  }) => {
    const toTriple = (arr?: number[]): [number, number, number] | undefined =>
      Array.isArray(arr) && arr.length === 3 ? [arr[0], arr[1], arr[2]] : undefined;
    rawAddObject(
      params.type,
      params.id,
      params.modelPath,
      toTriple(params.position),
      toTriple(params.rotation),
      toTriple(params.scale),
      params.color
    );
  };

  const [prompt, setPrompt] = useState("");
  const [enhancedPrompt, setEnhancedPrompt] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedDescription, setGeneratedDescription] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"prompt" | "image" | "description" | "design">("prompt");
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [bevelDistance, setBevelDistance] = useState("0.1");
  const [cutPosition, setCutPosition] = useState("0.5");
  const [slideDistance, setSlideDistance] = useState("0.1");
  const inputRef = useRef<HTMLInputElement>(null);

  // WebSocket for collaboration
  useEffect(() => {
    const wsUrl = process.env.NEXT_PUBLIC_WEBSOCKET_URL || "ws://localhost:8000/api/collaborate";
    const ws = new WebSocket(wsUrl);
    ws.onopen = () => console.log("WebSocket connected");
    ws.onmessage = (event) => {
      try {
        const newObjects = JSON.parse(event.data);
        newObjects.forEach((obj: ModelObject) => {
          if (isValidObjectType(obj.type)) {
            addObject({
              type: obj.type,
              id: obj.id,
              modelPath: obj.modelPath,
              position: obj.position,
              rotation: obj.rotation,
              scale: obj.scale,
              color: obj.color,
            });
          }
        });
      } catch (err) {
        setError("Failed to process collaboration data.");
      }
    };
    ws.onerror = () => setError("Collaboration failed. Check backend connection.");
    ws.onclose = () => console.log("WebSocket closed");
    return () => ws.close();
  }, [addObject]);

  // Fetch ConceivoAI components
  useEffect(() => {
    const fetchComponents = async () => {
      try {
        const response = await axios.get("http://localhost:8000/api/conceivo/components");
        response.data.components.forEach((comp: any) => {
          if (comp.model_path) {
            // Simulate STL import with model path
            addObject({ type: "stl", id: comp.id, modelPath: comp.model_path });
          }
        });
      } catch (err) {
        setError("Failed to load components from backend.");
      }
    };
    fetchComponents();
  }, [addObject]);

  // Focus input on load
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.focus();
    }
  }, []);

  // Handle prompt submission
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim()) return;
    setIsGenerating(true);
    setError(null);

    try {
      const response = await axios.post("http://localhost:8000/api/conceivo", {
        message: `Generate a 3D model: ${prompt}`,
      });
      const { command, edit_result, new_knowledge } = response.data;
      if (command?.action === "model" || command?.action === "add") {
        const type = prompt.toLowerCase().includes("rover") ? "box" :
                     prompt.toLowerCase().includes("helmet") ? "sphere" :
                     prompt.toLowerCase().includes("station") ? "cylinder" :
                     prompt.toLowerCase().includes("chair") ? "plane" :
                     prompt.toLowerCase().includes("cockpit") ? "cone" :
                     command.params?.type || "box";
        if (isValidObjectType(type)) {
          addObject({ type });
          setGeneratedDescription(
            new_knowledge ||
            `A 3D model of ${prompt} with space-inspired features:\n- Zero-gravity optimized\n- Reflective alloy finish\n- Holographic interfaces\n- Lunar/orbital use`
          );
          setActiveTab("image");
        } else {
          setError("Invalid object type generated.");
        }
      } else {
        setError(edit_result || "Failed to generate model.");
      }
    } catch (error) {
      console.error("AI generation failed:", error);
      setError("Backend unavailable. Ensure the server is running.");
    } finally {
      setIsGenerating(false);
    }
  };

  // Enhance prompt
  const enhancePrompt = async () => {
    if (!prompt.trim()) return;
    try {
      const response = await axios.post("http://localhost:8000/api/enhance-prompt", { prompt });
      const enhanced = response.data.enhanced_prompt;
      setEnhancedPrompt(enhanced);
      setPrompt(enhanced);
    } catch (error) {
      console.error("Prompt enhancement failed:", error);
      const fallback = `${prompt} with a sleek space-grade alloy exterior, glowing neon accents, and compatibility for extraterrestrial conditions`;
      setEnhancedPrompt(fallback);
      setPrompt(fallback);
    }
  };

  // Random prompt
  const selectRandomPrompt = () => {
    const randomPrompt = promptOptions[Math.floor(Math.random() * promptOptions.length)];
    setPrompt(randomPrompt);
  };

  // Process commands
  const processCommand = async () => {
    const commandInput = document.getElementById("commandInput") as HTMLInputElement;
    const command = commandInput.value.toLowerCase();
    if (!selectedObject && !command.includes("add")) {
      setError("Select an object first.");
      return;
    }
    try {
      const selectedObj = objects.find((obj: ModelObject) => obj.id === selectedObject);
      const params: any = {};
      if (command.includes("bevel-edges")) {
        params.distance = parseFloat(bevelDistance) || 0.1;
      } else if (command.includes("loop-cut-and-slide")) {
        params.cut_position = parseFloat(cutPosition) || 0.5;
        params.slide_distance = parseFloat(slideDistance) || 0.1;
      }
      const response = await axios.post("http://localhost:8000/api/conceivo", {
        message: command,
        selected_part: selectedObj?.type,
      });
      const { edit_result, command: cmd, new_knowledge } = response.data;
      if (edit_result) {
        if (cmd?.action === "add" && isValidObjectType(cmd.params?.type)) {
          addObject(cmd.params.type);
        } else if (arrayIncludes(["delete", "fix", "fill", "rotate", "scale"], cmd?.action)) {
          applyMeshOperation(cmd.action);
        } else if (arrayIncludes(["extrude", "revolve", "mirror", "union", "subtract", "intersect", "bevel-edges", "loop-cut-and-slide"], cmd?.action)) {
          applyModelingOperation(cmd.action);
        } else {
          setError("Command not recognized.");
        }
        setGeneratedDescription(new_knowledge || edit_result);
        setActiveTab("description");
      } else {
        setError("Command failed or not recognized.");
      }
    } catch (error: any) {
      console.error("Command failed:", error);
      setError(error.response?.data?.detail || "Backend unavailable. Ensure the server is running.");
    }
    commandInput.value = "";
  };

  return (
    <div className="min-h-screen bg-cosmic-gradient/95 text-white flex flex-col">
      <Navbar />
      {objects.length === 0 && <Hero />}
      <div className="flex-1 flex">
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
                  {objects.map((obj: ModelObject) => (
                    <li
                      key={obj.id}
                      className={cn(
                        "text-gray-300 hover:text-star-yellow hover:bg-nebula-purple/20 p-1 rounded flex items-center",
                        selectedObject === obj.id && "text-red-500"
                      )}
                      onClick={() => setSelectedObject(obj.id)}
                    >
                      {obj.type} {selectedObject === obj.id && "(selected)"}
                    </li>
                  ))}
                </ul>
                <h3 className="text-sm font-semibold text-star-yellow mt-4 mb-2">Tools</h3>
                {toolCategories.map((category: ToolCategory) => (
                  <div key={category.name}>
                    <h4 className="text-xs font-medium text-gray-300">{category.name}</h4>
                    {category.tools.map((tool: Tool) => (
                      <Button
                        key={tool.name}
                        variant="ghost"
                        size="sm"
                        className="w-full justify-start text-gray-300 hover:text-star-yellow hover:bg-nebula-purple/20"
                        onClick={() => {
                          if (tool.action.startsWith("add-")) {
                            const type = tool.action.replace("add-", "");
                            if (isValidObjectType(type)) {
                              addObject({ type });
                            }
                          } else if (tool.action === "import-stl") {
                            const input = document.createElement("input");
                            input.type = "file";
                            input.accept = ".stl";
                            input.onchange = (e) => {
                              const file = (e.target as HTMLInputElement).files?.[0];
                              if (file) importSTL(file);
                            };
                            input.click();
                          } else if (tool.action === "show-stress") {
                            showStress();
                          } else if (tool.action === "select-all") {
                            selectAll();
                          } else if (tool.action === "view-wireframe" || tool.action === "view-shaded") {
                            setViewMode(tool.action.replace("view-", "") as "wireframe" | "shaded");
                          } else if (tool.action === "animate") {
                            animate();
                          } else if (tool.action === "bevel-edges" && selectedObject) {
                            applyModelingOperation("bevel-edges");
                          } else if (tool.action === "loop-cut-and-slide" && selectedObject) {
                            applyModelingOperation("loop-cut-and-slide", {
                              objectIds: [selectedObject],
                              params: { cut_position: parseFloat(cutPosition) || 0.5, slide_distance: parseFloat(slideDistance) || 0.1 },
                            });
                          } else if (["union", "subtract", "intersect", "revolve", "mirror", "pattern", "duplicate-object", "extrude"].indexOf(tool.action) !== -1) {
                            applyModelingOperation(tool.action);
                          } else if (tool.action === "delete-selected") {
                            deleteSelected();
                          }
                        }}
                      >
                        {tool.icon} {tool.name}
                      </Button>
                    ))}
                  </div>
                ))}
                <h3 className="text-sm font-semibold text-star-yellow mt-4 mb-2">Projects</h3>
                <Button
                  variant="ghost"
                  size="sm"
                  className="w-full justify-start text-gray-300 hover:text-star-yellow hover:bg-nebula-purple/20 mb-2"
                >
                  <Folder className="h-4 w-4 mr-2" /> Project 1
                </Button>
                <h3 className="text-sm font-semibold text-star-yellow mt-4 mb-2">History</h3>
                <Button
                  variant="ghost"
                  size="sm"
                  className="w-full justify-start text-gray-300 hover:text-star-yellow hover:bg-nebula-purple/20"
                  onClick={() => historyIndex > 0 && undo()}
                >
                  <History className="h-4 w-4 mr-2" /> Undo
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  className="w-full justify-start text-gray-300 hover:text-star-yellow hover:bg-nebula-purple/20"
                  onClick={() => historyIndex < history.length - 1 && redo()}
                >
                  <History className="h-4 w-4 mr-2" /> Redo
                </Button>
              </>
            )}
          </div>
        </aside>
        <main className="flex-1 flex flex-col">
          <section className="relative flex-1 flex items-center justify-center px-4 py-20">
            <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,rgba(255,255,255,0.05)_0%,transparent_70%)] bg-[size:20px_20px]" />
            <div className="absolute top-20 left-10 w-72 h-72 bg-nebula-purple/20 rounded-full blur-3xl animate-pulse" />
            <div
              className="absolute bottom-20 right-10 w-96 h-96 bg-star-yellow/20 rounded-full blur-3xl animate-pulse"
              style={{ animationDelay: "1s" }}
            />
            <div className="max-w-3xl w-full mx-auto z-10">
              {error && (
                <div className="bg-red-500/20 text-red-300 p-4 rounded-lg mb-4">{error}</div>
              )}
              {objects.length === 0 ? (
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
                        activeTab === "prompt" ? "bg-nebula-gradient text-white" : "text-gray-300 hover:text-white"
                      )}
                      onClick={() => setActiveTab("prompt")}
                      aria-label="Switch to prompt tab"
                    >
                      <Lightbulb className="h-4 w-4 inline mr-2" /> Prompt
                    </button>
                    <button
                      className={cn(
                        "px-4 py-2 rounded-md text-sm font-medium transition-all",
                        activeTab === "image" ? "bg-nebula-gradient text-white" : "text-gray-300 hover:text-white"
                      )}
                      onClick={() => setActiveTab("image")}
                      aria-label="Switch to 3D model tab"
                    >
                      <ImageIcon className="h-4 w-4 inline mr-2" /> 3D Model
                    </button>
                    <button
                      className={cn(
                        "px-4 py-2 rounded-md text-sm font-medium transition-all",
                        activeTab === "description" ? "bg-nebula-gradient text-white" : "text-gray-300 hover:text-white"
                      )}
                      onClick={() => setActiveTab("description")}
                      aria-label="Switch to description tab"
                    >
                      <Sparkles className="h-4 w-4 inline mr-2" /> AI Description
                    </button>
                    <button
                      className={cn(
                        "px-4 py-2 rounded-md text-sm font-medium transition-all",
                        activeTab === "design" ? "bg-nebula-gradient text-white" : "text-gray-300 hover:text-white"
                      )}
                      onClick={() => setActiveTab("design")}
                      aria-label="Switch to design tab"
                    >
                      <Edit className="h-4 w-4 inline mr-2" /> Design
                    </button>
                  </div>
                </div>
              )}
              <Card className="bg-gray-900/50 backdrop-blur-sm border border-nebula-purple/30 shadow-2xl">
                <CardContent className={cn("p-6", objects.length && activeTab !== "prompt" ? "hidden" : "block")}>
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
                        <Sparkles className="h-4 w-4 mr-1" /> Inspire me
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
                        <Sparkles className="h-4 w-4 mr-1" /> Enhance Prompt
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
                            <Loader2 className="mr-2 h-5 w-5 animate-spin" /> Generating...
                          </>
                        ) : (
                          <>
                            Launch into 3D <ArrowRight className="ml-2 h-5 w-5" />
                          </>
                        )}
                      </Button>
                    </div>
                  </form>
                </CardContent>
                {objects.length > 0 && activeTab === "image" && (
                  <CardContent className="p-6">
                    <div className="flex flex-col items-center">
                      <div className="relative w-full max-w-md aspect-square mb-4 border border-nebula-purple/30 rounded-lg overflow-hidden">
                        <ModelViewerComponent />
                      </div>
                      <p className="text-gray-300 text-center mb-6">
                        AI-generated 3D model for: <span className="text-star-yellow">{prompt}</span>
                      </p>
                      <Link href="/design">
                        <Button
                          className="btn-nebula-gradient text-white px-8 py-6 text-lg font-medium border-0 shadow-lg hover:shadow-nebula-purple/30"
                          aria-label="Start 3D design process"
                        >
                          Start 3D Design <Cube className="ml-2 h-5 w-5" />
                        </Button>
                      </Link>
                    </div>
                  </CardContent>
                )}
                {objects.length > 0 && activeTab === "design" && (
                  <CardContent className="p-6">
                    <div className="flex flex-col items-center">
                      <div className="relative w-full max-w-md aspect-square mb-4 border border-nebula-purple/30 rounded-lg overflow-hidden">
                        <ModelViewerComponent />
                      </div>
                      <p className="text-gray-300 text-center mb-6">
                        Edit the model: <span className="text-star-yellow">{prompt}</span>
                      </p>
                      <div className="flex flex-col space-y-4 w-full max-w-md">
                        <div className="flex space-x-4">
                          <Input
                            id="commandInput"
                            type="text"
                            placeholder="e.g., add sphere, extrude body, show stress"
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
                        {selectedObject && (
                          <div className="space-y-2">
                            <p className="text-star-yellow">
                              Selected: {objects.find((obj: ModelObject) => obj.id === selectedObject)?.type}
                            </p>
                            <div className="flex flex-col space-y-2">
                              <div>
                                <label className="text-sm text-gray-300">Bevel Distance (m):</label>
                                <Input
                                  type="number"
                                  step="0.01"
                                  value={bevelDistance}
                                  onChange={(e) => setBevelDistance(e.target.value)}
                                  className="bg-black/50 border-nebula-purple/30 text-white h-8"
                                  placeholder="0.1"
                                />
                              </div>
                              <div>
                                <label className="text-sm text-gray-300">Loop Cut Position (0-1):</label>
                                <Input
                                  type="number"
                                  step="0.01"
                                  min="0"
                                  max="1"
                                  value={cutPosition}
                                  onChange={(e) => setCutPosition(e.target.value)}
                                  className="bg-black/50 border-nebula-purple/30 text-white h-8"
                                  placeholder="0.5"
                                />
                              </div>
                              <div>
                                <label className="text-sm text-gray-300">Slide Distance (m):</label>
                                <Input
                                  type="number"
                                  step="0.01"
                                  value={slideDistance}
                                  onChange={(e) => setSlideDistance(e.target.value)}
                                  className="bg-black/50 border-nebula-purple/30 text-white h-8"
                                  placeholder="0.1"
                                />
                              </div>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  </CardContent>
                )}
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
                          Start 3D Design <Cube className="ml-2 h-5 w-5" />
                        </Button>
                      </Link>
                    </div>
                  </CardContent>
                )}
              </Card>
              {objects.length === 0 && (
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
          {objects.length === 0 && (
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
  );
}

// Add type declaration for process
declare const process: {
  env: {
    NEXT_PUBLIC_WEBSOCKET_URL?: string;
  };
};
