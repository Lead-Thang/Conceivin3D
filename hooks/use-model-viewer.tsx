"use client";

import type React from "react";
import { useState, useCallback, useEffect } from "react";
import {
  Box,
  Cylinder,
  Move3D,
  RotateCcw,
  Scale,
  Ruler,
  Square,
  Circle,
  Triangle,
  Copy,
  CornerDownRight,
  Layers,
  Palette,
} from "lucide-react";
import { Canvas, useThree, useLoader, extend } from "@react-three/fiber";
import { OrbitControls } from '@react-three/drei';
import { STLLoader } from 'three-stdlib';
import * as THREE from "three";

// Add type declaration for OrbitControls
declare module '@react-three/fiber' {
  interface ThreeElements {
    orbitControls: typeof OrbitControls;
  }
}

// Extend React Three Fiber with additional components

// Add missing types for React Three Fiber
declare global {
  namespace JSX {
    interface IntrinsicElements {
      mesh: any;
      group: any;
      primitive: any;
      boxGeometry: any;
      sphereGeometry: any;
      cylinderGeometry: any;
      coneGeometry: any;
      torusGeometry: any;
      planeGeometry: any;
      ambientLight: any;
      directionalLight: any;
      pointLight: any;
      spotLight: any;
      meshStandardMaterial: any;
      meshBasicMaterial: any;
    }
  }
}

import { useDrag } from "@use-gesture/react";
import axios from "axios";
import type { ModelObject, ToolCategory } from "@/types/model-object";
import { TOOL_CATEGORIES } from "@/types/tool-category";

type ValidObjectType = "box" | "sphere" | "cylinder" | "cone" | "torus" | "plane";

// Define tool categories for CAD operations
const defaultToolCategories: ToolCategory[] = [
  {
    name: TOOL_CATEGORIES.CREATION,
    icon: <Box className="h-4 w-4" />,
    expanded: true,
    tools: [
      { name: "Box", icon: <Box className="h-3 w-3" />, action: "add-box" },
      { name: "Sphere", icon: <Circle className="h-3 w-3" />, action: "add-sphere" },
      { name: "Cylinder", icon: <Cylinder className="h-3 w-3" />, action: "add-cylinder" },
      { name: "Cone", icon: <Triangle className="h-3 w-3" />, action: "add-cone" },
      { name: "Torus", icon: <Circle className="h-3 w-3" />, action: "add-torus" },
      { name: "Plane", icon: <Square className="h-3 w-3" />, action: "add-plane" },
      { name: "Import STL", icon: <Layers className="h-3 w-3" />, action: "import-stl" }, // New: Import external models
    ],
  },
  {
    name: TOOL_CATEGORIES.TRANSFORMATION,
    icon: <Move3D className="h-4 w-4" />,
    expanded: false,
    tools: [
      { name: "Move", icon: <Move3D className="h-3 w-3" />, action: "move" },
      { name: "Rotate", icon: <RotateCcw className="h-3 w-3" />, action: "rotate" },
      { name: "Scale", icon: <Scale className="h-3 w-3" />, action: "scale" },
    ],
  },
  {
    name: TOOL_CATEGORIES.MEASUREMENT,
    icon: <Ruler className="h-4 w-4" />,
    expanded: false,
    tools: [
      { name: "Distance", icon: <Ruler className="h-3 w-3" />, action: "measure-distance" },
      { name: "Area", icon: <Square className="h-3 w-3" />, action: "measure-area" },
      { name: "Volume", icon: <Layers className="h-3 w-3" />, action: "measure-volume" },
      { name: "Stress", icon: <Palette className="h-3 w-3" />, action: "show-stress" }, // New: Visualize stress
    ],
  },
  {
    name: TOOL_CATEGORIES.HISTORY,
    icon: <Copy className="h-4 w-4" />,
    expanded: false,
    tools: [
      { name: "Undo", icon: <CornerDownRight className="h-3 w-3" />, action: "undo" },
      { name: "Redo", icon: <CornerDownRight className="h-3 w-3" />, action: "redo" },
      { name: "Delete", icon: <Copy className="h-3 w-3" />, action: "delete-selected" },
    ],
  },
];

export interface UseModelViewerReturn {
  objects: ModelObject[];
  selectedObject: string | null;
  setSelectedObject: (id: string | null) => void;
  addObject: (type: ModelObject["type"], id?: string, modelPath?: string, position?: [number, number, number], rotation?: [number, number, number], scale?: [number, number, number], color?: string) => void;
  importSTL: (file: File) => void; // New: Import STL files
  deleteSelected: () => void;
  updatePosition: (axis: "x" | "y" | "z", value: number) => void;
  updateScale: (axis: "x" | "y" | "z", value: number) => void;
  updateRotation: (axis: "x" | "y" | "z", value: number) => void; // New: Rotation control
  updateColor: (color: string) => void;
  showStress: () => void; // New: Visualize stress
  resetCamera: () => void;
  toolCategories: ToolCategory[];
  setToolCategories: React.Dispatch<React.SetStateAction<ToolCategory[]>>;
  addToHistory: (newState: ModelObject[]) => void;
  history: ModelObject[][];
  historyIndex: number;
  undo: () => void;
  redo: () => void;
  selectAll: () => void;
  setViewMode: (mode: "wireframe" | "shaded") => void;
  animate: () => void;
  applyMeshOperation: (operation: string) => void;
  applyModelingOperation: (operation: string, options?: { objectIds?: string[]; params?: Record<string, any> }) => void;
}

// Add this interface definition if it doesn't exist already
declare global {
  interface Window {
    THREE: any;
  }
}

// Component to render 3D models
const ModelRenderer: React.FC<{ objects: ModelObject[]; selectedObject: string | null }> = ({ objects, selectedObject }) => {
  return (
    <Canvas camera={{ position: [0, 5, 10], fov: 50 }}>
      {objects.map((obj) => (
        <Model key={obj.id} obj={obj} isSelected={obj.id === selectedObject} />
      ))}
      <OrbitControls />
      <ambientLight intensity={0.5} />
      <directionalLight position={[10, 10, 5]} intensity={1} />
    </Canvas>
  );
};

// Individual model component
const Model: React.FC<{ obj: ModelObject; isSelected: boolean }> = ({ obj, isSelected }) => {
  const geometryTypes: { [key: string]: React.ReactNode } = {
    box: <boxGeometry args={obj.scale} />,
    sphere: <sphereGeometry args={[obj.scale[0], 32, 32]} />,
    cylinder: <cylinderGeometry args={[obj.scale[0], obj.scale[0], obj.scale[1], 32]} />,
    cone: <coneGeometry args={[obj.scale[0], obj.scale[1], 32]} />,
    torus: <torusGeometry args={[obj.scale[0], obj.scale[1] / 4, 16, 100]} />,
    plane: <planeGeometry args={obj.scale} />,
    stl: obj.modelPath ? <STLModel modelPath={obj.modelPath} /> : null,
  };

  return (
    <mesh
      position={obj.position}
      rotation={obj.rotation}
      scale={obj.scale}
      material={new THREE.MeshStandardMaterial({
        color: isSelected ? '#ff0000' : obj.color, // Highlight selected
        metalness: obj.stress ? 0.5 : 0, // Stress visualization
        roughness: obj.stress ? 1 - obj.stress : 1,
      })}
    >
      {geometryTypes[obj.type]}
    </mesh>
  );
};

// STL model loader
const STLModel: React.FC<{ modelPath: string }> = ({ modelPath }) => {
  const geometry = useLoader(STLLoader, modelPath);
  return <primitive object={geometry} />;
};

export function useModelViewer(): UseModelViewerReturn {
  const [objects, setObjects] = useState<ModelObject[]>([
    {
      id: "1",
      type: "box",
      position: [0, 0.5, 0],
      rotation: [0, 0, 0],
      scale: [1, 1, 1],
      color: "#8b5cf6",
      createdAt: Date.now(),
      updatedAt: Date.now(),
    },
  ]);
  const [history, setHistory] = useState<ModelObject[][]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [selectedObject, setSelectedObject] = useState<string | null>(null);
  const [toolCategories, setToolCategories] = useState(defaultToolCategories);

  // History management
  const addToHistory = useCallback(
    (newState: ModelObject[]) => {
      setHistory((prev) => [...prev.slice(0, historyIndex + 1), newState]);
      setHistoryIndex((idx) => idx + 1);
      // Sync with backend
      axios.post("/api/save-model", { objects: newState }).catch((error) => {
        console.error("Failed to sync with backend:", error);
      });
    },
    [historyIndex],
  );

  const undo = useCallback(() => {
    if (historyIndex >= 0) {
      setHistoryIndex((idx) => idx - 1);
      setObjects([...history[historyIndex]]);
    }
  }, [historyIndex, history]);

  const redo = useCallback(() => {
    if (historyIndex < history.length - 1) {
      setHistoryIndex((idx) => idx + 1);
      setObjects([...history[historyIndex + 1]]);
    }
  }, [historyIndex, history]);

  // Add new object
  const addObject = useCallback(
    (type: ModelObject["type"], id?: string, modelPath?: string, position?: [number, number, number], rotation?: [number, number, number], scale?: [number, number, number], color?: string) => {
      const newObject: ModelObject = {
        id: id || Date.now().toString(),
        type,
        position: position || [Math.random() * 4 - 2, 0.5, Math.random() * 4 - 2],
        rotation: rotation || [0, 0, 0],
        scale: scale || [1, 1, 1],
        color: color || `hsl(${Math.random() * 360}, 70%, 60%)`,
        modelPath,
        createdAt: Date.now(),
        updatedAt: Date.now(),
      };
      setObjects((prev) => [...prev, newObject]);
      addToHistory([...objects, newObject]);
      setSelectedObject(newObject.id);
      // Notify ConceivoAI
      axios.post("/api/conceivo", {
        message: "add",
        params: { type },
      });
    },
    [objects, addToHistory],
  );

  // Import STL file
  const importSTL = useCallback(
    (file: File) => {
      const url = URL.createObjectURL(file);
      const newObject: ModelObject = {
        id: Date.now().toString(),
        type: "stl",
        position: [Math.random() * 4 - 2, 0.5, Math.random() * 4 - 2],
        rotation: [0, 0, 0],
        scale: [1, 1, 1],
        color: `hsl(${Math.random() * 360}, 70%, 60%)`,
        modelPath: url,
        createdAt: Date.now(),
        updatedAt: Date.now(),
      };
      setObjects((prev) => [...prev, newObject]);
      addToHistory([...objects, newObject]);
      setSelectedObject(newObject.id);
      // Store in ConceivoAI
      axios.post("/api/conceivo/component", {
        id: newObject.id,
        name: file.name,
        type: "imported",
        model_path: url,
      });
    },
    [objects, addToHistory],
  );

  // Delete selected object
  const deleteSelected = useCallback(() => {
    if (selectedObject) {
      const newObjects = objects.filter((obj) => obj.id !== selectedObject);
      setObjects(newObjects);
      addToHistory(newObjects);
      setSelectedObject(null);
    }
  }, [selectedObject, objects, addToHistory]);

  // Update position with drag-and-drop
  const updatePosition = useCallback(
    (axis: "x" | "y" | "z", value: number) => {
      if (!selectedObject) return;
      const index = objects.findIndex((obj) => obj.id === selectedObject);
      if (index !== -1) {
        const newObj = { ...objects[index] };
        newObj.position = newObj.position.map((v, i) =>
          i === ["x", "y", "z"].indexOf(axis) ? value : v,
        ) as [number, number, number];
        newObj.updatedAt = Date.now();
        const newObjs = [...objects];
        newObjs[index] = newObj;
        setObjects(newObjs);
        addToHistory(newObjs);
      }
    },
    [selectedObject, objects, addToHistory],
  );

  // Update scale
  const updateScale = useCallback(
    (axis: "x" | "y" | "z", value: number) => {
      if (!selectedObject) return;
      const index = objects.findIndex((obj) => obj.id === selectedObject);
      if (index !== -1) {
        const newObj = { ...objects[index] };
        newObj.scale = newObj.scale.map((v, i) =>
          i === ["x", "y", "z"].indexOf(axis) ? Math.max(0.1, value) : v,
        ) as [number, number, number];
        newObj.updatedAt = Date.now();
        const newObjs = [...objects];
        newObjs[index] = newObj;
        setObjects(newObjs);
        addToHistory(newObjs);
      }
    },
    [selectedObject, objects, addToHistory],
  );

  // Update rotation
  const updateRotation = useCallback(
    (axis: "x" | "y" | "z", value: number) => {
      if (!selectedObject) return;
      const index = objects.findIndex((obj) => obj.id === selectedObject);
      if (index !== -1) {
        const newObj = { ...objects[index] };
        newObj.rotation = newObj.rotation.map((v, i) =>
          i === ["x", "y", "z"].indexOf(axis) ? value : v,
        ) as [number, number, number];
        newObj.updatedAt = Date.now();
        const newObjs = [...objects];
        newObjs[index] = newObj;
        setObjects(newObjs);
        addToHistory(newObjs);
      }
    },
    [selectedObject, objects, addToHistory],
  );

  // Update color
  const updateColor = useCallback(
    (color: string) => {
      if (!selectedObject) return;
      const index = objects.findIndex((obj) => obj.id === selectedObject);
      if (index !== -1) {
        const newObj = { ...objects[index], color, updatedAt: Date.now() };
        const newObjs = [...objects];
        newObjs[index] = newObj;
        setObjects(newObjs);
        addToHistory(newObjs);
      }
    },
    [selectedObject, objects, addToHistory],
  );

  // Show stress visualization
  const showStress = useCallback(async () => {
    if (!selectedObject) return;
    try {
      const response = await axios.post("/api/simulate", { objectId: selectedObject });
      const stress = response.data.stress; // Assume backend returns normalized stress (0-1)
      const index = objects.findIndex((obj) => obj.id === selectedObject);
      if (index !== -1) {
        const newObj = { ...objects[index], stress, updatedAt: Date.now() };
        const newObjs = [...objects];
        newObjs[index] = newObj;
        setObjects(newObjs);
        addToHistory(newObjs);
      }
    } catch (error) {
      console.error("Failed to fetch stress data:", error);
    }
  }, [selectedObject, objects, addToHistory]);

  // Reset camera
  const resetCamera = useCallback(() => {
    console.log("Resetting camera...");
    // Implement camera reset logic if needed
  }, []);

  // Drag-and-drop controls
  const bindDrag = useDrag(({ offset: [x, y] }) => {
    if (selectedObject) {
      updatePosition("x", x / 100);
      updatePosition("y", -y / 100);
    }
  });

  // Select all objects
  const selectAll = useCallback(() => {
    // Implement select all logic
    console.log("Select all objects");
    // This would typically set selectedObject to some value that represents multiple selections
  }, []);

  // Set view mode (wireframe/shaded)
  const setViewMode = useCallback((mode: "wireframe" | "shaded") => {
    // Implement view mode logic
    console.log(`View mode set to ${mode}`);
    // This would typically update some state that controls the rendering
  }, []);

  // Animate objects
  const animate = useCallback(() => {
    // Implement animation logic
    console.log("Animation started");
    // This would typically enable/disable animation controls
  }, []);

  // Apply mesh operation (e.g., extrude, bevel)
  const applyMeshOperation = useCallback((operation: string) => {
    // Implement mesh operation
    console.log(`Applying mesh operation: ${operation}`);
    // This would typically modify the geometry of selected objects
  }, []);

  // Apply modeling operation (e.g., union, subtract)
  const applyModelingOperation = useCallback((operation: string, options?: { objectIds?: string[]; params?: Record<string, any> }) => {
    // Implement modeling operation
    console.log(`Applying modeling operation: ${operation}`, options);
    // This would typically combine multiple objects using the specified operation
  }, []);

  return {
    objects,
    selectedObject,
    setSelectedObject,
    addObject,
    importSTL,
    deleteSelected,
    updatePosition,
    updateScale,
    updateRotation,
    updateColor,
    showStress,
    resetCamera,
    toolCategories,
    setToolCategories,
    history,
    historyIndex,
    addToHistory,
    undo,
    redo,
    selectAll,
    setViewMode,
    animate,
    applyMeshOperation,
    applyModelingOperation,
  };
}

// Example usage in a component
export const ModelViewerComponent: React.FC = () => {
  const {
    objects,
    selectedObject,
    importSTL,
    addObject,
    updatePosition,
    updateScale,
    updateRotation,
    updateColor,
    showStress,
    toolCategories,
  } = useModelViewer();

  return (
    <div className="flex h-screen">
      <div className="w-1/4 p-4 bg-gray-100">
        <h2>Tools</h2>
        {toolCategories.map((category) => (
          <div key={category.name}>
            <h3>{category.name}</h3>
            {category.tools.map((tool) => (
              <button
                key={tool.name}
                onClick={() => {
                  if (tool.action.startsWith("add-")) {
                    const objType = tool.action.replace("add-", "") as ValidObjectType;
                    addObject(objType);
                  }
                  if (tool.action === "import-stl") {
                    const input = document.createElement("input");
                    input.type = "file";
                    input.accept = ".stl";
                    input.onchange = (e) => {
                      const file = (e.target as HTMLInputElement).files?.[0];
                      if (file) importSTL(file);
                    };
                    input.click();
                  }
                  if (tool.action === "show-stress") showStress();
                }}
              >
                {tool.icon} {tool.name}
              </button>
            ))}
          </div>
        ))}
        <input
          type="color"
          onChange={(e) => updateColor(e.target.value)}
          disabled={!selectedObject}
        />
      </div>
      <div className="w-3/4">
        <ModelViewerComponent />
      </div>
    </div>
  );
};