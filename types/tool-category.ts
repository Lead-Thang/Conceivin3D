// Tool Categories
export const TOOL_CATEGORIES = {
  CREATION: "Creation",
  TRANSFORMATION: "Transformation",
  SELECTION: "Selection",
  HISTORY: "History",
  VIEW: "View",
  MESH_EDITING: "Mesh Editing",
  MEASUREMENT: "Measurement",
  MODELING: "Modeling",
} as const;

type ToolCategoryKey = keyof typeof TOOL_CATEGORIES;

export type ToolAction =
  // Creation
  | "add-box"
  | "add-sphere"
  | "add-cylinder"
  | "add-cone"
  | "add-torus"
  | "add-plane"
  | "add-wedge"
  
  // Transformation
  | "move"
  | "rotate"
  | "scale"
  | "change-color"

  // Selection & Navigation
  | "select-all"
  | "reset-camera"

  // History / State Management
  | "undo"
  | "redo"
  | "delete-selected"

  // View Mode Control
  | "view-wireframe"
  | "view-shaded"
  | "animate"

  // Mesh-Level Editing (Advanced)
  | "bevel-edges"
  | "bridge-edge-loops"
  | "loop-cut-and-slide"
  | "offset-edge-slide"
  | "knife-topology-tool"
  | "bisect"
  | "edge-split"
  | "edge-crease"
  | "edge-bevel-weight"
  | "mark-seam"
  | "clear-seam"
  | "mark-sharp"
  | "clear-sharp"
  | "set-sharpness-by-angle"
  | "un-subdivide"
  | "split"
  | "separate"
  | "dissolve-edges"
  | "delete-edges"
  | "delete-faces"
  | "delete-vertices"

  // Measurement
  | "measure-distance"
  | "measure-volume"
  | "measure-area"

  // Boolean Operations
  | "union"
  | "subtract"
  | "intersect"

  // Modeling Helpers
  | "extrude"
  | "revolve"
  | "sweep"
  | "mirror"
  | "pattern"
  | "duplicate-object";