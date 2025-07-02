export const TOOLS = [
  {
    type: "function",
    name: "addObject",
    description: "Add a new 3D object to the scene",
    parameters: {
      type: "object",
      properties: {
        type: {
          type: "string",
          enum: ["box", "sphere", "cylinder", "cone", "torus", "plane", "wedge"],
          description: "The type of 3D object to create",
        },
        position: {
          type: "array",
          items: { type: "number" },
          minItems: 3,
          maxItems: 3,
          default: [0, 0.5, 0],
          description: "Position as [x, y, z] coordinates",
        },
        rotation: {
          type: "array",
          items: { type: "number" },
          minItems: 3,
          maxItems: 3,
          default: [0, 0, 0],
          description: "Rotation as [x, y, z] angles in radians",
        },
        scale: {
          type: "array",
          items: { type: "number" },
          minItems: 3,
          maxItems: 3,
          default: [1, 1, 1],
          description: "Scale as [x, y, z] multipliers",
        },
        color: {
          type: "string",
          pattern: "^#[0-9a-fA-F]{6}$",
          default: "#8b5cf6",
          description: "Hex color code",
        },
      },
      required: ["type"],
    },
  },
  {
    type: "function",
    name: "updateProperty",
    description: "Update a property of the selected object",
    parameters: {
      type: "object",
      properties: {
        property: {
          type: "string",
          enum: ["position", "rotation", "scale", "color", "wireframe", "materialType"],
          description: "The property to update",
        },
        value: {
          oneOf: [
            { type: "string" },
            { type: "number" },
            { type: "boolean" },
            {
              type: "array",
              items: { type: "number" },
              minItems: 3,
              maxItems: 3,
            },
          ],
          description: "The new value for the property",
        },
      },
      required: ["property", "value"],
    },
  },
  {
    type: "function",
    name: "deleteSelected",
    description: "Delete the currently selected object",
    parameters: {
      type: "object",
      properties: {},
    },
  },
  {
    type: "function",
    name: "measureDistance",
    description: "Measure distance between two points or objects",
    parameters: {
      type: "object",
      properties: {
        point1: {
          type: "array",
          items: { type: "number" },
          minItems: 3,
          maxItems: 3,
          description: "First point as [x, y, z]",
        },
        point2: {
          type: "array",
          items: { type: "number" },
          minItems: 3,
          maxItems: 3,
          description: "Second point as [x, y, z]",
        },
      },
      required: ["point1", "point2"],
    },
  },
]

export type ToolName = "addObject" | "updateProperty" | "deleteSelected" | "measureDistance"

export interface ToolCall {
  name: ToolName
  parameters: Record<string, any>
}
