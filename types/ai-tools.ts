// types/ai-tools.ts
export type ToolName =
  | "add"
  | "delete"
  | "update"
  | "move"
  | "scale"
  | "rotate"
  | "color"
  | "measure"
  | "view"
  | "reset"
  | "undo"
  | "redo"

export interface ToolCall {
  name: ToolName
  args: Record<string, any>
}

export interface AIResponse {
  text: string
  audioUrl?: string
  command?: ToolCall | null
}
