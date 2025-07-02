// types/ai-provider.ts
// Base interface for all AI providers
export interface AIProvider {
  sendMessage: (messages: Array<{ role: 'user' | 'assistant'; content: string }>, userId?: string) => Promise<AIResponse>
}

// Extended interface for 2D image generation providers
export interface ImageGenerationProvider extends AIProvider {
  generateImage: (prompt: string, options?: ImageGenerationOptions) => Promise<string> // Returns image URL
}

// Extended interface for 3D modeling providers
export interface ModelingProvider extends AIProvider {
  generate3DModel: (prompt: string, options?: ModelingOptions) => Promise<string> // Returns model URL
  executeModelingCommand: (command: ModelingCommand) => Promise<ModelingResponse>
}

// Types for AI response
export interface AIResponse {
  message: string
  command?: CommandResponse | null
}

export interface ImageGenerationOptions {
  size?: string
  quality?: number
  style?: string
}

export interface ModelingOptions {
  complexity?: string
  detailLevel?: number
  format?: string
}

export interface ModelingCommand {
  type: CommandType
  parameters: Record<string, any>
}

export type CommandType = 
  'add' | 
  'remove' | 
  'modify' | 
  'scale' | 
  'rotate' | 
  'position' | 
  'material' | 
  'generate-mesh'

export interface ModelingResponse {
  success: boolean
  result?: any
  error?: string
}

export interface CommandResponse {
  type: 'image-generation' | 'modeling' | 'text-response'
  data: any
}
