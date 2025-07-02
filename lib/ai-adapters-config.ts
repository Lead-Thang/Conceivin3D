// lib/ai-adapters-config.ts
// Configuration for AI providers

// Configuration for AI providers
export const aiProviders = {
  // 2D Image Generation Providers
  'dall-e': {
    name: 'DALL-E',
    type: 'image-generation',
    description: 'High-quality image generation from text prompts',
    enabled: true
  },
  'gemini': {
    name: 'Gemini',
    type: 'image-generation',
    description: 'Google\'s powerful multi-modal model for image generation',
    enabled: true
  },
  'grok': {
    name: 'Grok',
    type: 'image-generation',
    description: 'X\'s advanced AI model for image generation',
    enabled: true
  },
  // 3D Modeling Providers
  'hunyuan3d': {
    name: 'Hunyuan3D',
    type: 'modeling',
    description: 'Advanced 3D modeling and generation capabilities',
    enabled: true
  },
  'rendair': {
    name: 'RendAir',
    type: 'modeling',
    description: 'Cloud-based 3D modeling and rendering service',
    enabled: true
  },
  'tribo': {
    name: 'Tribo',
    type: 'modeling',
    description: 'AI-powered 3D modeling and animation tools',
    enabled: true
  }
}

// Type definitions
export type AIProviderType = 
  | 'image-generation'
  | 'modeling'

export type AIProviderId = 
  | 'dall-e'
  | 'gemini'
  | 'grok'
  | 'hunyuan3d'
  | 'rendair'
  | 'tribo'

export interface AIProviderInfo {
  name: string
  type: AIProviderType
  description: string
  enabled: boolean
}