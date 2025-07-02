// types/ai-plugin.ts
export interface AIPlugin {
  name: string
  supportsCommands: string[]
  execute(command: string, context: any): Promise<any>
}
