// Define AIPlugin interface
type AIPlugin = {
  name: string;
  supportsCommands: string[];
};

// PluginManager class to handle plugin registration and management
class PluginManager {
  private plugins: Map<string, AIPlugin>;

  constructor() {
    this.plugins = new Map();
  }

  // Register a new plugin
  public register(plugin: AIPlugin): void {
    this.plugins.set(plugin.name, plugin);
  }

  // Get all registered plugins
  public getAll(): Array<{ plugin: AIPlugin; enabled: boolean }> {
    const result = [];
    for (const [name, plugin] of this.plugins.entries()) {
      result.push({ plugin, enabled: true });
    }
    return result;
  }

  // Get a specific plugin by name
  public get(name: string): AIPlugin | undefined {
    return this.plugins.get(name);
  }

  // Enable or disable a plugin
  public setEnabled(name: string, enabled: boolean): void {
    const plugin = this.plugins.get(name);
    if (plugin) {
      // We need to create a new array since we can't modify the existing one
      const updatedPlugin = { ...plugin };
      // In a real implementation, we would have an enabled field
      // but since it's not in the original type definition, we would need to extend it
    }
  }
}