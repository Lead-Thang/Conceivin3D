"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.ConceivoAI = void 0;
var ConceivoAI = /** @class */ (function () {
    function ConceivoAI() {
        this.components = [
            {
                id: "comp-001",
                name: "Rocket Nozzle",
                type: "propulsion",
                description: "Expands and accelerates exhaust gases to generate thrust.",
                function: "Converts thermal energy into kinetic energy for propulsion.",
                dependencies: ["comp-002"],
                performanceMetrics: { thrust: 50000, efficiency: 0.95, weight: 150 },
                failureModes: ["thermal cracking", "erosion"],
            },
            {
                id: "comp-002",
                name: "Combustion Chamber",
                type: "propulsion",
                description: "Where fuel and oxidizer combust to produce high-pressure gas.",
                function: "Generates high-temperature gas for nozzle acceleration.",
                dependencies: [],
                performanceMetrics: { pressure: 200, temperature: 3000, weight: 200 },
                failureModes: ["overpressure", "material fatigue"],
            },
        ];
        this.knowledgeBase = {};
    }
    // Method to add new knowledge
    ConceivoAI.prototype.learn = function (topic, content) {
        this.knowledgeBase[topic.toLowerCase()] = content;
    };
    // Method to retrieve learned knowledge
    ConceivoAI.prototype.getKnowledge = function (topic) {
        return this.knowledgeBase[topic.toLowerCase()];
    };
    ConceivoAI.prototype.getComponentById = function (id) {
        return this.components.find(function (comp) { return comp.id === id; });
    };
    ConceivoAI.prototype.analyzeSystem = function (componentIds) {
        var _this = this;
        // Enhanced to use learned knowledge
        var analysis = componentIds.map(function (id) {
            var comp = _this.getComponentById(id);
            if (!comp)
                return "Component ".concat(id, " not found.");
            // Check for related knowledge
            var additionalInsights = "";
            Object.entries(_this.knowledgeBase).forEach(function (_a) {
                var topic = _a[0], content = _a[1];
                if (topic.includes(comp.type) || topic.includes(comp.name.toLowerCase())) {
                    additionalInsights += "\nAdditional insights about ".concat(topic, ": ").concat(content.substring(0, 150), "...");
                }
            });
            return "Component: ".concat(comp.name, " (").concat(comp.type, ")\nDescription: ").concat(comp.description, "\nPerformance: ").concat(JSON.stringify(comp.performanceMetrics), "\nFailure Modes: ").concat(comp.failureModes.join(", "), "\n").concat(additionalInsights);
        });
        return analysis.join("\n\n");
    };
    return ConceivoAI;
}());
exports.ConceivoAI = ConceivoAI;
