from typing import Optional, Dict, List
import json
import logging
from datetime import datetime, timedelta
from fuzzywuzzy import fuzz  # pylint: disable=E0401

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class EngineeringComponent:
    def __init__(
        self,
        id: str,
        name: str,
        type: str,
        description: str,
        function: str = "",
        dependencies: Optional[List[str]] = None,
        performanceMetrics: Optional[Dict] = None,
        failureModes: Optional[List[str]] = None
    ):
        self.id = id
        self.name = name
        self.type = type
        self.description = description
        self.function = function
        self.dependencies = dependencies if dependencies is not None else []
        self.performanceMetrics = performanceMetrics if performanceMetrics is not None else {}
        self.failureModes = failureModes if failureModes is not None else []

class ConceivoAI:
    def __init__(self, components_file: Optional[str] = None, max_knowledge_entries: int = 100):
        """
        Initialize ConceivoAI with optional components file and knowledge base size limit.
        
        Args:
            components_file (str, optional): Path to JSON file containing component data.
            max_knowledge_entries (int): Maximum number of knowledge base entries to prevent memory issues.
        """
        self.max_knowledge_entries = max_knowledge_entries
        self.knowledge_base: Dict[str, Dict[str, str]] = {}  # {topic: {content, timestamp}}
        self.components: List[EngineeringComponent] = []
        
        # Load default components or from file
        if components_file:
            self.load_components(components_file)
        else:
            self._load_default_components()
        
        logger.info(f"Initialized ConceivoAI with {len(self.components)} components.")

    def _load_default_components(self):
        """Load default components if no file is provided."""
        self.components = [
            EngineeringComponent(
                id="comp-001",
                name="Rocket Nozzle",
                type="propulsion",
                description="Expands and accelerates exhaust gases to generate thrust.",
                function="Converts thermal energy into kinetic energy for propulsion.",
                dependencies=["comp-002"],
                performanceMetrics={"thrust": 50000, "efficiency": 0.95, "weight": 150},
                failureModes=["thermal cracking", "erosion"],
            ),
            EngineeringComponent(
                id="comp-002",
                name="Combustion Chamber",
                type="propulsion",
                description="Where fuel and oxidizer combust to produce high-pressure gas.",
                function="Generates high-temperature gas for nozzle acceleration.",
                dependencies=["comp-001"],
                performanceMetrics={"pressure": 200, "temperature": 3000, "weight": 200},
                failureModes=["overpressure", "material fatigue"],
            ),
            EngineeringComponent(
                id="comp-003",
                name="Rocket Engine",
                type="propulsion",
                description="Combines fuel and oxidizer for thrust.",
                function="Generates thrust by expelling high-speed exhaust.",
                dependencies=[],
                performanceMetrics={"pressure": 200, "temperature": 3000, "weight": 200},
                failureModes=["overpressure", "material fatigue"],
            ),
        ]

    def load_components(self, file_path: str) -> None:
        """
        Load components from a JSON file.
        
        Args:
            file_path (str): Path to the JSON file.
        
        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the JSON is invalid or missing required fields.
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                self.components = []
                for comp_data in data:
                    if not all(key in comp_data for key in ["id", "name", "type", "description"]):
                        raise ValueError(f"Invalid component data: missing required fields in {comp_data}")
                    self.components.append(EngineeringComponent(**comp_data))
            logger.info(f"Loaded {len(self.components)} components from {file_path}")
        except FileNotFoundError:
            logger.error(f"Components file {file_path} not found")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {file_path}: {str(e)}")
            raise ValueError(f"Invalid JSON in {file_path}: {str(e)}")

    def add_component(self, component: EngineeringComponent) -> None:
        """
        Add a new component to the system.
        
        Args:
            component (EngineeringComponent): The component to add.
        
        Raises:
            ValueError: If the component ID already exists.
        """
        if self.get_component_by_id(component.id):
            logger.error(f"Component with ID {component.id} already exists")
            raise ValueError(f"Component with ID {component.id} already exists")
        self.components.append(component)
        logger.info(f"Added component {component.id}: {component.name}")

    def update_component(self, id: str, updated_component: EngineeringComponent) -> None:
        """
        Update an existing component.
        
        Args:
            id (str): ID of the component to update.
            updated_component (EngineeringComponent): Updated component data.
        
        Raises:
            ValueError: If the component ID doesn't exist.
        """
        for i, comp in enumerate(self.components):
            if comp.id == id:
                self.components[i] = updated_component
                logger.info(f"Updated component {id}: {updated_component.name}")
                return
        logger.error(f"Component {id} not found for update")
        raise ValueError(f"Component {id} not found")

    def delete_component(self, id: str) -> None:
        """
        Delete a component by ID.
        
        Args:
            id (str): ID of the component to delete.
        
        Raises:
            ValueError: If the component ID doesn't exist.
        """
        for i, comp in enumerate(self.components):
            if comp.id == id:
                self.components.pop(i)
                logger.info(f"Deleted component {id}")
                return
        logger.error(f"Component {id} not found for deletion")
        raise ValueError(f"Component {id} not found")

    def learn(self, topic: str, content: str) -> None:
        """
        Store new knowledge in the knowledge base, with timestamp and size management.
        
        Args:
            topic (str): Topic of the knowledge.
            content (str): Content to store.
        """
        topic = topic.lower()
        self.knowledge_base[topic] = {
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        # Prune old entries if knowledge base exceeds max size
        if len(self.knowledge_base) > self.max_knowledge_entries:
            oldest_topic = min(
                self.knowledge_base,
                key=lambda k: datetime.fromisoformat(self.knowledge_base[k]["timestamp"])
            )
            del self.knowledge_base[oldest_topic]
            logger.info(f"Pruned oldest knowledge entry: {oldest_topic}")
        
        logger.info(f"Learned new topic: {topic}")

    def get_knowledge(self, topic: str) -> Optional[str]:
        """
        Retrieve knowledge by topic.
        
        Args:
            topic (str): Topic to retrieve.
        
        Returns:
            str or None: Content if found, else None.
        """
        entry = self.knowledge_base.get(topic.lower())
        return entry["content"] if entry else None

    def get_component_by_id(self, id: str) -> Optional[EngineeringComponent]:
        """
        Retrieve a component by ID.
        
        Args:
            id (str): Component ID.
        
        Returns:
            EngineeringComponent or None: Component if found, else None.
        """
        return next((comp for comp in self.components if comp.id == id), None)

    def analyze_system(self, component_ids: List[str]) -> str:
        """
        Analyze a system of components, including dependencies and knowledge base insights.
        
        Args:
            component_ids (List[str]): List of component IDs to analyze.
        
        Returns:
            str: Analysis report.
        
        Raises:
            ValueError: If no valid component IDs are provided.
        """
        if not component_ids:
            logger.error("No component IDs provided for analysis")
            raise ValueError("At least one component ID must be provided")
        
        analysis = []
        for id in component_ids:
            comp = self.get_component_by_id(id)
            if not comp:
                logger.warning(f"Component {id} not found")
                analysis.append(f"Component {id} not found.")
                continue
            
            # Validate dependencies
            dep_status = []
            for dep_id in comp.dependencies:
                dep_comp = self.get_component_by_id(dep_id)
                dep_status.append(f"Dependency {dep_id}: {'Found' if dep_comp else 'Missing'}")
            
            # Find relevant knowledge with fuzzy matching
            additional_insights = ""
            for topic, entry in self.knowledge_base.items():
                if (fuzz.partial_ratio(comp.type.lower(), topic) > 80 or 
                    fuzz.partial_ratio(comp.name.lower(), topic) > 80):
                    additional_insights += f"\nAdditional insights about {topic}: {entry['content'][:150]}..."
            
            analysis.append(
                f"Component: {comp.name} ({comp.type})\n"
                f"Description: {comp.description}\n"
                f"Function: {comp.function}\n"
                f"Performance: {comp.performanceMetrics}\n"
                f"Failure Modes: {', '.join(comp.failureModes)}\n"
                f"Dependencies: {', '.join(dep_status) if dep_status else 'None'}\n"
                f"{additional_insights}"
            )
        
        return "\n\n".join(analysis)