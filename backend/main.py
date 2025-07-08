from fastapi import FastAPI, Request, HTTPException, WebSocket
from fastapi.responses import JSONResponse
from pydantic import BaseModel, SecretStr
from contextlib import asynccontextmanager
import asyncio
import torch
import torch.nn as nn
from crawl4ai.async_webcrawler import AsyncWebCrawler
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from crawl4ai.types import LLMConfig
import os
import httpx
from fastapi import FastAPI, Request, HTTPException, WebSocket
from fastapi.responses import JSONResponse
from pydantic import BaseModel, SecretStr
from contextlib import asynccontextmanager
import asyncio
import torch
import torch.nn as nn
from crawl4ai.async_webcrawler import AsyncWebCrawler
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from crawl4ai.types import LLMConfig
from backend.crawl_config import SEED_URLS
from langchain_mistralai.chat_models import ChatMistralAI
import datetime
import sys
import os
from typing import TypedDict, Union, Optional, List, Dict

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Define global availability flag
CONCEIVO_AI_AVAILABLE = False  # Will be set to True when ConceivoAI is successfully initialized

from crawl4ai.async_configs import LLMConfig as CrawlLLMConfig
import httpx
from backend.utils import GrokAI
import pandas as pd
import numpy as np
import pyvista as pv
import trimesh
from trimesh import util, boolean
from fastapi.websockets import WebSocketDisconnect
import random
import json

# Add Tripo API integration
TRIPO_API_KEY = os.getenv("TRIPO_API_KEY")

class EngineeringComponent:
    def __init__(self, id: str, name: str, type: str, description: str = "", model_path: Optional[str] = None, metrics: Optional[Dict] = None):
        self.id = id
        self.name = name
        self.type = type
        self.description = description
        self.model_path = model_path
        self.metrics = metrics or {}
        self.performanceMetrics = self.metrics.get("performance", {})
        self.failureModes = self.metrics.get("failureModes", [])

class ConceivoAI:
    def __init__(self):
        self.components: List[EngineeringComponent] = []
        self.type = "default"

    def learn(self, topic: str, content: str):
        print(f"Learning about {topic}: {content}")

    def add_component(self, component: EngineeringComponent):
        self.components.append(component)

    def update_component(self, id: str, updates: Dict):
        for comp in self.components:
            if comp.id == id:
                for key, value in updates.items():
                    setattr(comp, key, value)
                break

    def get_components(self):
        return self.components

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.crawler = AsyncWebCrawler()
        configure_app(app)
        
        # Load system instructions first
        system_instructions = ""
        try:
            with open("system_instructions.md", "r") as f:
                system_prompt = f.read()
            print("[DEBUG] Loaded system instructions:", system_prompt[:100] + '...')  # Preview first 100 chars
            app.state.conceivo_ai.learn("system_instructions", system_prompt)
        except FileNotFoundError:
            print("system_instructions.md not found. Using default instructions.")
            system_instructions = """You are CONCEIVO AI, a specialized engineering design assistant.
- Focus on 3D modeling and component design
- Provide detailed technical specifications
- Prioritize structural integrity and material properties
- Suggest manufacturing processes and tolerances
- Optimize for cost-effectiveness and performance"""
            app.state.conceivo_ai.learn("system_instructions", system_instructions)
        
        # Load initial knowledge
        app.state.initial_knowledge = await crawl_engineering_knowledge() if app.state.llm_config else "No API key, crawling disabled."
        global CONCEIVO_AI_AVAILABLE
        CONCEIVO_AI_AVAILABLE = True
        
        start_background_learning(app)
    except Exception as e:
        print(f"Error during lifespan setup: {e}")
        app.state.initial_knowledge = "Initial knowledge fetch failed. Try 'learn more' to update."
    yield
    await app.state.crawler.close()

def configure_app(app: FastAPI):
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    xai_api_key = os.getenv("XAI_API_KEY")
    if mistral_api_key:
        secret_mistral_api_key = SecretStr(mistral_api_key)
        app.state.llm_config = CrawlLLMConfig(
            provider="mistral/gpt-4",
            api_token=secret_mistral_api_key.get_secret_value()
        )
        app.state.extraction_strategy = LLMExtractionStrategy(
            llm_config=app.state.llm_config,
            extraction_type="text",
            instruction="Extract extremely high-level engineering concepts."
        )
        app.state.mistral_client = ChatMistralAI(api_key=secret_mistral_api_key)
    else:
        print("MISTRAL_API_KEY environment variable is not set")
        app.state.llm_config = None
        app.state.extraction_strategy = None
    if xai_api_key:
        app.state.xai_api_key = SecretStr(xai_api_key)
    else:
        print("XAI_API_KEY environment variable is not set")
        app.state.xai_api_key = None

app = FastAPI(lifespan=lifespan)

class AIRequest(BaseModel):
    message: str
    metrics: list[float] = []
    selected_part: str | None = None

class AIResponse(BaseModel):
    message: str
    predicted_efficiency: float | None = None
    command: dict | None = None
    new_knowledge: str | None = None
    edit_result: str | None = None

class ModelingRequest(BaseModel):
    operation: str
    objectIds: List[str]
    params: Optional[Dict] = None

class ConceivoNet(nn.Module):
    def __init__(self, input_size: int = 3, hidden_size: int = 10, output_size: int = 1):
        """
        Neural network model for predicting engineering component efficiency.
        
        Args:
            input_size: Number of input features (default: 3, matching training data)
            hidden_size: Number of neurons in hidden layer (default: 10)
            output_size: Number of output features (default: 1)
        """
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        return torch.sigmoid(out)
        return out

model_ready = False
try:
    model_path = "conceivo_model.pth"
    
    if os.path.exists(model_path):
        try:
            # Load model with correct dimensions
            model = ConceivoNet(input_size=3)  # Match saved model's architecture
            model.load_state_dict(torch.load(model_path))
            model.eval()
            model_ready = True
            print("Model loaded successfully")
        except Exception as load_error:
            print(f"Error loading model weights: {str(load_error)}. Using random initialization.")
            # Create new instance with default dimensions for training
            model = ConceivoNet(input_size=3)
            
            # Add synthetic data based on engineering principles
            synthetic_data = [
                # Varying parameters to teach relationships
                [0.9, 0.9, 0.9, 0.9, 0.6],  # High cost, high strength, long build time, high stress -> lower efficiency
                [0.3, 0.3, 0.3, 0.3, 0.9],  # Low cost, low strength, short build time, low stress -> higher efficiency
                [0.5, 0.9, 0.4, 0.3, 0.92],  # Balanced with high material strength
                [0.7, 0.6, 0.8, 0.5, 0.75],  # High build time impacts efficiency
            ]
    else:
        print("Model file not found. Using randomly initialized weights. Predictions disabled.")
        # Create new instance with default dimensions for training
        model = ConceivoNet(input_size=3)
        
        # Add synthetic data based on engineering principles
        synthetic_data = [
            # Varying parameters to teach relationships
            [0.9, 0.9, 0.9, 0.9, 0.6],  # High cost, high strength, long build time, high stress -> lower efficiency
            [0.3, 0.3, 0.3, 0.3, 0.9],  # Low cost, low strength, short build time, low stress -> higher efficiency
            [0.5, 0.9, 0.4, 0.3, 0.92],  # Balanced with high material strength
            [0.7, 0.6, 0.8, 0.5, 0.75],  # High build time impacts efficiency
        ]
        
except Exception as e:
    print(f"Critical error during model initialization: {str(e)}. Using random weights.")
    # Fallback to random weights if anything goes wrong
    model = ConceivoNet(input_size=3)
    
    # Add basic synthetic data in case of critical errors
    synthetic_data = [
        [0.5, 0.5, 0.5, 0.5, 0.75],  # Balanced baseline
        [0.8, 0.3, 0.7, 0.9, 0.6],   # High stress, low strength
        [0.4, 0.9, 0.3, 0.2, 0.85],  # High material strength, low build time
    ]

async def crawl_engineering_knowledge():
    knowledge_parts = []
    if app.state.llm_config and app.state.extraction_strategy:
        for url in SEED_URLS:
            async for result in app.state.crawler.arun(
                url=url,
                word_count_threshold=100,
                extraction_strategy=app.state.extraction_strategy
            ):
                if result.extracted_content:
                    if "kaggle.com" in result.url:
                        knowledge_parts.append(f"[Kaggle Resource: {result.title}]\n{result.extracted_content}\nURL: {result.url}")
                    else:
                        knowledge_parts.append(result.extracted_content)
    else:
        print("Crawling skipped due to missing API key or configuration.")
    return "\n".join(knowledge_parts)

def start_background_learning(app: FastAPI):
    @app.on_event("startup")
    def _():
        async def auto_learn_loop():
            while True:
                try:
                    print("[Auto-Learning] Running periodic knowledge update...")
                    new_knowledge = await crawl_engineering_knowledge()
                    if new_knowledge:
                        app.state.initial_knowledge = new_knowledge
                        if CONCEIVO_AI_AVAILABLE and app.state.conceivo_ai:
                            app.state.conceivo_ai.learn(
                                f"engineering-knowledge-{datetime.datetime.now().timestamp()}",
                                new_knowledge.split('\n')[0][:200] + "..."
                            )
                    print(f"[Auto-Learning] Updated knowledge at {datetime.datetime.now()}")
                except Exception as e:
                    print(f"[Auto-Learning] Error during periodic update: {e}")
                await asyncio.sleep(3600)
        asyncio.create_task(auto_learn_loop())

@app.post("/api/conceivo")
async def get_ai_response(ai_request: AIRequest, request: Request):
    user_message = ai_request.message.lower()
    response = "I’m learning as we go! Ask me about 3D modeling or component efficiency."
    command = None
    predicted_efficiency = None
    new_knowledge = None
    edit_result = None

    if "learn more" in user_message:
        new_knowledge = await crawl_engineering_knowledge()
        response = "I’ve gathered some fresh engineering insights! Check the new knowledge."
        if CONCEIVO_AI_AVAILABLE and app.state.conceivo_ai:
            app.state.conceivo_ai.learn(
                f"engineering-knowledge-{datetime.datetime.now().timestamp()}",
                new_knowledge.split('\n')[0][:200] + "..."
            )

    elif any(phrase in response.lower() for phrase in ["not sure", "no api key", "disabled"]):
        print("[Auto-Learning] Triggering learning due to uncertain response")
        new_knowledge = await crawl_engineering_knowledge()
        if new_knowledge:
            app.state.initial_knowledge = new_knowledge
            response += "\nI've updated my knowledge base to better help you."
            if CONCEIVO_AI_AVAILABLE and app.state.conceivo_ai:
                app.state.conceivo_ai.learn(
                    f"engineering-knowledge-{datetime.datetime.now().timestamp()}",
                    new_knowledge.split('\n')[0][:200] + "..."
                )

    if "3d model" in user_message or "cad" in user_message:
        response = "I can help with 3D modeling! What specific component or design are you interested in?"
        command = {"action": "model", "params": {"type": "3D"}}

    if "efficiency" in user_message and ai_request.metrics:
        if not model_ready:
            response = "Efficiency predictions are disabled due to missing trained model (conceivo_model.pth)."
        elif len(ai_request.metrics) < 3:
            response = "Error: Please provide exactly 3 metrics (cost, material strength, time)."
        else:
            with torch.no_grad():
                # Use only the first 3 metrics matching the training data
                input_tensor = torch.tensor([ai_request.metrics[:3]], dtype=torch.float32)
                prediction = model(input_tensor)
                predicted_efficiency = prediction.item()
                response = f"Predicted efficiency: {predicted_efficiency:.2f}"

    if any(action in user_message for action in ["add sphere", "add box", "add cylinder", "add cone", "add torus", "add plane", "add wedge"]):
        action = user_message.split("add ")[1].split()[0]
        response = f"Added a {action} to the scene."
        command = {"action": "add", "params": {"type": action}}

    if ai_request.selected_part and any(action in user_message for action in ["delete", "fix", "fill", "rotate", "scale", "add part", "extrude", "revolve", "mirror", "union", "subtract", "intersect"]):
        part_name = ai_request.selected_part.lower()
        if "delete" in user_message:
            edit_result = f"Deleted {part_name} from the model."
            command = {"action": "delete", "params": {"part": part_name}}
        elif "fix" in user_message:
            edit_result = f"Fixed {part_name} (e.g., adjusted color or structure)."
            command = {"action": "fix", "params": {"part": part_name}}
        elif "fill" in user_message:
            edit_result = f"Filled {part_name} with default material."
            command = {"action": "fill", "params": {"part": part_name}}
        elif "rotate" in user_message:
            edit_result = f"Rotated {part_name} by 90 degrees."
            command = {"action": "rotate", "params": {"part": part_name, "angle": 90}}
        elif "scale" in user_message:
            edit_result = f"Scaled {part_name} by 1.5x."
            command = {"action": "scale", "params": {"part": part_name, "factor": 1.5}}
        elif "add part" in user_message:
            part_type = user_message.split("add part")[1].strip().lower()
            edit_result = f"Added {part_type} to the model."
            command = {"action": "add", "params": {"type": part_type}}
        elif any(action in user_message for action in ["extrude", "revolve", "mirror", "union", "subtract", "intersect"]):
            action = next(a for a in ["extrude", "revolve", "mirror", "union", "subtract", "intersect"] if a in user_message)
            edit_result = f"Applied {action} to {part_name}."
            command = {"action": action, "params": {"part": part_name}}
        else:
            edit_result = f"Unknown part or action for {part_name}."

    full_knowledge = request.app.state.initial_knowledge + (new_knowledge or "")
    if full_knowledge and any(keyword in user_message for keyword in ["design", "material", "stress"]):
        response += f"\nBased on what I’ve learned: {full_knowledge.split('\n')[0][:200]}..."

    return AIResponse(
        message=response,
        predicted_efficiency=predicted_efficiency,
        command=command,
        new_knowledge=new_knowledge,
        edit_result=edit_result
    )

@app.post("/api/conceivo/component")
async def add_component(component: Dict, request: Request):
    if CONCEIVO_AI_AVAILABLE and app.state.conceivo_ai:
        app.state.conceivo_ai.add_component(EngineeringComponent(
            id=component.get("id", str(datetime.datetime.now().timestamp())),
            name=component.get("name", "Unnamed"),
            type=component.get("type", "unknown"),
            description=component.get("description", ""),
            model_path=component.get("model_path"),
            metrics=component.get("metrics", {})
        ))
        return {"status": "added"}
    raise HTTPException(status_code=500, detail="ConceivoAI not available")

@app.get("/api/conceivo/components")
async def get_components(request: Request):
    if CONCEIVO_AI_AVAILABLE and app.state.conceivo_ai:
        return {"components": [vars(comp) for comp in app.state.conceivo_ai.get_components()]}
    return {"components": []}

@app.post("/api/simulate")
async def simulate(request: Dict):
    object_id = request.get("objectId")
    if not object_id:
        raise HTTPException(status_code=400, detail="Object ID required")
    try:
        mesh = pv.read(f"models/{object_id}.stl")
        stress = mesh.compute_stress().max() / 1e6
        return {"stress": stress}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")

@app.post("/api/modeling")
async def modeling(request: ModelingRequest):
    operation = request.operation
    object_ids = request.objectIds
    params = request.params or {}
    try:
        meshes = [trimesh.load(f"models/{id}.stl") for id in object_ids if os.path.exists(f"models/{id}.stl")]
        if not meshes:
            raise HTTPException(status_code=404, detail="No valid models found")
        
        result = None
        if operation == "union":
            result = util.concatenate(meshes)
        elif operation == "subtract" and len(meshes) >= 2:
            result = boolean.difference([meshes[0], meshes[1]])
        elif operation == "intersect" and len(meshes) >= 2:
            result = boolean.intersection([meshes[0], meshes[1]])
        elif operation == "extrude":
            distance = params.get("distance", 1.0)
            result = meshes[0]  # Placeholder
        elif operation == "revolve":
            axis = params.get("axis", "z")
            angle = params.get("angle", 360)
            result = meshes[0]  # Placeholder
        elif operation == "mirror":
            axis = params.get("axis", "x")
            result = meshes[0]  # Placeholder
        elif operation == "bevel-edges":
            distance = params.get("distance", 0.1)
            result = meshes[0]  # Placeholder
        elif operation == "loop-cut-and-slide":
            cut_position = params.get("cut_position", 0.5)
            slide_distance = params.get("slide_distance", 0.1)
            result = meshes[0]  # Placeholder
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported operation: {operation}")

        if result:
            result_path = f"models/{operation}_{datetime.datetime.now().timestamp()}.stl"
            result.export(result_path)
            new_object = {
                "id": f"{operation}_{datetime.datetime.now().timestamp()}",
                "type": "stl",
                "modelPath": result_path,
                "position": [0, 0.5, 0],
                "rotation": [0, 0, 0],
                "scale": [1, 1, 1],
                "color": f"hsl({random.randint(0, 360)}, 70%, 60%)",
                "createdAt": datetime.datetime.now().timestamp(),
                "updatedAt": datetime.datetime.now().timestamp()
            }
            if CONCEIVO_AI_AVAILABLE and app.state.conceivo_ai:
                app.state.conceivo_ai.add_component(EngineeringComponent(
                    id=new_object["id"],
                    name=f"{operation}_result",
                    type="modeled",
                    model_path=result_path
                ))
            return {"objects": [new_object]}
        return {"objects": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Modeling operation failed: {str(e)}")

@app.websocket("/api/collaborate")
async def collaborate_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            objects = json.loads(data)
            for client in app.state.clients:
                if client != websocket:
                    await client.send_text(data)
    except WebSocketDisconnect:
        app.state.clients.remove(websocket)

@app.post("/api/crawl")
async def ai_crawl(request: Request, url: Union[str, None] = None):
    urls = [url] if url else SEED_URLS
    if not urls:
        raise HTTPException(status_code=400, detail="No URLs provided for crawling")
    if not request.app.state.xai_api_key:
        raise HTTPException(status_code=401, detail="XAI_API_KEY is not configured")
    results = []
    async with httpx.AsyncClient() as client:
        for target_url in urls:
            try:
                response = await client.post(
                    "https://api.x.ai/v1/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {request.app.state.xai_api_key.get_secret_value()}"
                    },
                    json={
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are an AI assistant that extracts engineering knowledge from web pages."
                            },
                            {
                                "role": "user",
                                "content": f"Extract key engineering concepts and data from this URL: {target_url}"
                            }
                        ],
                        "model": "grok-3-latest",
                        "stream": False,
                        "temperature": 0
                    },
                    timeout=30.0
                )
                if response.status_code == 200:
                    data = response.json()
                    content = data["choices"][0]["message"]["content"]
                    results.append({
                        "url": target_url,
                        "extracted_content": content
                    })
                else:
                    results.append({
                        "url": target_url,
                        "error": f"HTTP {response.status_code}: {response.text}"
                    })
            except Exception as e:
                results.append({
                    "url": target_url,
                    "error": str(e)
                })
    return {"results": results}

@app.post("/api/enhance-prompt")
async def enhance_prompt(prompt: str):
    grok_ai = GrokAI()
    enhanced_prompt = await grok_ai.enhance_prompt(prompt)
    return {"enhanced_prompt": enhanced_prompt}

async def test_crawler():
    try:
        if app.state.llm_config and app.state.extraction_strategy:
            async for result in app.state.crawler.arun(
                url=SEED_URLS[0],
                word_count_threshold=100,
                extraction_strategy=app.state.extraction_strategy
            ):
                if result.extracted_content:
                    print(f"Test crawl successful for {result.url}: {result.extracted_content[:100]}...")
                else:
                    print("Test crawl failed or no content extracted.")
        else:
            print("Test crawl skipped due to missing API key or configuration.")
    except Exception as e:
        print(f"Test crawler error: {e}")

async def fetch_tripo_3d(prompt: str):
    """Send a prompt to Tripo API and return generated 3D model data."""
    if not TRIPO_API_KEY:
        raise HTTPException(status_code=401, detail="TRIPO_API_KEY not configured")
    
    headers = {
        "Authorization": f"Bearer {TRIPO_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "prompt": prompt,
        "format": "glb",
        "quality": "high"
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "https://api.tripo.ai/v1/engine/generate",  # Using correct Tripo API endpoint
                headers=headers,
                json=payload,
                timeout=30.0
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    model_id = data["result"]["id"]
                    download_url = data["result"]["preview_model_url"]
                    
                    # Download the model file
                    model_response = await client.get(download_url)
                    
                    # Save the model file locally
                    model_path = f"models/{model_id}.glb"
                    with open(model_path, "wb") as f:
                        f.write(model_response.content)
                    
                    return {
                        "model_id": model_id,
                        "model_path": model_path,
                        "download_url": download_url
                    }
                else:
                    error_msg = data.get("error", "Unknown error from Tripo API")
                    raise HTTPException(status_code=500, detail=f"Tripo API error: {error_msg}")
            else:
                raise HTTPException(status_code=response.status_code, detail=f"HTTP {response.status_code}: {response.text}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Tripo API request failed: {str(e)}")

@app.post("/api/tripo-generate")
async def generate_3d_model(prompt: str):
    """Endpoint to generate a 3D model from a text prompt using Tripo API."""
    model_data = await fetch_tripo_3d(prompt)
    
    # Create a new component for the generated model
    component = EngineeringComponent(
        id=model_data["model_id"],
        name=f"Generated Model: {prompt[:30]}...",
        type="generated",
        description=f"3D model generated from prompt: {prompt}",
        model_path=model_data["model_path"]
    )
    
    if CONCEIVO_AI_AVAILABLE and app.state.conceivo_ai:
        app.state.conceivo_ai.add_component(component)
    
    return {
        "status": "success",
        "model_id": model_data["model_id"],
        "model_path": model_data["model_path"],
        "download_url": model_data["download_url"]
    }

def train_conceivo_net():
    global model_ready  # Moved to top
    try:
        df = pd.read_csv(r"d:\Conceivin3D\architectural_data.csv")
        
        # Normalize using only the original 3 features
        df["cost"] = df["cost_million_usd"] / df["cost_million_usd"].max()
        df["material_strength"] = df["material_strength_mpa"] / df["material_strength_mpa"].max()
        df["build_time"] = df["time_months"] / df["time_months"].max()
        
        # We don't use stress from the CSV file anymore
        targets = torch.tensor(df[["efficiency"]].values, dtype=torch.float32)
        
        if CONCEIVO_AI_AVAILABLE:
            conceivo_ai = ConceivoAI()
            for comp in conceivo_ai.components:
                weight = comp.performanceMetrics.get("weight", 200)
                cost = min(1.0, max(0.0, weight / 300))
                strength = 0.9 if len(comp.failureModes) <= 1 else 0.7
                build_time = 0.7 if comp.type == "propulsion" else 0.4
                efficiency = comp.performanceMetrics.get("efficiency", 0.85)
                data.append([cost, strength, build_time, efficiency])

        # Add Tripo-generated models to training data
        tripogenerated_models = [
            # Example features from generated rocket components
            [0.6, 0.8, 0.7, 0.95],  # Rocket Nozzle
            [0.7, 0.7, 0.8, 0.80],  # Combustion Chamber
            [0.7, 0.7, 0.8, 0.85],  # Rocket Engine
            # Example features from generated mechanical parts
            [0.5, 0.9, 0.3, 0.90],  # Turbine Blade
            [0.4, 0.8, 0.4, 0.85],  # Gear Assembly
            [0.8, 0.5, 0.6, 0.70],  # Hydraulic Cylinder
        ]
        
        # Add synthetic data based on engineering principles
        synthetic_data = [
            # Varying parameters to teach relationships
            [0.9, 0.9, 0.9, 0.9, 0.6],  # High cost, high strength, long build time, high stress -> lower efficiency
            [0.3, 0.3, 0.3, 0.3, 0.9],  # Low cost, low strength, short build time, low stress -> higher efficiency
            [0.5, 0.9, 0.4, 0.3, 0.92],  # Balanced with high material strength
            [0.7, 0.6, 0.8, 0.5, 0.75],  # High build time impacts efficiency
        ]

        # Combine all data sources
        all_data = data + tripogenerated_models + synthetic_data
        
        # Create DataFrame with only the core 3 features
        df = pd.DataFrame(all_data, columns=["cost", "material_strength", "build_time", "efficiency"])
        
        # Prepare inputs using only the 3 core features
        inputs = torch.tensor(df[["cost", "material_strength", "build_time"]].values, dtype=torch.float32)
        targets = torch.tensor(df[["efficiency"]].values, dtype=torch.float32)

        trained_model = ConceivoNet(input_size=3)  # Explicitly set input_size to 3
        optimizer = torch.optim.Adam(trained_model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        for epoch in range(1000):
            optimizer.zero_grad()
            outputs = trained_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        torch.save(trained_model.state_dict(), "conceivo_model.pth")
        print("Trained model saved to conceivo_model.pth")
        trained_model.load_state_dict(torch.load("conceivo_model.pth"))
        trained_model.eval()
        model_ready = True
    except FileNotFoundError:
        print("architectural_data.csv not found. Using synthetic data.")
        data = [
            [0.6, 0.8, 0.7, 0.95],  # Rocket Nozzle
            [0.7, 0.7, 0.8, 0.80],  # Combustion Chamber
            [0.7, 0.7, 0.8, 0.85],  # Rocket Engine
            [0.5, 0.9, 0.3, 0.90],  # Solar Panel
            [0.4, 0.8, 0.4, 0.85],  # Rover Wheel
            [0.8, 0.5, 0.6, 0.70],  # Antenna
        ]

        # Add Tripo-generated models and synthetic data when using fallback
        fallback_data = data + [
            [0.5, 0.9, 0.3, 0.90],  # Additional turbine blade
            [0.6, 0.6, 0.7, 0.83],  # Jet engine component
            [0.9, 0.4, 0.8, 0.65],  # High-stress part
            # Synthetic variations
            [0.9, 0.9, 0.9, 0.6],
            [0.3, 0.3, 0.3, 0.9],
            [0.5, 0.9, 0.4, 0.92],
        ]
        
        df = pd.DataFrame(fallback_data, columns=["cost", "material_strength", "build_time", "efficiency"])
        
        # Prepare inputs using only the 3 core features
        inputs = torch.tensor(df[["cost", "material_strength", "build_time"]].values, dtype=torch.float32)
        targets = torch.tensor(df[["efficiency"]].values, dtype=torch.float32)

        trained_model = ConceivoNet(input_size=3)  # Explicitly set input_size to 3
        optimizer = torch.optim.Adam(trained_model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        for epoch in range(1000):
            optimizer.zero_grad()
            outputs = trained_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        torch.save(trained_model.state_dict(), "conceivo_model.pth")
        print("Trained model saved to conceivo_model.pth")
        trained_model.load_state_dict(torch.load("conceivo_model.pth"))
        trained_model.eval()
        model_ready = True

if __name__ == "__main__":
    import asyncio
    import uvicorn
    import random
    import json
    train_conceivo_net()
    asyncio.run(test_crawler())
    uvicorn.run(app, host="0.0.0.0", port=8000)