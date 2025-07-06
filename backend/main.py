from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, SecretStr
from contextlib import asynccontextmanager, contextmanager
import asyncio
import torch
import torch.nn as nn
from crawl4ai.async_webcrawler import AsyncWebCrawler
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from crawl4ai.types import LLMConfig
import os
from backend.crawl_config import SEED_URLS
from langchain_mistralai.chat_models import ChatMistralAI
import datetime
import sys
from typing import TypedDict, Union, Optional
from crawl4ai.async_configs import LLMConfig as CrawlLLMConfig
import httpx
from backend.utils import GrokAI
import pandas as pd
import numpy as np

# Add this function for background auto-learning
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
                await asyncio.sleep(3600)  # Run every hour

        asyncio.create_task(auto_learn_loop())


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.crawler = AsyncWebCrawler()
        configure_app(app)
        app.state.initial_knowledge = await crawl_engineering_knowledge() if app.state.llm_config else "No API key, crawling disabled."
        # Define availability flag after potential class initialization
        global CONCEIVO_AI_AVAILABLE
        CONCEIVO_AI_AVAILABLE = False
        if 'ConceivoAI' in globals():
            CONCEIVO_AI_AVAILABLE = True
            app.state.conceivo_ai = ConceivoAI()  # Initialize with default components
        
        # Start background learning
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

class ConceivoNet(nn.Module):
    def __init__(self, input_size: int = 3, hidden_size: int = 10, output_size: int = 1):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.sigmoid(out)
        return out

class ConceivoAI:
    def __init__(self):
        self.components = []  # Placeholder for components
        self.type = "default"

    def learn(self, topic: str, content: str):
        print(f"Learning about {topic}: {content}")

model_ready = False
try:
    model = ConceivoNet()
    model.load_state_dict(torch.load("conceivo_model.pth"))
    model.eval()
    model_ready = True
except FileNotFoundError:
    print("Model file not found. Using randomly initialized weights. Predictions disabled.")
    model = ConceivoNet()
except RuntimeError as e:
    print(f"Error loading model state: {e}. Using randomly initialized weights. Predictions disabled.")
    model = ConceivoNet()

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

@app.post("/api/conceivo")
async def get_ai_response(ai_request: AIRequest, request: Request):
    user_message = ai_request.message.lower()
    response = "I’m learning as we go! Ask me about 3D modeling or component efficiency."
    command = None
    predicted_efficiency = None
    new_knowledge = None
    edit_result = None

    # Detect if this is a learning request
    if "learn more" in user_message:
        new_knowledge = await crawl_engineering_knowledge()
        response = "I’ve gathered some fresh engineering insights! Check the new knowledge."
        if CONCEIVO_AI_AVAILABLE and app.state.conceivo_ai:
            app.state.conceivo_ai.learn(
                f"engineering-knowledge-{datetime.datetime.now().timestamp()}",
                new_knowledge.split('\n')[0][:200] + "..."
            )

    # Auto-learn on uncertain responses
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
        elif len(ai_request.metrics) != 3:
            response = "Error: Please provide exactly 3 metrics (cost, material strength, time)."
        else:
            with torch.no_grad():
                input_tensor = torch.tensor([ai_request.metrics], dtype=torch.float32)
                prediction = model(input_tensor)
                predicted_efficiency = prediction.item()
                response = f"Predicted efficiency: {predicted_efficiency:.2f}"

    if "add sphere" in user_message:
        response = "Added a sphere to the scene."
        command = {"action": "add", "params": {"type": "sphere"}}

    if ai_request.selected_part and any(action in user_message for action in ["delete", "fix", "fill", "rotate", "scale", "add part"]):
        part_name = ai_request.selected_part.lower()
        if "delete" in user_message and part_name in ["body", "windshield", "door", "handle"]:
            edit_result = f"Deleted {part_name} from the model."
            command = {"action": "delete", "params": {"part": part_name}}
        elif "fix" in user_message and part_name in ["body", "windshield", "door", "handle"]:
            edit_result = f"Fixed {part_name} (e.g., adjusted color or structure)."
            command = {"action": "fix", "params": {"part": part_name}}
        elif "fill" in user_message and part_name in ["body", "windshield", "door", "handle"]:
            edit_result = f"Filled {part_name} with default material."
            command = {"action": "fill", "params": {"part": part_name}}
        elif "rotate" in user_message and part_name in ["body", "windshield", "door", "handle"]:
            edit_result = f"Rotated {part_name} by 90 degrees."
            command = {"action": "rotate", "params": {"part": part_name, "angle": 90}}
        elif "scale" in user_message and part_name in ["body", "windshield", "door", "handle"]:
            edit_result = f"Scaled {part_name} by 1.5x."
            command = {"action": "scale", "params": {"part": part_name, "factor": 1.5}}
        elif "add part" in user_message:
            part_type = user_message.split("add part")[1].strip().lower()
            edit_result = f"Added {part_type} to the model."
            command = {"action": "add", "params": {"type": part_type}}
        else:
            edit_result = f"Unknown part or action for {part_name}. Valid parts: body, windshield, door, handle."

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

def train_conceivo_net():
    """Train ConceivoNet using architectural_data.csv and optionally ConceivoAI components."""
    try:
        # Load and normalize architectural_data.csv
        df = pd.read_csv(r"d:\Conceivin3D\architectural_data.csv")
        df["cost"] = df["cost_million_usd"] / df["cost_million_usd"].max()  # Normalize to 0-1
        df["material_strength"] = df["material_strength_mpa"] / df["material_strength_mpa"].max()
        df["build_time"] = df["time_months"] / df["time_months"].max()

        # Initialize data with CSV
        data = df[["cost", "material_strength", "build_time", "efficiency"]].values.tolist()

        # Optionally add ConceivoAI components
        if CONCEIVO_AI_AVAILABLE:
            conceivo_ai = ConceivoAI()
            for comp in conceivo_ai.components:
                weight = comp.performanceMetrics.get("weight", 200)
                cost = min(1.0, max(0.0, weight / 300))  # Normalize weight (150-200 kg -> 0.5-0.7)
                strength = 0.9 if len(comp.failureModes) <= 1 else 0.7
                build_time = 0.7 if comp.type == "propulsion" else 0.4
                efficiency = comp.performanceMetrics.get("efficiency", 0.85)
                data.append([cost, strength, build_time, efficiency])

        df = pd.DataFrame(data, columns=["cost", "material_strength", "build_time", "efficiency"])
        inputs = torch.tensor(df[["cost", "material_strength", "build_time"]].values, dtype=torch.float32)
        targets = torch.tensor(df[["efficiency"]].values, dtype=torch.float32)

        model = ConceivoNet()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        # Train for 1000 epochs
        for epoch in range(1000):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

            # Save trained weights
            torch.save(model.state_dict(), "conceivo_model.pth")
            print("Trained model saved to conceivo_model.pth")
            model.load_state_dict(torch.load("conceivo_model.pth"))
            model.eval()
            global model_ready
            model_ready = True
    except FileNotFoundError:
        print("architectural_data.csv not found. Falling back to ConceivoAI components and synthetic data.")
        data = [
            [0.6, 0.8, 0.7, 0.95],  # Rocket Nozzle
            [0.7, 0.7, 0.8, 0.80],  # Combustion Chamber
            [0.7, 0.7, 0.8, 0.85],  # Rocket Engine
            [0.5, 0.9, 0.3, 0.90],  # Solar Panel
            [0.4, 0.8, 0.4, 0.85],  # Rover Wheel
            [0.8, 0.5, 0.6, 0.70],  # Antenna
        ]
        if CONCEIVO_AI_AVAILABLE:
            conceivo_ai = ConceivoAI()
            for comp in conceivo_ai.components:
                weight = comp.performanceMetrics.get("weight", 200)
                cost = min(1.0, max(0.0, weight / 300))
                strength = 0.9 if len(comp.failureModes) <= 1 else 0.7
                build_time = 0.7 if comp.type == "propulsion" else 0.4
                efficiency = comp.performanceMetrics.get("efficiency", 0.85)
                data.append([cost, strength, build_time, efficiency])

            df = pd.DataFrame(data, columns=["cost", "material_strength", "build_time", "efficiency"])
            inputs = torch.tensor(df[["cost", "material_strength", "build_time"]].values, dtype=torch.float32)
            targets = torch.tensor(df[["efficiency"]].values, dtype=torch.float32)

            # Reassign the global model variable by returning and handling it outside if needed
            trained_model = ConceivoNet()
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
            return trained_model
if __name__ == "__main__":
    import asyncio
    import uvicorn
    train_conceivo_net()
    asyncio.run(test_crawler())
    uvicorn.run(app, host="0.0.0.0", port=8000)