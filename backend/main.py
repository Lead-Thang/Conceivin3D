from fastapi import FastAPI, Request
from pydantic import BaseModel
import torch
import torch.nn as nn
from contextlib import asynccontextmanager
from crawl4ai.async_webcrawler import AsyncWebCrawler
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from crawl4ai.types import LLMConfig
import os
from backend.crawl_config import SEED_URLS
from langchain_mistralai.chat_models import ChatMistralAI

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.crawler = AsyncWebCrawler()
        # Remove warmup() as it's not a valid method
        app.state.initial_knowledge = await crawl_engineering_knowledge()
        # Initialize Mistral AI client using ChatMistralAI class directly
        api_key = os.getenv("MISTRAL_API_KEY")
        if api_key:
            app.state.mistral_client = ChatMistralAI(
                model="mistral-large-latest",  # Specify a valid model name
                temperature=0.7,  # Add temperature parameter
                max_tokens=500,   # Add max_tokens parameter
                api_key=api_key   # Using string directly as ChatMistralAI expects a string
            )
        else:
            print("MISTRAL_API_KEY environment variable is not set")
    except Exception as e:
        print(f"Error during lifespan setup: {e}")
        app.state.initial_knowledge = "Initial knowledge fetch failed. Try 'learn more' to update."
    yield
    await app.state.crawler.close()

app = FastAPI(lifespan=lifespan)

class AIRequest(BaseModel):
    message: str
    metrics: list[float] = []

class AIResponse(BaseModel):
    message: str
    predicted_efficiency: float | None = None
    command: dict | None = None
    new_knowledge: str | None = None

class ConceivoNet(nn.Module):
    def __init__(self, input_size: int = 3, hidden_size: int = 10, output_size: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model_ready = False
try:
    model = ConceivoNet()
    model.load_state_dict(torch.load("conceivo_model.pth"))
    model.eval()
    model_ready = True
except FileNotFoundError:
    print("Model file not found. Using randomly initialized weights. Predictions disabled.")
    model = ConceivoNet()

async def crawl_engineering_knowledge():
    knowledge_parts = []
    for url in SEED_URLS:
        async for result in app.state.crawler.arun(
            url=url,
            word_count_threshold=100,
            extraction_strategy=LLMExtractionStrategy(
                llm_config=LLMConfig(
                    provider="mistral",
                    model="mistral-large-latest",
                    temperature=0.7,
                    max_tokens=500,
                    api_key=os.getenv("MISTRAL_API_KEY")
                ),
                extraction_type="text",
                instruction="Extract extremely high-level engineering concepts, advanced formulas, and design principles."
            )
        ):
            if result.extracted_content:
                knowledge_parts.append(result.extracted_content)
    return "\n".join(knowledge_parts)

@app.post("/api/conceivo")
async def get_ai_response(ai_request: AIRequest, request: Request):
    user_message = ai_request.message.lower()
    response = "I’m learning as we go! Ask me about 3D modeling or component efficiency."
    command = None
    predicted_efficiency = None
    new_knowledge = None

    if "efficiency" in user_message and ai_request.metrics and model_ready:
        with torch.no_grad():
            input_tensor = torch.tensor([ai_request.metrics], dtype=torch.float32)
            prediction = model(input_tensor)
            predicted_efficiency = prediction.item()
            response = f"Predicted efficiency: {predicted_efficiency:.2f}"

    if "add sphere" in user_message:
        response = "Added a sphere to the scene."
        command = {"action": "add", "params": {"type": "sphere"}}

    if "learn more" in user_message:
        new_knowledge = await crawl_engineering_knowledge()
        response = "I’ve gathered some fresh engineering insights! Check the new knowledge."

    full_knowledge = request.app.state.initial_knowledge + (new_knowledge or "")
    if full_knowledge and any(keyword in user_message for keyword in ["design", "material", "stress"]):
        response += f"\nBased on what I’ve learned: {full_knowledge.split('\n')[0][:200]}..."

    return AIResponse(message=response, predicted_efficiency=predicted_efficiency, command=command, new_knowledge=new_knowledge)

async def test_crawler():
    try:
        async for result in app.state.crawler.arun(
            url=SEED_URLS[0],
            word_count_threshold=100,
            extraction_strategy=LLMExtractionStrategy(
                llm_config=LLMConfig(
                    provider="mistral/mixtral-8x7b",
                    api_key=os.getenv("MISTRAL_API_KEY")
                ),
                extraction_type="text",
                instruction="Extract extremely high-level engineering concepts."
            )
        ):
            if result.extracted_content:
                print(f"Test crawl successful for {result.url}: {result.extracted_content[:100]}...")
    except Exception as e:
        print(f"Test crawler error: {e}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_crawler())
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)