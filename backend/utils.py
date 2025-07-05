import os
import httpx
from typing import Optional
import logging
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class GrokAI:
    def __init__(self):
        self.api_key = os.getenv("XAI_API_KEY")
        self.base_url = "https://api.x.ai/v1/chat/completions"
        self.default_model = os.getenv("GROK_MODEL", "grok-3-latest")
        self.default_temperature = float(os.getenv("GROK_TEMPERATURE", 0.7))
        self.default_max_tokens = int(os.getenv("GROK_MAX_TOKENS", 500))
        self.timeout = float(os.getenv("GROK_TIMEOUT", 30.0))
        
        if not self.api_key:
            logger.error("XAI_API_KEY environment variable is not set")
            raise ValueError("XAI_API_KEY environment variable is not set")

    @lru_cache(maxsize=100)
    async def enhance_prompt(self, prompt: str, model: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
        """
        Enhance a prompt using the xAI API with caching.
        
        Args:
            prompt (str): The prompt to enhance.
            model (str, optional): The model to use (defaults to GROK_MODEL or grok-3-latest).
            temperature (float, optional): Sampling temperature (defaults to GROK_TEMPERATURE or 0.7).
            max_tokens (int, optional): Maximum tokens for response (defaults to GROK_MAX_TOKENS or 500).
        
        Returns:
            str: Enhanced prompt content.
        
        Raises:
            ValueError: If prompt is empty or too long.
            httpx.HTTPStatusError: If the API returns a non-200 status.
            httpx.RequestError: If the network request fails.
        """
        if not prompt or not prompt.strip():
            logger.error("Prompt is empty or whitespace")
            raise ValueError("Prompt cannot be empty")
        if len(prompt) > 1000:
            logger.error("Prompt exceeds 1000 characters")
            raise ValueError("Prompt must be 1000 characters or less")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "model": model or self.default_model,
            "temperature": temperature if temperature is not None else self.default_temperature,
            "max_tokens": max_tokens or self.default_max_tokens
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                logger.info(f"Sending prompt enhancement request for: {prompt[:50]}...")
                response = await client.post(self.base_url, json=payload, headers=headers)
                response.raise_for_status()  # Raises HTTPStatusError for non-2xx responses
                
                data = response.json()
                enhanced_content = data["choices"][0]["message"]["content"]
                logger.info(f"Successfully enhanced prompt: {enhanced_content[:50]}...")
                return enhanced_content
                
        except httpx.HTTPStatusError as e:
            logger.error(f"Grok API HTTP error: {e.response.status_code} - {e.response.text}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Grok API network error: {str(e)}")
            raise
        except (KeyError, ValueError) as e:
            logger.error(f"Invalid response format from Grok API: {str(e)}")
            raise ValueError(f"Invalid response from Grok API: {str(e)}")