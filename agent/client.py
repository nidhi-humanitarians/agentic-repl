from typing import List, Dict, Any, Optional
import openai
from agentic_search_pi.config.settings import (
    PARTICLE_BASE_URL,
    PARTICLE_API_KEY,
    DEFAULT_MODEL
)

class ParticleAIClient:
    def __init__(self, api_key: str = PARTICLE_API_KEY, base_url: str = PARTICLE_BASE_URL):
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: str = DEFAULT_MODEL,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = "auto",
        temperature: float = 0.0,
        max_tokens: int = 2048
    ) -> Any:
        """
        Calls Particle AI chat completions API.
        Supports standard messages and optional function calling tools.
        """
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        return self.client.chat.completions.create(**kwargs)

if __name__ == "__main__":
    client = ParticleAIClient()
    response = client.chat_completion(
        messages=[{"role": "user", "content": "Ping test: respond with PONG"}],
        max_tokens=64
    )
    print("Response:", response.choices[0].message.content)
