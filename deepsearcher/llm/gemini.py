import os
from typing import Dict, List

from deepsearcher.llm.base import BaseLLM, ChatResponse


class Gemini(BaseLLM):
    """https://ai.google.dev/gemini-api/docs/sdks"""

    def __init__(self, model: str = "gemini-2.0-flash", **kwargs):
        from google import genai

        self.model = model
        if "api_key" in kwargs:
            api_key = kwargs.pop("api_key")
        else:
            api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key, **kwargs)

    def chat(self, messages: List[Dict]) -> ChatResponse:
        response = self.client.models.generate_content(
            model=self.model,
            contents="\n".join([m["content"] for m in messages]),
        )
        return ChatResponse(
            content=response.text,
            total_tokens=response.usage_metadata.total_token_count,
        )
