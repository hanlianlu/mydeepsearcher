import os
from typing import Dict, List

from deepsearcher.llm.base import BaseLLM, ChatResponse


class Anthropic(BaseLLM):
    def __init__(self, model: str = "claude-3-5-sonnet-latest", max_tokens: int = 8192, **kwargs):
        import anthropic

        self.model = model
        self.max_tokens = max_tokens
        if "api_key" in kwargs:
            api_key = kwargs.pop("api_key")
        else:
            api_key = os.getenv("ANTHROPIC_API_KEY")
        if "base_url" in kwargs:
            base_url = kwargs.pop("base_url")
        else:
            base_url = None
        self.client = anthropic.Anthropic(api_key=api_key, base_url=base_url, **kwargs)

    def chat(self, messages: List[Dict]) -> ChatResponse:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=messages,
        )
        return ChatResponse(
            content=message.content[0].text,
            total_tokens=message.usage.input_tokens + message.usage.output_tokens,
        )
