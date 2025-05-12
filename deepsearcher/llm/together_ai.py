import os
from typing import Dict, List

from deepsearcher.llm.base import BaseLLM, ChatResponse


class TogetherAI(BaseLLM):
    """https://www.together.ai/"""

    def __init__(self, model: str = "deepseek-ai/DeepSeek-V3", **kwargs):
        from together import Together

        self.model = model
        if "api_key" in kwargs:
            api_key = kwargs.pop("api_key")
        else:
            api_key = os.getenv("TOGETHER_API_KEY")
        self.client = Together(api_key=api_key, **kwargs)

    def chat(self, messages: List[Dict]) -> ChatResponse:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return ChatResponse(
            content=response.choices[0].message.content,
            total_tokens=response.usage.total_tokens,
        )
