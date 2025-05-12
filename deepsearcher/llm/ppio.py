import os
from typing import Dict, List

from deepsearcher.llm.base import BaseLLM, ChatResponse


class PPIO(BaseLLM):
    """
    PPIO API
    """

    def __init__(self, model: str = "deepseek/deepseek-v3/community", **kwargs):
        from openai import OpenAI as OpenAI_

        self.model = model
        if "api_key" in kwargs:
            api_key = kwargs.pop("api_key")
        else:
            api_key = os.getenv("PPIO_API_KEY")
        if "base_url" in kwargs:
            base_url = kwargs.pop("base_url")
        else:
            base_url = "https://api.ppinfra.com/v3/openai"
        self.client = OpenAI_(api_key=api_key, base_url=base_url, **kwargs)

    def chat(self, messages: List[Dict]) -> ChatResponse:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return ChatResponse(
            content=completion.choices[0].message.content,
            total_tokens=completion.usage.total_tokens,
        )
