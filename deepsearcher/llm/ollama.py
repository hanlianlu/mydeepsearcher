from typing import Dict, List

from deepsearcher.llm.base import BaseLLM, ChatResponse


class Ollama(BaseLLM):
    def __init__(self, model: str = "qwq", **kwargs):
        from ollama import Client

        self.model = model
        if "base_url" in kwargs:
            base_url = kwargs.pop("base_url")
        else:
            base_url = "http://localhost:11434"
        self.client = Client(host=base_url)

    def chat(self, messages: List[Dict]) -> ChatResponse:
        completion = self.client.chat(model=self.model, messages=messages)

        return ChatResponse(
            content=completion.message.content,
            total_tokens=completion.prompt_eval_count + completion.eval_count,
        )
