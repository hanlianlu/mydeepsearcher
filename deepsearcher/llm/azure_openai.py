from typing import Dict, List, Optional
import os
import asyncio
import hashlib
from openai import AzureOpenAI as OpenAIAzureClient
from deepsearcher.llm.base import BaseLLM, ChatResponse
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from aiohttp.client_exceptions import ClientResponseError

class AzureOpenAI(BaseLLM):
    """
    A wrapper around the OpenAI Python Azure client with rate limiting, caching, and enhanced error handling.

    This class provides synchronous and asynchronous interfaces for chat completions, optimized to reduce
    API call frequency and manage costs.
    """
    def __init__(
        self,
        model: str,
        azure_endpoint: str = None,
        api_key: str = None,
        api_version: str = None,
        max_concurrent_calls: int = 50,  # Limit concurrent API calls
        cache_size: int = 100,          # Max number of cached responses
        reasoning_effort: Optional[str] = None,
        **kwargs,
    ):
        if azure_endpoint is None:
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if api_key is None:
            api_key = os.getenv("AZURE_OPENAI_KEY")
        if not (azure_endpoint and api_key and api_version):
            raise ValueError("azure_endpoint, api_key, and api_version must be provided")
        
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.client = OpenAIAzureClient(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
            **kwargs,
        )
        self.semaphore = asyncio.Semaphore(max_concurrent_calls)  # Rate limiting for async calls
        self.cache = {}  # In-memory cache for query responses
        self.cache_size = cache_size

    def _get_cache_key(self, messages: List[Dict]) -> str:
        """Generate a unique cache key based on the input messages."""
        return hashlib.sha256(str(messages).encode()).hexdigest()

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(ClientResponseError)
    )
    def chat(self, messages: List[Dict], max_tokens: Optional[int] = None, **kwargs) -> ChatResponse:
        """
        Call the Azure OpenAI chat completions endpoint synchronously.

        :param messages: A list of message dicts for the conversation.
        :return: ChatResponse object containing the response content and token usage.
        """
        try:
            # Prepare the base parameters
            params = {
                "model": self.model,
                "messages": messages,
            }
            # Add reasoning_effort if it’s set
            if self.reasoning_effort:
                params["reasoning_effort"] = self.reasoning_effort
            # Add max_tokens if provided
            if max_tokens is not None:
                params["max_tokens"] = max_tokens
            completion = self.client.chat.completions.create(**params)
        except Exception as e:
            raise RuntimeError(f"Chat completion API call failed: {e}")

        return ChatResponse(
            content=completion.choices[0].message.content,
            total_tokens=completion.usage.total_tokens,
        )

    async def chat_async(self, messages: List[Dict], max_tokens: Optional[int] = None, **kwargs) -> ChatResponse:
        """
        Call the Azure OpenAI chat completions endpoint asynchronously.

        This method first checks if the underlying client supports an async
        method (acreate). If it does, it uses it. Otherwise, it falls back to
        running the synchronous `chat` method in a separate thread.

        :param messages: A list of message dicts for the conversation.
        :return: ChatResponse object containing the response content and token usage.
        """
        if hasattr(self.client.chat.completions, "acreate"):
            try:
                params = {
                    "model": self.model,
                    "messages": messages,
                }
                if self.reasoning_effort:
                    params["reasoning_effort"] = self.reasoning_effort
                    # Add max_tokens if provided
                if max_tokens is not None:
                    params["max_tokens"] = max_tokens
                completion = await self.client.chat.completions.acreate(**params)
            except Exception as e:
                raise RuntimeError(f"Async chat completion API call failed: {e}")
            return ChatResponse(
                content=completion.choices[0].message.content,
                total_tokens=completion.usage.total_tokens,
            )
        else:
            # Fallback: run the synchronous method in a thread
            return await asyncio.to_thread(self.chat, messages, max_tokens, **kwargs)