from typing import Dict, List, Optional
import os
import asyncio
import hashlib
import logging
from cachetools import TTLCache
from openai import AzureOpenAI as OpenAIAzureClient
from deepsearcher.llm.base import BaseLLM, ChatResponse
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from aiohttp.client_exceptions import ClientResponseError

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        cache_size: int = 10,          # Max number of cached responses
        cache_ttl: int = 1800,          # Cache time-to-live in seconds (0.5 hour)
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
        self.cache = TTLCache(maxsize=cache_size, ttl=cache_ttl)  # TTL cache with LRU eviction
        self.cache_size = cache_size

    def _get_cache_key(self, messages: List[Dict], max_tokens: Optional[int] = None) -> str:
        """Generate a unique cache key based on the input messages and max_tokens."""
        key_str = str(messages) + str(max_tokens) + str(self.reasoning_effort)
        return hashlib.sha256(key_str.encode()).hexdigest()

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(ClientResponseError)
    )
    def chat(self, messages: List[Dict], max_tokens: Optional[int] = None, **kwargs) -> ChatResponse:
        """
        Call the Azure OpenAI chat completions endpoint synchronously.

        :param messages: A list of message dicts for the conversation.
        :param max_tokens: Optional maximum number of tokens to generate.
        :return: ChatResponse object containing the response content and token usage.
        """
        cache_key = self._get_cache_key(messages, max_tokens)
        if cache_key in self.cache:
            logger.info("Cache hit for key: %s", cache_key)
            return self.cache[cache_key]

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
            logger.error("Chat completion API call failed: %s", str(e))
            raise RuntimeError(f"Chat completion API call failed: {e}")

        response = ChatResponse(
            content=completion.choices[0].message.content,
            total_tokens=completion.usage.total_tokens,
        )
        self.cache[cache_key] = response
        logger.info("Cache miss - stored response for key: %s", cache_key)
        return response

    async def chat_async(self, messages: List[Dict], max_tokens: Optional[int] = None, **kwargs) -> ChatResponse:
        """
        Call the Azure OpenAI chat completions endpoint asynchronously.

        This method first checks if the underlying client supports an async
        method (acreate). If it does, it uses it. Otherwise, it falls back to
        running the synchronous `chat` method in a separate thread.

        :param messages: A list of message dicts for the conversation.
        :param max_tokens: Optional maximum number of tokens to generate.
        :return: ChatResponse object containing the response content and token usage.
        """
        cache_key = self._get_cache_key(messages, max_tokens)
        if cache_key in self.cache:
            logger.info("Cache hit for key: %s", cache_key)
            return self.cache[cache_key]

        if hasattr(self.client.chat.completions, "acreate"):
            try:
                params = {
                    "model": self.model,
                    "messages": messages,
                }
                if self.reasoning_effort:
                    params["reasoning_effort"] = self.reasoning_effort
                if max_tokens is not None:
                    params["max_tokens"] = max_tokens
                async with self.semaphore:  # Apply rate limiting
                    completion = await self.client.chat.completions.acreate(**params)
            except Exception as e:
                logger.error("Async chat completion API call failed: %s", str(e))
                raise RuntimeError(f"Async chat completion API call failed: {e}")
            response = ChatResponse(
                content=completion.choices[0].message.content,
                total_tokens=completion.usage.total_tokens,
            )
        else:
            # Fallback: run the synchronous method in a thread
            async with self.semaphore:  # Apply rate limiting
                response = await asyncio.to_thread(self.chat, messages, max_tokens, **kwargs)

        self.cache[cache_key] = response
        logger.info("Cache miss - stored response for key: %s", cache_key)
        return response