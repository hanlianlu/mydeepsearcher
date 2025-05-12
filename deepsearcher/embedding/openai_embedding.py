import os
from typing import List

from openai._types import NOT_GIVEN
from deepsearcher.embedding.base import BaseEmbedding

OPENAI_MODEL_DIM_MAP = {
    "text-embedding-ada-002": 1536,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}

class OpenAIEmbedding(BaseEmbedding):
    """
    Implements text embeddings using Azure OpenAI.
    Based on https://platform.openai.com/docs/guides/embeddings/use-cases.
    """
    def __init__(self, model: str = "text-embedding-3-large", **kwargs):
        """
        Args:
            model (str):
                One of:
                    'text-embedding-ada-002': default dimension 1536,
                    'text-embedding-3-small': default dimension 1536,
                    'text-embedding-3-large': default dimension 3072.
            Additional kwargs must include Azure parameters: 
                - api_key,
                - azure_endpoint (or base_url),
                - api_version.
            Extra keys (like 'dimension') are used only locally.
        """
        from openai import AzureOpenAI

        # Get API key: use provided one or environment variable (using AZURE_OPENAI_KEY for Azure)
        api_key = kwargs.pop("api_key", os.getenv("AZURE_OPENAI_KEY"))
        # Get endpoint: check for azure_endpoint or base_url
        endpoint = kwargs.pop("azure_endpoint", None) or kwargs.pop("base_url", os.getenv("OPENAI_BASE_URL"))
        # Get api_version from kwargs or environment variable, defaulting to a chosen version for embeddings.
        api_version = kwargs.pop("api_version", os.getenv("OPENAI_API_VERSION", "2023-05-15"))
        if not (endpoint and api_key and api_version):
            raise ValueError("Must provide endpoint, api_key, and api_version for Azure OpenAI")

        # Handle dimension separately; remove from kwargs so it's not passed to AzureOpenAI
        if "dimension" in kwargs:
            dimension = kwargs.pop("dimension")
        else:
            dimension = OPENAI_MODEL_DIM_MAP.get(model, NOT_GIVEN)
        self.dim = dimension
        self.model = model
        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
            **kwargs
        )

    def _get_dim(self):
        return self.dim if self.model != "text-embedding-ada-002" else NOT_GIVEN

    def embed_query(self, text: str) -> List[float]:
        kwargs = {}
        if self.model in ["text-embedding-3-small", "text-embedding-3-large"]:
            kwargs["dimensions"] = self.dim
        response = self.client.embeddings.create(
            input=[text],
            model=self.model,
            **kwargs
        )
        return response.data[0].embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        kwargs = {}
        if self.model in ["text-embedding-3-small", "text-embedding-3-large"]:
            kwargs["dimensions"] = self.dim
        response = self.client.embeddings.create(
            input=texts,
            model=self.model,
            **kwargs
        )
        return [r.embedding for r in response.data]

    @property
    def dimension(self) -> int:
        return self.dim
