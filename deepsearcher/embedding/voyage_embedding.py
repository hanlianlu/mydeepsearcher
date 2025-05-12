import os
from typing import List

from deepsearcher.embedding.base import BaseEmbedding

VOYAGE_MODEL_DIM_MAP = {
    "voyage-3-large": 1024,
    "voyage-3": 1024,
    "voyage-3-lite": 512,
}


class VoyageEmbedding(BaseEmbedding):
    """
    https://docs.voyageai.com/embeddings/
    """

    def __init__(self, model="voyage-3", **kwargs):
        if "model_name" in kwargs and (not model or model == "voyage-3"):
            model = kwargs.pop("model_name")
        self.model = model
        if "api_key" in kwargs:
            api_key = kwargs.pop("api_key")
        else:
            api_key = os.getenv("VOYAGE_API_KEY")
        self.voyageai_api_key = api_key

        import voyageai

        voyageai.api_key = self.voyageai_api_key
        self.vo = voyageai.Client(**kwargs)

    def embed_query(self, text: str) -> List[float]:
        """
        input_type (`str`): "query" or "document" for retrieval case.
        """
        embeddings = self.vo.embed([text], model=self.model, input_type="query")
        return embeddings.embeddings[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.vo.embed(texts, model=self.model, input_type="document")
        return embeddings.embeddings

    @property
    def dimension(self) -> int:
        return VOYAGE_MODEL_DIM_MAP[self.model]
