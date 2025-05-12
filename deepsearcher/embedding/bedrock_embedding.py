import json
import os
from typing import List

from deepsearcher.embedding.base import BaseEmbedding

MODEL_ID_TITAN_TEXT_G1 = "amazon.titan-embed-text-v1"
MODEL_ID_TITAN_TEXT_V2 = "amazon.titan-embed-text-v2:0"
MODEL_ID_TITAN_MULTIMODAL_G1 = "amazon.titan-embed-image-v1"
MODEL_ID_COHERE_ENGLISH_V3 = "cohere.embed-english-v3"
MODEL_ID_COHERE_MULTILINGUAL_V3 = "cohere.embed-multilingual-v3"

BEDROCK_MODEL_DIM_MAP = {
    MODEL_ID_TITAN_TEXT_G1: 1536,
    MODEL_ID_TITAN_TEXT_V2: 1024,
    MODEL_ID_TITAN_MULTIMODAL_G1: 1024,
    MODEL_ID_COHERE_ENGLISH_V3: 1024,
    MODEL_ID_COHERE_MULTILINGUAL_V3: 1024,
}

DEFAULT_MODEL_ID = MODEL_ID_TITAN_TEXT_V2


class BedrockEmbedding(BaseEmbedding):
    def __init__(self, model: str = DEFAULT_MODEL_ID, **kwargs):
        """
        Args:
            model_name (`str`):
                Can be one of the following:
                'amazon.titan-embed-text-v2:0': dimensions include 256, 512, 1024, default is 1024,
        """
        import boto3

        aws_access_key_id = kwargs.pop("aws_access_key_id", os.getenv("AWS_ACCESS_KEY_ID"))
        aws_secret_access_key = kwargs.pop(
            "aws_secret_access_key", os.getenv("AWS_SECRET_ACCESS_KEY")
        )

        if model in {None, DEFAULT_MODEL_ID} and "model_name" in kwargs:
            model = kwargs.pop("model_name")  # overwrites `model` with `model_name`

        self.model = model

        # TODO: initiate boto3 client
        self.client = boto3.client(
            "bedrock-runtime",
            region_name="us-east-1",  # FIXME: allow users to specify
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )

    def embed_query(self, text: str) -> List[float]:
        response = self.client.invoke_model(
            modelId=self.model, body=json.dumps({"inputText": text})
        )
        model_response = json.loads(response["body"].read())
        embedding = model_response["embedding"]
        return embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]

    @property
    def dimension(self) -> int:
        return BEDROCK_MODEL_DIM_MAP[self.model]
