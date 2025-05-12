from typing import List

from tqdm import tqdm

from deepsearcher.loader.splitter import Chunk


class BaseEmbedding:
    def embed_query(self, text: str) -> List[float]:
        pass

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_chunks(self, chunks: List[Chunk], batch_size=256) -> List[Chunk]:
        texts = [chunk.text for chunk in chunks]
        batch_texts = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
        embeddings = []
        for batch_text in tqdm(batch_texts, desc="Embedding chunks"):
            batch_embeddings = self.embed_documents(batch_text)
            embeddings.extend(batch_embeddings)
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        return chunks

    @property
    def dimension(self) -> int:
        pass
