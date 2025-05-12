from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import TokenTextSplitter
import tiktoken

class Chunk:
    def __init__(
        self,
        text: str,
        reference: str,
        metadata: dict = None,
        embedding: List[float] = None,
    ):
        self.text = text
        self.reference = reference
        self.metadata = metadata or {}
        self.embedding = embedding or None

def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count the number of tokens in a text using tiktoken."""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))

def split_large_chunk(
    doc: Document,
    max_tokens: int = 1024,
    target_chunk_size: int = 512,
    chunk_overlap: int = 50,
    encoding_name: str = "cl100k_base"
) -> List[Document]:
    """Split a large document into smaller chunks if it exceeds max_tokens."""
    token_count = count_tokens(doc.page_content, encoding_name)
    if token_count <= max_tokens:
        return [doc]
    
    # Split if too large
    text_splitter = TokenTextSplitter(
        chunk_size=target_chunk_size,
        chunk_overlap=chunk_overlap,
        encoding_name=encoding_name
    )
    split_docs = text_splitter.split_documents([doc])
    
    # Add sub_chunk_id to metadata
    for i, split_doc in enumerate(split_docs):
        split_doc.metadata["sub_chunk_id"] = i
    return split_docs

def split_docs_to_chunks(
    documents: List[Document],
    max_tokens: int = 1024,
    target_chunk_size: int = 512,
    chunk_overlap: int = 50,
    encoding_name: str = "cl100k_base"
) -> List[Chunk]:
    """Split documents into chunks, preserving structure unless too large."""
    all_chunks = []
    for doc in documents:
        split_docs = split_large_chunk(
            doc,
            max_tokens=max_tokens,
            target_chunk_size=target_chunk_size,
            chunk_overlap=chunk_overlap,
            encoding_name=encoding_name
        )
        for split_doc in split_docs:
            reference = split_doc.metadata.get("reference", "")
            chunk = Chunk(
                text=split_doc.page_content,
                reference=reference,
                metadata=split_doc.metadata.copy()  # Shallow copy
            )
            all_chunks.append(chunk)
    return all_chunks