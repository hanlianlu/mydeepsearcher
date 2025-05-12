import os
import json
from typing import List, Optional, Union

import numpy as np
try:
    from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility, MilvusException
except Exception:
    # Stub out pymilvus if it isn’t available or misconfigured
    class MilvusException(Exception):
        """Dummy exception when pymilvus import fails."""
        pass

    connections = None
    Collection = type("Collection", (), {})
    FieldSchema = type("FieldSchema", (), {})
    CollectionSchema = type("CollectionSchema", (), {})
    DataType = None
    utility = None

from deepsearcher.tools import log
from deepsearcher.loader.splitter import Chunk
from deepsearcher.vector_db.base import BaseVectorDB, CollectionInfo, RetrievalResult

class Milvus(BaseVectorDB):
    """Milvus client for standalone usage with HNSW indexing and COSINE similarity."""

    def __init__(
        self,
        default_collection: str = "default",
        host: str = "localhost",
        port: str = "19530",
        **kwargs
    ):
        """Initialize the Milvus client and establish connection."""
        super().__init__(default_collection)
        self.default_collection = default_collection

        if not connections.has_connection("default"):
            safe_host = os.getenv("MILVUS_HOST", host)
            safe_port = os.getenv("MILVUS_PORT", port)
            try:
                connections.connect(alias="default", host=safe_host, port=safe_port)
                log.info(f"Connected to Milvus at {safe_host}:{safe_port}")
            except Exception as e:
                log.critical(f"Failed to connect to Milvus at {safe_host}:{safe_port}: {e}")
                raise
        else:
            log.info("Using existing Milvus connection")

    def init_collection(
        self,
        dim: int = 3072,
        collection: Optional[str] = None,
        description: Optional[str] = "",
        force_new_collection: bool = False,
        text_max_length: int = 65535,
        reference_max_length: int = 2048,
        metric_type: str = "COSINE",
        *args,
        **kwargs,
    ) -> None:
        """Create or initialize a collection with HNSW index."""
        collection = collection or self.default_collection
        try:
            if force_new_collection and utility.has_collection(collection):
                utility.drop_collection(collection)
                log.info(f"Dropped collection '{collection}' (force_new_collection=True)")

            if utility.has_collection(collection):
                log.info(f"Collection '{collection}' exists")
                return

            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=text_max_length),
                FieldSchema(name="reference", dtype=DataType.VARCHAR, max_length=reference_max_length),
                FieldSchema(name="metadata", dtype=DataType.JSON),
            ]
            log.info(f"Initializing collection '{collection}' with description: {description}")
            schema = CollectionSchema(fields=fields, description=description or "")
            collection_obj = Collection(name=collection, schema=schema)
            log.info(f"Created collection '{collection}'")

            index_params = {
                "index_type": "HNSW",
                "metric_type": metric_type,
                "params": {"M": 80, "efConstruction": 400}
            }
            collection_obj.create_index("vector", index_params)
            log.info(f"Created HNSW index with {metric_type} metric")

            collection_obj.load()
            log.info(f"Loaded collection '{collection}'")
        except MilvusException as e:
            log.critical(f"Milvus error initializing collection '{collection}': {e}")
            raise
        except Exception as e:
            log.critical(f"Unexpected error initializing collection '{collection}': {e}")
            raise

    def insert_data(
        self,
        collection: Optional[str] = None,
        chunks: List[Chunk] = None,
        batch_size: int = 256,
        *args,
        **kwargs,
    ):
        """Insert chunks into a Milvus collection with error handling."""
        collection = collection or self.default_collection
        if not chunks or not isinstance(chunks, list):
            log.error("No valid chunks provided")
            return

        if not utility.has_collection(collection):
            log.error(f"Collection '{collection}' does not exist")
            return

        collection_obj = Collection(collection)
        vector_field = next(f for f in collection_obj.schema.fields if f.name == "vector")
        expected_dim = vector_field.dim
        log.info(f"Inserting into '{collection}' (dim={expected_dim})")

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_data = []

            for chunk in batch_chunks:
                if not isinstance(chunk.embedding, list) or len(chunk.embedding) != expected_dim or not all(isinstance(x, float) for x in chunk.embedding):
                    log.error(f"Invalid embedding for chunk (ref={chunk.reference}): skipping")
                    continue

                metadata = json.dumps(chunk.metadata) if isinstance(chunk.metadata, dict) else chunk.metadata
                batch_data.append({
                    "vector": chunk.embedding,
                    "text": chunk.text,
                    "reference": chunk.reference,
                    "metadata": metadata,
                })

            if not batch_data:
                log.warning(f"Batch {i//batch_size + 1} empty after validation")
                continue

            try:
                mr = collection_obj.insert(batch_data)
                log.info(f"Inserted {len(batch_data)} entities (IDs: {mr.primary_keys[:5]}...)")
            except MilvusException as e:
                log.error(f"Milvus error inserting batch into '{collection}': {e}")
            except Exception as e:
                log.error(f"Unexpected error inserting batch into '{collection}': {e}")

        log.info(f"Processed {len(chunks)} chunks into '{collection}'")

    def search_data(
        self,
        collection: Optional[str] = None,
        vector: Union[np.ndarray, List[float]] = None,
        top_k: int = 9,
        include_embedding: bool = False,  # New parameter
        *args,
        **kwargs,
    ) -> List[RetrievalResult]:
        """Search for similar vectors using COSINE similarity.

        Args:
            collection (Optional[str]): The collection to search in. Defaults to the default collection.
            vector (Union[np.ndarray, List[float]]): The query vector.
            top_k (int): The number of top results to retrieve. Defaults to 11.
            include_embedding (bool): Whether to include the embedding in the results. Defaults to False.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            List[RetrievalResult]: A list of retrieval results.
        """
        collection = collection or self.default_collection
        if not vector:
            log.error("No vector provided for search")
            return []

        if isinstance(vector, np.ndarray):
            vector = vector.tolist()

        if not utility.has_collection(collection):
            log.error(f"Collection '{collection}' does not exist")
            return []

        collection_obj = Collection(collection)
        try:
            collection_obj.load()
            log.info(f"Loaded collection '{collection}' into memory")
        except MilvusException as e:
            log.critical(f"Failed to load collection '{collection}': {e}")
            raise

        search_params = kwargs.get("search_params", {"metric_type": "COSINE", "params": {"ef": 450}})

        # Dynamically set output fields based on include_embedding
        output_fields = ["text", "reference", "metadata"]
        if include_embedding:
            output_fields.append("vector")

        try:
            search_results = collection_obj.search(
                data=[vector],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                output_fields=output_fields,
            )
            if not search_results:
                log.info(f"No search results found in '{collection}'")
                return []

            retrieval_results = []
            for hits in search_results:  # search_results is a list of Hits objects
                for hit in hits:  # hits is iterable, containing Hit objects
                    entity = hit.entity
                    # Safely retrieve fields using fields attribute
                    text = entity.get("text") if "text" in entity.fields else ""
                    reference = entity.get("reference") if "reference" in entity.fields else ""
                    metadata_raw = entity.get("metadata") if "metadata" in entity.fields else "{}"
                    try:
                        metadata = json.loads(metadata_raw)
                    except json.JSONDecodeError as e:
                        log.warning(f"Failed to parse metadata for hit (id={hit.id}): {e}")
                        metadata = {}
                    # Conditionally include embedding if requested
                    embedding = entity.get("vector") if include_embedding and "vector" in entity.fields else None
                    retrieval_results.append(
                        RetrievalResult(
                            embedding=embedding,
                            text=text,
                            reference=reference,
                            score=hit.distance,  # COSINE: higher is better
                            metadata=metadata
                        )
                    )
            log.info(f"Retrieved {len(retrieval_results)} results from '{collection}'")
            return retrieval_results
        except MilvusException as e:
            log.critical(f"Milvus error searching '{collection}': {e}")
            raise
        except Exception as e:
            log.critical(f"Unexpected error searching '{collection}': {e}")
            raise

    def list_collections(self, collections: Optional[List[str]] = None, *args, **kwargs) -> List[CollectionInfo]:
        """List all or specified collections."""
        try:
            all_collections = utility.list_collections()
            target_collections = collections or all_collections
            return [
                CollectionInfo(
                    collection_name=coll,
                    description=Collection(coll).description
                )
                for coll in target_collections if coll in all_collections
            ]
        except MilvusException as e:
            log.critical(f"Milvus error listing collections: {e}")
            return []
        except Exception as e:
            log.critical(f"Unexpected error listing collections: {e}")
            return []

    def clear_db(self, collection: str = None, *args, **kwargs) -> None:
        """Drop a collection."""
        collection = collection or self.default_collection
        try:
            if utility.has_collection(collection):
                utility.drop_collection(collection)
                log.info(f"Dropped collection '{collection}'")
            else:
                log.info(f"Collection '{collection}' does not exist, nothing to drop")
        except MilvusException as e:
            log.warning(f"Milvus error dropping '{collection}': {e}")
        except Exception as e:
            log.warning(f"Unexpected error dropping '{collection}': {e}")

    def delete_document(self, collection: str, reference: str) -> None:
        """Delete documents by reference."""
        if not utility.has_collection(collection):
            log.info(f"Collection '{collection}' does not exist, nothing to delete")
            return

        collection_obj = Collection(collection)
        try:
            ids = [d["id"] for d in collection_obj.query(expr=f'reference == "{reference}"', output_fields=["id"])]
            if ids:
                collection_obj.delete(expr=f"id in [{','.join(map(str, ids))}]")
                log.info(f"Deleted {len(ids)} documents (ref={reference})")
            else:
                log.info(f"No documents found (ref={reference})")
        except MilvusException as e:
            log.critical(f"Milvus error deleting documents (ref={reference}): {e}")
            raise
        except Exception as e:
            log.critical(f"Unexpected error deleting documents (ref={reference}): {e}")
            raise

    def rename_collection(self, old_name: str, new_name: str) -> None:
        """Rename a collection using Milvus's built-in utility.

        Args:
            old_name (str): The current name of the collection to rename.
            new_name (str): The new name for the collection.
        """
        try:
            # Check if the old collection exists
            if not utility.has_collection(old_name):
                log.error(f"Collection '{old_name}' does not exist")
                return
            # Check if the new collection name already exists
            if utility.has_collection(new_name):
                log.error(f"Collection '{new_name}' already exists")
                return
            # Rename the collection
            utility.rename_collection(old_name, new_name)
            log.info(f"Successfully renamed collection '{old_name}' to '{new_name}'")
        except MilvusException as e:
            log.critical(f"Milvus error renaming collection '{old_name}' to '{new_name}': {e}")
            raise
        except Exception as e:
            log.critical(f"Unexpected error renaming collection '{old_name}' to '{new_name}': {e}")
            raise