import os
from typing import List, Union, Tuple, Dict, Any
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from deepsearcher import configuration
from deepsearcher.loader.splitter import split_docs_to_chunks

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def batch_embed_chunks(embedding_model, chunks: List, batch_size: int = 10, max_workers: int = 5) -> List:
    """
    Process embedding for a list of chunks in concurrent batches.

    Args:
        embedding_model: The embedding model object from configuration.
        chunks: List of chunk objects to embed.
        batch_size: Number of chunks to process in each API call. Defaults to 10.
        max_workers: Number of concurrent workers. Defaults to 5.

    Returns:
        List: List of embedded chunks with their embeddings.

    Raises:
        Exception: Propagates embedding errors for logging in the caller.
    """
    results = []
    batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {
            executor.submit(embedding_model.embed_chunks, batch): batch for batch in batches
        }
        for future in tqdm(as_completed(future_to_batch), total=len(batches), desc="Embedding batches"):
            try:
                batch_result = future.result()
                results.extend(batch_result)
            except Exception as e:
                logger.error(f"Error processing a batch: {e}")
    return results

def load_from_local_files(
    paths_or_directory: Union[str, List[Union[str, Tuple[str, Dict[str, Any]]]]],
    collection_name: str = None,
    collection_description: str = None,
    force_new_collection: bool = False,
    chunk_size: int = 3000,
    chunk_overlap: int = 280,
    batch_size: int = 10,
    max_workers: int = 5
) -> None:
    """
    Load, split, embed, and insert local files or directories into Milvus efficiently.

    Args:
        paths_or_directory: A single path (str) or list of paths or tuples (path, metadata).
        collection_name: Name of the Milvus collection. Defaults to vector_db's default if None.
        collection_description: Description of the collection. Optional.
        force_new_collection: If True, drops the existing collection. Defaults to False.
        chunk_size: Size of each document chunk in characters. Defaults to 3000.
        chunk_overlap: Overlap between chunks in characters. Defaults to 280.
        batch_size: Number of chunks per embedding batch. Defaults to 10.
        max_workers: Number of concurrent workers for loading files. Defaults to 5.
    """
    vector_db = configuration.vector_db
    embedding_model = configuration.embedding_model
    file_loader = configuration.file_loader

    if collection_name is None:
        collection_name = vector_db.default_collection
    collection_name = collection_name.replace(" ", "_").replace("-", "_")

    try:
        vector_db.init_collection(
            dim=embedding_model.dimension,
            collection=collection_name,
            description=collection_description,
            force_new_collection=force_new_collection,
        )
    except Exception as e:
        logger.error(f"Failed to initialize collection '{collection_name}': {e}")
        return

    if isinstance(paths_or_directory, str):
        paths_or_directory = [paths_or_directory]

    all_docs = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {}
        for item in paths_or_directory:
            if isinstance(item, tuple):
                path, extra_meta = item
            else:
                path, extra_meta = item, {}

            if not os.path.exists(path):
                logger.error(f"File or directory '{path}' does not exist.")
                continue

            if os.path.isdir(path):
                future = executor.submit(file_loader.load_directory, path)
                future_to_path[future] = (path, extra_meta)
            else:
                # Silently ignore .bin files
                if path.lower().endswith('.bin'):
                    continue
                future = executor.submit(file_loader.load_file, path)
                future_to_path[future] = (path, extra_meta)

        for future in tqdm(as_completed(future_to_path), total=len(future_to_path), desc="Loading files"):
            path, extra_meta = future_to_path[future]
            try:
                docs = future.result()
                for doc in docs:
                    base_filename = doc.metadata.get("file_name", os.path.basename(path))
                    doc.metadata["reference"] = extra_meta.get("reference", base_filename)
                    doc.metadata.update(extra_meta)
                    logger.debug(f"Set reference for file: {doc.metadata['reference']}")
                all_docs.extend(docs)
            except Exception as e:
                logger.error(f"Failed to load '{path}': {e}")

    if not all_docs:
        logger.info("No documents loaded. Aborting.")
        return

    chunks = split_docs_to_chunks(all_docs)
    if not chunks:
        logger.error("No chunks generated. Aborting.")
        return

    embedded_chunks = batch_embed_chunks(embedding_model, chunks, batch_size=batch_size)
    if not embedded_chunks:
        logger.error("No embeddings generated. Aborting.")
        return

    try:
        vector_db.insert_data(collection=collection_name, chunks=embedded_chunks)
        logger.info(f"Inserted {len(embedded_chunks)} chunks into '{collection_name}'.")
    except Exception as e:
        logger.error(f"Failed to insert chunks into '{collection_name}': {e}")

def load_from_website(
    urls: Union[str, List[str]],
    collection_name: str = "web_crawler",
    collection_description: str = "This is a temp websearch collection.",
    force_new_collection: bool = False,
    **crawl_kwargs
) -> None:
    """
    Load, split, embed, and insert web content from URLs into Milvus.

    Args:
        urls: Single URL (str) or list of URLs to crawl.
        collection_name: Name of the Milvus collection. Defaults to "web_crawler".
        collection_description: Description of the collection. Defaults to a temp description.
        force_new_collection: If True, drops the existing collection. Defaults to False.
        **crawl_kwargs: Additional arguments for the web crawler.
    """
    vector_db = configuration.vector_db
    embedding_model = configuration.embedding_model
    web_crawler = configuration.web_crawler

    if isinstance(urls, str):
        urls = [urls]

    try:
        vector_db.init_collection(
            dim=embedding_model.dimension,
            collection=collection_name,
            description=collection_description,
            force_new_collection=force_new_collection,
        )
    except Exception as e:
        logger.error(f"Failed to initialize collection '{collection_name}': {e}")
        return

    try:
        all_docs = web_crawler.crawl_urls(urls, **crawl_kwargs)
        if not all_docs:
            logger.error("No documents crawled. Aborting.")
            return
    except Exception as e:
        logger.error(f"Failed to crawl URLs: {e}")
        return

    chunks = split_docs_to_chunks(all_docs)
    if not chunks:
        logger.error("No chunks generated from crawled documents. Aborting.")
        return

    embedded_chunks = batch_embed_chunks(embedding_model, chunks)
    if not embedded_chunks:
        logger.error("No embeddings generated. Aborting.")
        return

    try:
        vector_db.insert_data(collection=collection_name, chunks=embedded_chunks)
        logger.info(f"Inserted {len(embedded_chunks)} chunks into '{collection_name}'.")
    except Exception as e:
        logger.error(f"Failed to insert chunks into '{collection_name}': {e}")