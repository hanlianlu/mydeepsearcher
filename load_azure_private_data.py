from dotenv import load_dotenv
import argparse
import logging
import os
import tempfile
import shutil
from typing import Set, List, Tuple, Dict, Any
from pymilvus import Collection, utility, MilvusException
import re
from deepsearcher import configuration
from deepsearcher.offline_loading import load_from_local_files
from deepsearcher.online_query import query
from deepsearcher.configuration import Configuration, init_config
from deepsearcher.vector_db.milvus import Milvus
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress noisy third-party logs
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("azure").setLevel(logging.ERROR)

# Retrieve environment variables for storage accounts and containers
normal_account_url = os.environ.get("AZURE_NORMAL_ACCOUNT_URL")
normal_containers = os.environ.get("NORMAL_ACCOUNT_CONTAINERS", "").split(",") if os.environ.get("NORMAL_ACCOUNT_CONTAINERS") else []
secret_account_url = os.environ.get("AZURE_SECRET_ACCOUNT_URL")
secret_containers = os.environ.get("SECRET_ACCOUNT_CONTAINERS", "").split(",") if os.environ.get("SECRET_ACCOUNT_CONTAINERS") else []

# Create the container list as (account_url, container_name) tuples
CONTAINER_LIST = []
for container in normal_containers:
    if container.strip():
        CONTAINER_LIST.append((normal_account_url, container.strip()))
for container in secret_containers:
    if container.strip():
        CONTAINER_LIST.append((secret_account_url, container.strip()))

# Validate environment variables
if not CONTAINER_LIST or not normal_account_url or not secret_account_url:
    raise ValueError("No valid containers or account URLs defined in environment variables. Check AZURE_NORMAL_ACCOUNT_URL, NORMAL_ACCOUNT_CONTAINERS, AZURE_SECRET_ACCOUNT_URL, and SECRET_ACCOUNT_CONTAINERS in your .env file.")

def normalize_to_blob_endpoint(account_url: str) -> str:
    """Convert a DFS endpoint to a Blob endpoint if necessary."""
    if account_url.endswith(".dfs.core.windows.net"):
        account_name = account_url.split("//")[1].split(".")[0]
        return f"https://{account_name}.blob.core.windows.net"
    elif account_url.endswith(".blob.core.windows.net"):
        return account_url
    else:
        raise ValueError(f"Unsupported account URL format: {account_url}")

def get_account_name(account_url: str) -> str:
    """Extract the account name from the account URL."""
    match = re.search(r"https://([^.]+)\.(?:blob|dfs)\.core\.windows\.net", account_url)
    if match:
        return match.group(1)
    raise ValueError(f"Invalid account URL: {account_url}")

def get_embedded_references(vector_db: Milvus, collection_name: str) -> Set[str]:
    """Retrieve all embedded references from a Milvus collection efficiently."""
    if not utility.has_collection(collection_name):
        logger.info(f"Collection '{collection_name}' does not exist.")
        return set()

    collection_obj = Collection(name=collection_name)
    embedded_refs = set()
    try:
        iterator = collection_obj.query_iterator(batch_size=1000, output_fields=["reference"], expr="")
        while True:
            batch = iterator.next()
            if not batch:
                break
            embedded_refs.update(doc["reference"] for doc in batch if "reference" in doc)
        iterator.close()
        logger.info(f"Retrieved {len(embedded_refs)} embedded references from '{collection_name}'.")
        if embedded_refs:
            logger.debug(f"Sample references: {list(embedded_refs)[:5]}")
    except MilvusException as e:
        logger.error(f"Milvus error retrieving references from '{collection_name}': {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error retrieving references from '{collection_name}': {e}")
        raise
    return embedded_refs

def delete_removed_documents(vector_db: Milvus, collection_name: str, removed_refs: Set[str]) -> None:
    """Delete documents from Milvus whose references are no longer in the container."""
    for ref in removed_refs:
        try:
            vector_db.delete_document(collection=collection_name, reference=ref)
            logger.info(f"Deleted document with reference: {ref}")
        except Exception as e:
            logger.error(f"Error deleting reference '{ref}' from '{collection_name}': {e}")

def download_blobs_from_container(account_url: str, container: str, blob_names_to_download: Set[str]) -> Tuple[List[Tuple[str, Dict[str, Any]]], str]:
    """Download specified blobs from an Azure Blob Storage container with preserved file extensions."""
    credential = DefaultAzureCredential()
    normalized_account_url = normalize_to_blob_endpoint(account_url)
    blob_service_client = BlobServiceClient(account_url=normalized_account_url, credential=credential)
    container_client = blob_service_client.get_container_client(container)

    temp_dir = tempfile.mkdtemp(prefix=f"azure_blob_{container}_")
    downloaded_files = []

    for blob_name in blob_names_to_download:
        blob_client = container_client.get_blob_client(blob_name)
        extension = os.path.splitext(blob_name)[1] or ".bin"  # Default to .bin if no extension
        with tempfile.NamedTemporaryFile(dir=temp_dir, delete=False, suffix=extension) as temp_file:
            local_path = temp_file.name
            try:
                with open(local_path, "wb") as f:
                    stream = blob_client.download_blob()
                    f.write(stream.readall())
                blob_properties = blob_client.get_blob_properties()
                meta = {
                    "reference": blob_name,
                    "last_modified": str(blob_properties.last_modified),
                    "content_type": blob_properties.content_settings.content_type or "",
                    "creation_time": str(blob_properties.creation_time),
                    "url": blob_client.url
                }
                downloaded_files.append((local_path, meta))
            except Exception as e:
                logger.error(f"Failed to download blob '{blob_name}' from '{container}': {e}")
                if os.path.exists(local_path):
                    os.remove(local_path)
    return downloaded_files, temp_dir

def transform_collection_name(container: str) -> str:
    """Transform the input string into a 'word_word' style collection name."""
    words = re.split(r'[^a-zA-Z0-9]+', container)
    capitalized_words = [word.capitalize() for word in words if word]
    return '_'.join(capitalized_words)

def process_container(vector_db: Milvus, account_url: str, container: str) -> None:
    """Process a single Azure Blob Storage container, embedding new blobs and removing deleted ones."""
    account_name = get_account_name(account_url)
    collection_name = transform_collection_name(container)
    logger.info(f"Processing container '{container}' in account '{account_name}' with collection '{collection_name}'.")

    embedded_refs = get_embedded_references(vector_db, collection_name) if utility.has_collection(collection_name) else set()

    credential = DefaultAzureCredential()
    normalized_account_url = normalize_to_blob_endpoint(account_url)
    blob_service_client = BlobServiceClient(account_url=normalized_account_url, credential=credential)
    container_client = blob_service_client.get_container_client(container)

    try:
        container_description = container_client.get_container_properties().metadata.get("description", "")
    except Exception as e:
        logger.error(f"Error retrieving properties for '{container}' in '{account_name}': {e}")
        container_description = ""

    blob_names = set()
    try:
        for blob in container_client.list_blobs():
            blob_names.add(blob.name)
    except Exception as e:
        logger.error(f"Error listing blobs in '{container}' of '{account_name}': {e}")
        return

    new_blobs = blob_names - embedded_refs
    removed_blobs = embedded_refs - blob_names

    logger.info(f"New blobs to embed: {len(new_blobs)}")
    logger.debug(f"New blobs: {new_blobs}")
    logger.debug(f"Existing embedded refs: {embedded_refs}")
    logger.info(f"Removed blobs: {len(removed_blobs)}")

    if removed_blobs:
        delete_removed_documents(vector_db, collection_name, removed_blobs)

    temp_dir = None
    if new_blobs:
        files_to_embed, temp_dir = download_blobs_from_container(account_url, container, new_blobs)
        if files_to_embed:
            logger.info(f"Embedding {len(files_to_embed)} new files from '{container}' in '{account_name}'.")
            load_from_local_files(
                paths_or_directory=files_to_embed,
                collection_name=collection_name,
                collection_description=f"Collection for '{container}' in '{account_name}': {container_description}",
                force_new_collection=False,
                chunk_size=2048,
                chunk_overlap=220,
                batch_size=10
            )
        else:
            logger.info(f"No new files downloaded for '{container}' in '{account_name}'.")

    if temp_dir and os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.error(f"Error cleaning up '{temp_dir}': {e}")

def main() -> None:
    """Main function to process Azure Blob containers and optionally run a test query."""
    config = Configuration()
    init_config(config=config)
    vector_db = configuration.vector_db

    parser = argparse.ArgumentParser(description="Azure Blob Incremental Embedding")
    parser.add_argument("--container", type=str, help="Specific container to process in 'account_url:container_name' format")
    parser.add_argument("--test", action="store_true", help="Run a test query after embedding")
    args = parser.parse_args()

    if args.container:
        parts = args.container.split(":", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid container specification: {args.container}. Expected 'account_url:container_name'.")
        containers_to_process = [(parts[0].strip(), parts[1].strip())]
    else:
        containers_to_process = CONTAINER_LIST

    if not containers_to_process:
        logger.info("No containers to process. Exiting.")
        return

    for account_url, container in containers_to_process:
        process_container(vector_db, account_url, container)

    if args.test:
        question = "what do you know about from pike-rag-pipeline?"
        try:
            final_answer, retrieved_results, consumed_tokens = query(question, max_iter=1)
            logger.info(f"Test query result: {final_answer}")
            logger.info(f"Consumed tokens: {consumed_tokens}")
        except Exception as e:
            logger.error(f"Test query failed: {e}")

if __name__ == "__main__":
    main()