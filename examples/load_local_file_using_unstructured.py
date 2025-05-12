import logging
import os
from deepsearcher.offline_loading import load_from_local_files
from deepsearcher.online_query import query
from deepsearcher.configuration import Configuration, init_config

# Suppress unnecessary logging from third-party libraries
logging.getLogger("httpx").setLevel(logging.WARNING)

# Set API keys (ensure these are set securely in real applications)
os.environ['UNSTRUCTURED_API_KEY'] = '***************'
os.environ['UNSTRUCTURED_API_URL'] = '***************'


def main():
    # Step 1: Initialize configuration
    config = Configuration()

    # Configure Vector Database (Milvus) and File Loader (UnstructuredLoader)
    config.set_provider_config("vector_db", "Milvus", {})
    config.set_provider_config("file_loader", "UnstructuredLoader", {})

    # Apply the configuration
    init_config(config)

    # Step 2: Load data from a local file or directory into Milvus
    input_file = "your_local_file_or_directory"  # Replace with your actual file path
    collection_name = "Unstructured"
    collection_description = "All Milvus Documents"

    load_from_local_files(paths_or_directory=input_file, collection_name=collection_name, collection_description=collection_description)

    # Step 3: Query the loaded data
    question = "What is Milvus?"  # Replace with your actual question
    result = query(question)


if __name__ == "__main__":
    main()
