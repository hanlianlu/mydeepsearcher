import logging
import os
from deepsearcher import configuration
from deepsearcher.offline_loading import load_from_local_files
from deepsearcher.online_query import query
from deepsearcher.configuration import Configuration, init_config
from deepsearcher.vector_db.milvus import Milvus
import glob

def collection_exists(vector_db, collection_name: str) -> bool:
    try:
        return vector_db.client.has_collection(collection_name)
    except Exception as e:
        print(f"Error checking collection: {e}")
        return False

httpx_logger = logging.getLogger("httpx")  # disable openai's logger output
httpx_logger.setLevel(logging.WARNING)

config = Configuration()  # Customize your config here
init_config(config=config)
vector_db = configuration.vector_db

# You should clone the milvus docs repo to your local machine first, execute:
# git clone https://github.com/milvus-io/milvus-docs.git
# Then replace the path below with the path to the milvus-docs repo on your local machine

all_md_files = glob.glob(r'/home/pmoAI/GITCODE/milvus-docs/site/en/**/*.md', recursive=True)
collection_name = "milvus_docs".replace(" ", "_").replace("-", "_").lower()
#check
existing_collections = vector_db.list_collections()
print("Existing collections:", [coll.collection_name for coll in existing_collections])

if collection_exists(vector_db, collection_name):
    print(f"Collection '{collection_name}' already exists. Skipping embedding and insertion.")
else:
    print(f"Collection '{collection_name}' does not exist. Proceeding with embedding and insertion.")
    
    load_from_local_files(
        paths_or_directory=all_md_files,
        collection_name=collection_name,
        collection_description="All Milvus Documents",
        force_new_collection=False,
        chunk_size=1500,
        chunk_overlap=100,
        batch_size=10  # Adjust this value as needed
    )



question = "what sustainability project in your azure blob data is most promising? "

_, _, consumed_token = query(question, max_iter=2)
print(f"Consumed tokens: {consumed_token}")
#Milvus.stop_server()  # Stop the Milvus server after use