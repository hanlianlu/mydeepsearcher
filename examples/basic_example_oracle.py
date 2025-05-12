import sys, os
from pathlib import Path
script_directory = Path(__file__).resolve().parent.parent
sys.path.append(os.path.abspath(script_directory))

import logging

httpx_logger = logging.getLogger("httpx")  # disable openai's logger output
httpx_logger.setLevel(logging.WARNING)

current_dir = os.path.dirname(os.path.abspath(__file__))

# Customize your config here
from deepsearcher.configuration import Configuration, init_config

config = Configuration()
init_config(config=config)

# # Load your local data
# # Hint: You can load from a directory or a single file, please execute it in the root directory of the deep searcher project

from deepsearcher.offline_loading import load_from_local_files

load_from_local_files(
    paths_or_directory=os.path.join(current_dir, "data/WhatisMilvus.pdf"),
    collection_name="milvus_docs",
    collection_description="All Milvus Documents",
    # force_new_collection=True, # If you want to drop origin collection and create a new collection every time, set force_new_collection to True
)

# Query
from deepsearcher.online_query import query

question = 'Write a report comparing Milvus with other vector databases.'
answer, retrieved_results, consumed_token = query(question)
print(answer)

# # get consumed tokens, about: 2.5~3w tokens when using openai gpt-4o model
# print(f"Consumed tokens: {consumed_token}")

