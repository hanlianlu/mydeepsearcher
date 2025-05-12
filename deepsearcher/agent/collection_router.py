from typing import List, Tuple

from deepsearcher.agent.base import BaseAgent
from deepsearcher.llm.base import BaseLLM
from deepsearcher.tools import log
from deepsearcher.vector_db.base import BaseVectorDB

COLLECTION_ROUTE_PROMPT  = """
You are provided with a QUESTION, a COLLECTION_INFO containing collection_name(s) and their corresponding collection_description(s), and the Current Iteration.

Your task is to:
1. Identify all collection_name(s) relevant to answering the QUESTION.
2. Strictly IGNORE any information provided by the current iteration UNLESS Current Iteration is exactly "first".

Rules you MUST follow:
- If Current Iteration is "first" AND you find NO relevant collections to answer the question, you MUST return ALL available collection_names as a Python list of strings.

Your response MUST be a valid Python list of strings containing ONLY collection_name(s), WITHOUT any additional text or explanation.

QUESTION: {question}
COLLECTION_INFO: {collection_info}
Current Iteration: {curr_iter_str}

Your selected collection name list is:
"""


class CollectionRouter(BaseAgent):
    def __init__(self, llm: BaseLLM, vector_db: BaseVectorDB, **kwargs):
        self.llm = llm
        self.vector_db = vector_db
        self.all_collections = [
            collection_info.collection_name for collection_info in self.vector_db.list_collections()
        ]

    def invoke(self, query: str, collections_names: list = None, curr_iter:int=1, **kwargs) -> Tuple[List[str], int]:
        if curr_iter == 1:
            curr_iter_str = "first"
        elif curr_iter == 0:
            curr_iter_str = "first"
        else:
            curr_iter_str = "subsequent"
        print(f"curr_iter_str is {curr_iter_str}\n ")
        consume_tokens = 0
        if not collections_names:
            collection_infos = self.vector_db.list_collections()
        else:
            collection_infos = self.vector_db.list_collections(collections_names )
        if len(collection_infos) == 0:
            log.warning(
                "No collections found in the vector database. Please check the database connection."
            )
            return [], 0
        if len(collection_infos) == 1:
            the_only_collection = collection_infos[0].collection_name
            log.color_print(
                f"<think> Perform search [{query}] on the vector DB collection: {the_only_collection} </think>\n"
            )
            return [the_only_collection], 0
        vector_db_search_prompt = COLLECTION_ROUTE_PROMPT.format(
            question=query,
            collection_info=[
                {
                    "collection_name": collection_info.collection_name,
                    "collection_description": collection_info.description
                   
                }
                for collection_info in collection_infos
            ],
            curr_iter_str = curr_iter_str
        )
        chat_response = self.llm.chat(
            messages=[{"role": "user", "content": vector_db_search_prompt}]
        )
        selected_collections = self.llm.literal_eval(chat_response.content)
        consume_tokens += chat_response.total_tokens
        
        for collection_info in collection_infos:
            # If a collection description is not provided, use the query as the search query
            if not collection_info.description:
                selected_collections.append(collection_info.collection_name)
            # If the default collection exists, use the query as the search query
            if self.vector_db.default_collection == collection_info.collection_name:
                selected_collections.append(collection_info.collection_name)
        selected_collections = list(set(selected_collections))
        log.color_print(
            f"<think> Perform search [{query}] on the vector DB collections: {selected_collections} </think>\n"
        )
        return selected_collections, consume_tokens
