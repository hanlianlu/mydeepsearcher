from typing import List, Tuple

from deepsearcher.agent.base import RAGAgent
from deepsearcher.agent.collection_router import CollectionRouter
from deepsearcher.embedding.base import BaseEmbedding
from deepsearcher.llm.base import BaseLLM
from deepsearcher.tools import log
from deepsearcher.vector_db.base import BaseVectorDB, RetrievalResult, deduplicate_results

SUMMARY_PROMPT = """You are a AI content analysis expert, good at summarizing content. Please summarize a specific and detailed answer or report based on the previous queries and the retrieved document chunks.

Original Query: {query}

Related Chunks: 
{mini_chunk_str}
"""


class NaiveRAG(RAGAgent):
    def __init__(
        self,
        llm: BaseLLM,
        embedding_model: BaseEmbedding,
        vector_db: BaseVectorDB,
        top_k: int = 11,
        route_collection: bool = True,
        text_window_splitter: bool = True,
        **kwargs,
    ):
        self.llm = llm
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        self.top_k = top_k
        self.route_collection = route_collection
        if self.route_collection:
            self.collection_router = CollectionRouter(llm=self.llm, vector_db=self.vector_db)
        self.text_window_splitter = text_window_splitter

    def retrieve(self, query: str, collections_names: list = None, **kwargs) -> Tuple[List[RetrievalResult], int, dict]:
        consume_tokens = 0

        if self.route_collection:
            selected_collections, n_token_route = self.collection_router.invoke(query=query, collections_names=collections_names)
        elif collections_names is not None:
            selected_collections = collections_names
            n_token_route = 0
        else:
            selected_collections = self.collection_router.all_collections
            n_token_route = 0

        consume_tokens += n_token_route
        all_retrieved_results = []
        for collection in selected_collections:
            retrieval_res = self.vector_db.search_data(
                collection=collection,
                vector=self.embedding_model.embed_query(query),
                top_k=max(self.top_k // len(selected_collections), 1),
            )
            all_retrieved_results.extend(retrieval_res)
        all_retrieved_results = deduplicate_results(all_retrieved_results)
        return all_retrieved_results, consume_tokens, {}

    def query(self, query: str, **kwargs) -> Tuple[str, List[RetrievalResult], int]:
        all_retrieved_results, n_token_retrieval, _ = self.retrieve(query)
        chunk_texts = []
        for chunk in all_retrieved_results:
            if self.text_window_splitter and "wider_text" in chunk.metadata:
                chunk_texts.append(chunk.metadata["wider_text"])
            else:
                chunk_texts.append(chunk.text)
        mini_chunk_str = ""
        for i, chunk in enumerate(chunk_texts):
            mini_chunk_str += f"""<chunk_{i}>\n{chunk}\n</chunk_{i}>\n"""

        summary_prompt = SUMMARY_PROMPT.format(query=query, mini_chunk_str=mini_chunk_str)
        char_response = self.llm.chat([{"role": "user", "content": summary_prompt}])
        final_answer = char_response.content
        log.color_print("\n==== FINAL ANSWER====\n")
        log.color_print(final_answer)
        return final_answer, all_retrieved_results, n_token_retrieval + char_response.total_tokens
