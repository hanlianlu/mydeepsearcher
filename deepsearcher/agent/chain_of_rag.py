import asyncio
from typing import List, Tuple, Dict
from deepsearcher.agent.base import RAGAgent, describe_class
from deepsearcher.agent.collection_router import CollectionRouter
from deepsearcher.embedding.base import BaseEmbedding
from deepsearcher.llm.base import BaseLLM
from deepsearcher.tools import log
from deepsearcher.vector_db import RetrievalResult
from deepsearcher.vector_db.base import BaseVectorDB, deduplicate_results_with_scores
from deepsearcher import configuration
from deepsearcher.contextmgmt.cachemanager import CacheManager
import time
import json
import numpy as np

# Prompt Templates
FOLLOWUP_QUERY_PROMPT = """You are using a search tool to answer the main query by iteratively searching the database. Given the following intermediate queries and answers, identify what key information is still needed to fully answer the main query. Generate up to three follow-up questions that target these gaps. Each question should be distinct and aim to uncover a specific piece of information that has not been addressed yet. If fewer than three questions are needed, provide only those.

## Previous intermediate queries and answers
{intermediate_context}

## Main query to answer
{query}

Respond with a Python list of follow-up questions (e.g., ["question1", "question2"]). Do not explain yourself or output anything else.
"""

INTERMEDIATE_ANSWER_PROMPT = """Given the following documents, generate an appropriate answer for the query. DO NOT hallucinate any information, only use the provided documents to generate the answer. Respond "No relevant information found" if the documents do not contain useful information.

## Documents
{retrieved_documents}

## Query
{sub_query}

Respond with a concise answer only, do not explain yourself or output anything else.
"""

FINAL_ANSWER_PROMPT = """You are an AI content analysis expert. Based on the following retrieved documents, intermediate queries and answers, and history context, generate a final comprehensive answer for the main query. Your final answer must include two clearly demarcated sections:

Section 1: Factual Analysis
- Provide a structured synthesis of the answer strictly based on the retrieved documents and intermediate queries. Only use information that appears in the documents or was generated during the iterative query process. If no factual data was retrieved, clearly state: "No factual information available from retrieved documents."
- When referencing specific information, cite the source using the format [Document X] for retrieved documents or [Intermediate Answer Y] for intermediate results.

Section 2: AI Augmented Insights
- Provide additional context, analysis, or interpretation that augments the factual analysis. This section reflects your pretrained knowledge, professional judgement and may include broader insights or interpretations not directly derived from the retrieved documents.

## Retrieved Documents:
{retrieved_documents}

## Intermediate Queries and Answers:
{intermediate_context}

## Main Query:
{query}

## History Context:
{history_context}

Respond with your final answer in the following exact sections:

Factual Analysis:
[Your factual analysis here.]

AI Augmented Insights:
[Your additional insights here.]

Please produce your final answer within these sections **in Markdown format**, observing these rules:
1. **Headings**  
   - Use `#`, `##`, `###`, etc. to structure major sections.
2. **Bullet lists**  
   - Use `-` or `*` for unordered lists, and `1., 2., 3., …` for ordered lists.
3. **Tables**  
   - Any tabular data must be formatted as a pipe‑delimited Markdown table with a header row and separator line.
4. **Code snippets**  
   - Use triple backticks (```) for any code or example commands.
5. **Emphasis**  
   - Use `**bold**` or `*italic*` where appropriate.
6. **Links and images**  
   - If referencing URLs or images, use standard Markdown syntax: `[text](url)` or `![alt](url)`.
Do not include any extra commentary or text outside these three sections.
"""

REFLECTION_PROMPT = """Given the following intermediate queries and answers, estimate your confidence (as a number between 0 and 1) that you have enough information to answer the main query.

## Intermediate queries and answers
{intermediate_context}

## Main query
{query}

Respond with a single number with double decimal precision between 0.00 and 1.00 only.
"""

@describe_class(
    "ChainOfRAG is a specialized, precision-focused agent, ideal for queries that demand laser-like accuracy and multi-step reasoning. "
    "It dynamically adapts to gaps in information, crafting reactive subqueries to uncover hidden connections and deliver structured, citation-rich answers. "
    "While not the first choice for broad exploration, it shines in scenarios where precision, transparency, and factual rigor are paramount. "
    "It is NOT suitable for queries requiring critical thinking, comparative studies or long thesis/reports."
)
class ChainOfRAG(RAGAgent):
    """Chain of Retrieval-Augmented Generation (RAG) agent with ranking mechanism for improved precision."""

    def __init__(
        self,
        llm: BaseLLM,
        lightllm: BaseLLM,
        highllm: BaseLLM,
        embedding_model: BaseEmbedding,
        vector_db: BaseVectorDB,
        max_iter: int = 5,
        early_stopping: bool = True,
        route_collection: bool = True,
        text_window_splitter: bool = True,
        confidence_threshold: float = 0.91,
        metadata_keys: List[str] = ["wider_text", "text"],
        user_id: str = "default_user",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.llm = llm
        self.lightllm = lightllm
        self.highllm = highllm
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.route_collection = route_collection
        self.confidence_threshold = confidence_threshold
        self.text_window_splitter = text_window_splitter
        self.metadata_keys = metadata_keys
        self.user_id = user_id
        self.cache_manager = CacheManager()
        self.semaphore = asyncio.Semaphore(10)
        self.collection_router = CollectionRouter(
            llm=self.lightllm, vector_db=self.vector_db, dim=embedding_model.dimension
        )
        self.collection_cache = {}  # Cache for collection routing results

    async def _reflect_get_subqueries(self, query: str, intermediate_context: List[str]) -> Tuple[List[str], int]:
        """Generate follow-up subqueries using light LLM."""
        prompt = FOLLOWUP_QUERY_PROMPT.format(query=query, intermediate_context="\n".join(intermediate_context))
        chat_response = await self.llm.chat_async([{"role": "user", "content": prompt}])
        subqueries = self.llm.literal_eval(chat_response.content)
        if not isinstance(subqueries, list):
            subqueries = []
        return subqueries[:3], chat_response.total_tokens

    async def _search_collection(self, collection: str, query_vector) -> List[Tuple[RetrievalResult, float]]:
        """Search a collection with controlled parallelism."""
        async with self.semaphore:
            log.color_print(f"<search> Searching in collection [{collection}]... </search>\n")
            retrieved_results = await asyncio.to_thread(self.vector_db.search_data, collection=collection, vector=query_vector)
            return [(result, getattr(result, 'score', 0.5)) for result in retrieved_results]

    async def _search_web(self, query: str, history_context="", **kwargs) -> Tuple[List[Tuple[RetrievalResult, float]], int]:
        """Search the web and assign relevance scores."""
        if not configuration.web_search_agent:
            log.error("Web search agent is not initialized!")
            return [], 0
        web_results, tokens, _ = await configuration.web_search_agent.retrieve(query=query, history_context= history_context, **kwargs)
        log.color_print(f"<think> Retrieved {len(web_results)} web results for [{query}] </think>\n")
        return [(result, getattr(result, 'score', 0.5)) for result in web_results], tokens

    async def _retrieve_and_answer(self, query: str, collections_names: list = None, curr_iter: int = 1, use_web_search: bool = False) -> Tuple[str, List[Tuple[RetrievalResult, float]], int]:
        """Retrieve documents, compute similarity scores, and generate an answer with caching."""
        sub_query_key = self.cache_manager.generate_key(self.user_id, query, collections_names, curr_iter=curr_iter)
        cached_result = self.cache_manager.get(sub_query_key)
        if cached_result:
            log.color_print(f"<think> Using cached results for sub-query: {query} </think>\n")
            return cached_result["answer"], cached_result["results"], cached_result["tokens"]

        token_usage = 0
        all_retrieved_results_with_scores = []
        selected_collections = collections_names or self.collection_router.all_collections

        query_vector = self.embedding_model.embed_query(query)
        tasks = [self._search_collection(collection, query_vector) for collection in selected_collections]
        local_results = await asyncio.gather(*tasks, return_exceptions=True)
        local_retrieved_results_with_scores = [result for sublist in local_results if isinstance(sublist, list) for result in sublist]
        log.color_print(f"<think> Retrieved {len(local_retrieved_results_with_scores)} local results for [{query}] </think>\n")

        if use_web_search or (not local_retrieved_results_with_scores and curr_iter < self.max_iter):
            web_retrieved_results_with_scores, web_tokens = await self._search_web(query)
            token_usage += web_tokens
            all_retrieved_results_with_scores.extend(web_retrieved_results_with_scores)
        all_retrieved_results_with_scores.extend(local_retrieved_results_with_scores)
        all_retrieved_results = [result for result, _ in all_retrieved_results_with_scores]

        # Generate intermediate answer
        prompt = INTERMEDIATE_ANSWER_PROMPT.format(
            retrieved_documents=self._format_retrieved_results(all_retrieved_results),
            sub_query=query
        ) if all_retrieved_results else f"No documents retrieved for query: {query}. Provide a general response based on the query alone."
        
        chat_response = await self.llm.chat_async([{"role": "user", "content": prompt}])
        token_usage += chat_response.total_tokens
        intermediate_answer = chat_response.content

        # Compute similarity-based scores
        if intermediate_answer.strip() != "No relevant information found" and all_retrieved_results:
            answer_vector = self.embedding_model.embed_query(intermediate_answer)
            supported_results_with_scores = []
            for result in all_retrieved_results:
                if hasattr(result, 'vector') and isinstance(result.vector, np.ndarray):
                    similarity = np.dot(answer_vector, result.vector)
                    combined_score = (result.score + similarity) / 2
                else:
                    combined_score = result.score
                supported_results_with_scores.append((result, combined_score))
        else:
            supported_results_with_scores = []

        cache_data = {
            "answer": intermediate_answer,
            "results": supported_results_with_scores,
            "tokens": token_usage
        }
        self.cache_manager.set(sub_query_key, cache_data)
        return intermediate_answer, supported_results_with_scores, token_usage

    async def _check_has_enough_info(self, query: str, intermediate_contexts: List[str]) -> Tuple[float, int]:
        """Assess confidence using light LLM."""
        prompt = REFLECTION_PROMPT.format(query=query, intermediate_context="\n".join(intermediate_contexts))
        chat_response = await self.llm.chat_async([{"role": "user", "content": prompt}])
        try:
            confidence = float(chat_response.content.strip())
            confidence = min(max(confidence, 0.0), 1.0)
        except Exception:
            confidence = 0.0
        return confidence, chat_response.total_tokens

    async def async_retrieve(self, query: str, history_context: str = "", collections_names: list = None, use_web_search: bool = False, **kwargs) -> Tuple[List[RetrievalResult], int, Dict]:
        """Perform iterative retrieval with confidence-based early stopping."""
        max_iter = kwargs.get("max_iter", self.max_iter)
        intermediate_contexts = []
        all_retrieved_results_with_scores = []
        token_usage = 0
        previous_subqueries = set()

        # Cache collection routing
        cache_key = self.cache_manager.generate_key(self.user_id, query, "collection_routing")
        if self.route_collection and collections_names is None:
            cached_collections = self.collection_cache.get(cache_key)
            if cached_collections:
                selected_collections = cached_collections
            else:
                selected_collections, n_token_route = await self.collection_router.invoke(query=query, curr_iter=0)
                token_usage += n_token_route
                self.collection_cache[cache_key] = selected_collections
        else:
            selected_collections = collections_names or self.collection_router.all_collections

        if use_web_search:
            log.color_print("<think> Performing initial search with original query... </think>\n")
            initial_answer, initial_results_with_scores, initial_tokens = await self._retrieve_and_answer(query, selected_collections, curr_iter=0, use_web_search=use_web_search)
            all_retrieved_results_with_scores.extend(initial_results_with_scores)
            token_usage += initial_tokens
            intermediate_contexts.append(f"Initial query: {query}\nInitial answer: {initial_answer}")

        for iter in range(max_iter):
            log.color_print(f"<think> Iteration: {iter + 1} </think>\n")
            subqueries, n_token0 = await self._reflect_get_subqueries(query, intermediate_contexts)
            token_usage += n_token0

            new_subqueries = [sq for sq in subqueries if sq not in previous_subqueries]
            if not new_subqueries:
                log.color_print("<think> No new subqueries generated. Stopping iteration. </think>\n")
                break
            previous_subqueries.update(new_subqueries)

            tasks = [self._retrieve_and_answer(sq, selected_collections, iter + 1, use_web_search) for sq in new_subqueries]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, tuple):
                    intermediate_answer, retrieved_results_with_scores, n_token1 = result
                    sq = new_subqueries[results.index(result)]
                    all_retrieved_results_with_scores.extend(retrieved_results_with_scores)
                    intermediate_idx = len(intermediate_contexts) + 1
                    intermediate_contexts.append(f"Intermediate query{intermediate_idx}: {sq}\nIntermediate answer{intermediate_idx}: {intermediate_answer}")
                    token_usage += n_token1

            # Deduplicate once per iteration
            all_retrieved_results_with_scores = deduplicate_results_with_scores(all_retrieved_results_with_scores)

            if self.early_stopping:
                confidence, n_token_check = await self._check_has_enough_info(query, intermediate_contexts)
                token_usage += n_token_check
                if confidence >= self.confidence_threshold:
                    log.color_print(f"<think> CoR Early stopping at iteration {iter + 1}: Confidence {confidence:.2f} >= {self.confidence_threshold:.2f} </think>\n")
                    break

        all_retrieved_results_with_scores.sort(key=lambda x: x[1], reverse=True)
        all_retrieved_results_with_scores = all_retrieved_results_with_scores[:20]  # Select top 20
        all_retrieved_results = [result for result, _ in all_retrieved_results_with_scores]
        return all_retrieved_results, token_usage, {"intermediate_context": intermediate_contexts}

    def retrieve(self, query: str, history_context: str = "", collections_names: list = None, use_web_search: bool = False, **kwargs) -> Tuple[List[RetrievalResult], int, Dict]:
        """Synchronous retrieval with timing."""
        start_time = time.time()
        results = asyncio.run(self.async_retrieve(query, history_context, collections_names, use_web_search, **kwargs))
        log.color_print(f"<think> Retrieval completed in {time.time() - start_time:.2f} seconds </think>\n")
        return results

    async def async_query(self, query: str, history_context: str = "", collections_names: list = None, use_web_search: bool = False, **kwargs) -> Tuple[str, List[RetrievalResult], int]:
        """Asynchronous query with modular caching."""
        start_time = time.time()
        query_key = self.cache_manager.generate_key(self.user_id, query, collections_names)
        cached_result = self.cache_manager.get(query_key)
        if cached_result:
            log.color_print(f"<think> Using cached response for query: {query} </think>\n")
            return cached_result["answer"], cached_result["results"], cached_result["tokens"]

        results, n_token_retrieval, additional_info = await self.async_retrieve(query, history_context, collections_names, use_web_search, **kwargs)
        intermediate_context = additional_info["intermediate_context"]

        prompt = FINAL_ANSWER_PROMPT.format(
            retrieved_documents=self._format_retrieved_results(results),
            intermediate_context="\n".join(intermediate_context),
            query=query,
            history_context=history_context
        )
        chat_response = await self.highllm.chat_async([{"role": "user", "content": prompt}])
        final_answer = chat_response.content
        log.color_print("\n==== FINAL ANSWER ====\n")
        log.color_print(final_answer)

        total_tokens = n_token_retrieval + chat_response.total_tokens
        cache_data = {"answer": final_answer, "results": results, "tokens": total_tokens}
        self.cache_manager.set(query_key, cache_data)
        log.color_print(f"<think> Query completed in {time.time() - start_time:.2f} seconds </think>\n")
        return final_answer, results, total_tokens

    def query(self, query: str, history_context: str = "", collections_names: list = None, use_web_search: bool = False, **kwargs) -> Tuple[str, List[RetrievalResult], int]:
        """Synchronous query with error handling."""
        start_time = time.time()
        try:
            results = asyncio.run(self.async_query(query, history_context, collections_names, use_web_search, **kwargs))
            log.color_print(f"<think> Query completed in {time.time() - start_time:.2f} seconds </think>\n")
            return results
        except Exception as e:
            log.error(f"Query failed: {e}")
            return "Error processing query.", [], 0

    def _format_retrieved_results(self, retrieved_results: List[RetrievalResult]) -> str:
        formatted_documents = []
        for i, result in enumerate(retrieved_results):
            if self.text_window_splitter and "wider_text" in result.metadata:
                text = result.metadata["wider_text"]
            else:
                text = result.text
            metadata_str = json.dumps(result.metadata)
            reference_str = json.dumps(result.reference)
            formatted_documents.append(
                f"<Document {i}>\n"
                f"Text: {text}\n"
                f"Reference: {reference_str}\n"
                f"Metadata: {metadata_str}\n"
                f"</Document {i}>"
            )
        return "\n".join(formatted_documents)