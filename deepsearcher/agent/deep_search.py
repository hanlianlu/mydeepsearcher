import asyncio
import json
import time
from typing import List, Tuple, Dict, Any
from deepsearcher.contextmgmt.cachemanager import CacheManager
from deepsearcher.agent.base import RAGAgent, describe_class
from deepsearcher.agent.collection_router import CollectionRouter
from deepsearcher.embedding.base import BaseEmbedding
from deepsearcher.llm.base import BaseLLM
from deepsearcher.tools import log
from deepsearcher.vector_db import RetrievalResult
from deepsearcher.vector_db.base import BaseVectorDB, deduplicate_results
from deepsearcher import configuration
from deepsearcher.agent.final_paper import FinalPaperAgent

from deepsearcher.utils.rag_helpers import format_retrieved_results, parse_rerank_response, confidence_prompt

from deepsearcher.utils.rag_prompts import (
    SUB_QUERY_PROMPT, 
    RERANK_PROMPT,
    REFLECT_PROMPT,
    SUMMARY_PROMPT
)

@describe_class(
    "DeepSearch is the ultimate all-purpose agent, designed to handle the vast majority of your queries with unmatched depth and flexibility. "
    "It seamlessly breaks down even the most complex questions into manageable subqueries, ensuring a thorough and systematic exploration of every angle. "
    "Whether you need a detailed report, a panoramic survey, or a rich contextual overview, DeepSearch delivers cohesive, narrative-driven answers that leave no stone unturned. "
    "For most tasks, this agent is your reliable, go-to choice—excelling at both broad exploration and in-depth analysis. "
)
class DeepSearch(RAGAgent):
    def __init__(
        self,
        llm: BaseLLM,
        lightllm: BaseLLM,
        highllm: BaseLLM,
        embedding_model: BaseEmbedding,
        vector_db: BaseVectorDB,
        max_iter: int = 5,
        route_collection: bool = True,
        text_window_splitter: bool = True,
        confidence_threshold: float = 0.9,
        user_id: str = "default_user",
        **kwargs,
    ):
        self.llm = llm
        self.lightllm = lightllm
        self.highllm = highllm
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        self.max_iter = max_iter
        self.route_collection = route_collection
        self.confidence_threshold = confidence_threshold
        self.text_window_splitter = text_window_splitter
        self.final_answer_agent = FinalPaperAgent(lightllm=lightllm, highllm=highllm)
        self.semaphore = asyncio.Semaphore(10)
        self.user_id = user_id
        self.cache_manager = CacheManager()
        self.collection_router = CollectionRouter(llm=self.lightllm, vector_db=self.vector_db)
        self.collection_cache = {}  # Cache for collection routing results

    async def _generate_sub_queries(self, original_query: str, history_context: str = "") -> Tuple[List[str], int]:
        prompt = SUB_QUERY_PROMPT.format(original_query=original_query, history_context=history_context)
        chat_response = await self.llm.chat_async(messages=[{"role": "user", "content": prompt}])
        sub_queries = self.llm.literal_eval(chat_response.content)
        if not isinstance(sub_queries, list):
            sub_queries = []
        log.color_print(f"<think> Generated {len(sub_queries)} sub-queries: {sub_queries} </think>\n")
        return sub_queries, chat_response.total_tokens

    async def _search_in_collection(self, collection: str, query_vector) -> List[RetrievalResult]:
        async with self.semaphore:
            log.color_print(f"<think> Searching in collection [{collection}]... </think>\n")
            retrieved_results = await asyncio.to_thread(self.vector_db.search_data, collection=collection, vector=query_vector)
            for result in retrieved_results:
                if "source" not in result.metadata:
                    result.metadata["source"] = "local"
            return retrieved_results

    async def _search_web(self, sub_query: str, history_context: str="", **kwargs) -> List[RetrievalResult]:
        if not configuration.web_search_agent:
            log.error("Web search agent is not initialized!")
            return []
        web_results, _, _ = await configuration.web_search_agent.retrieve(query=sub_query , history_context= history_context)
        log.color_print(f"<think> Retrieved {len(web_results)} web results for [{sub_query}] </think>\n")
        return web_results

    async def _rerank_chunks(self, chunks: List[RetrievalResult], original_query: str, sub_queries: List[str]) -> List[Tuple[RetrievalResult, float]]:
        if not chunks:
            return []
        batch_size = min(9, len(chunks))
        accepted_results_with_scores = []
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            batch_formatted = format_retrieved_results(batch)
            prompt = RERANK_PROMPT.format(original_query=original_query, sub_queries=", ".join(sub_queries), retrieved_chunk=batch_formatted)
            chat_response = await self.llm.chat_async([{"role": "user", "content": prompt}])
            try:
                batch_responses = json.loads(chat_response.content)
                for result, response in zip(batch, batch_responses):
                    decision, score = response
                    score = float(score)
                    if decision.upper() == "YES" and score > 0.4:
                        accepted_results_with_scores.append((result, score))
            except (ValueError, json.JSONDecodeError) as e:
                log.warning(f"Invalid rerank response: {e}")
        return accepted_results_with_scores

    async def _retrieve_for_sub_query(self, query: str, sub_queries: List[str], curr_iter: int, collections_names: list, use_web_search: bool, original_query: str) -> List[RetrievalResult]:
        sub_query_key = self.cache_manager.generate_key(self.user_id, query, collections_names, curr_iter=curr_iter)
        cached_result = self.cache_manager.get(sub_query_key)
        if cached_result:
            log.color_print(f"<think> Using cached results for sub-query: {query} </think>\n")
            return cached_result["results"]

        query_vector = self.embedding_model.embed_query(query)
        tasks = [self._search_in_collection(collection, query_vector) for collection in collections_names]
        if use_web_search:
            tasks.append(self._search_web(query))
        results = await asyncio.gather(*tasks)
        all_retrieved_results = [result for sublist in results for result in sublist]
        accepted_results_with_scores = await self._rerank_chunks(all_retrieved_results, original_query, sub_queries)
        accepted_results_with_scores.sort(key=lambda x: x[1], reverse=True)
        accepted_results = [result for result, _ in accepted_results_with_scores]
        self.cache_manager.set(sub_query_key, {"results": accepted_results})
        return accepted_results

    async def _generate_gap_queries(self, original_query: str, all_sub_queries: List[str], all_chunks: List[RetrievalResult]) -> List[str]:
        chunk_str = format_retrieved_results(all_chunks) if all_chunks else "No chunks retrieved."
        prompt = REFLECT_PROMPT.format(question=original_query, mini_questions=", ".join(all_sub_queries), chunk_str=chunk_str)
        chat_response = await self.llm.chat_async([{"role": "user", "content": prompt}])
        gap_queries = self.llm.literal_eval(chat_response.content)
        if not isinstance(gap_queries, list):
            gap_queries = []
        log.color_print(f"<think> Generated {len(gap_queries)} gap queries: {gap_queries} </think>\n")
        return gap_queries

    async def _assess_confidence(self, original_query: str, all_sub_queries: List[str], all_chunks: List[RetrievalResult]) -> float:
        chunk_str = format_retrieved_results(all_chunks) if all_chunks else "No chunks retrieved."
        confidence_prompt = (
            "Based on the original query, previous sub-queries, and the retrieved chunks, assess your confidence (as a number between 0 and 1) that you have enough information to answer the query comprehensively with profound insights.\n\n"
            "Original Query: {original_query}\n"
            "Previous Sub Queries: {sub_queries}\n"
            "Retrieved Chunks: {chunk_str}\n\n"
            "Respond with a single number with double decimal precision between 0.00 and 1.00."
        ).format(original_query=original_query, sub_queries=", ".join(all_sub_queries), chunk_str=chunk_str)
        chat_response = await self.llm.chat_async([{"role": "user", "content": confidence_prompt}])
        try:
            return float(chat_response.content.strip())
        except ValueError:
            return 0.0

    async def async_retrieve(self, original_query: str, history_context: str = "", collections_names: list = None, use_web_search: bool = False, **kwargs) -> Tuple[List[RetrievalResult], int, Dict]:
        max_iter = kwargs.get("max_iter", self.max_iter)
        all_search_res = []
        all_sub_queries = []
        total_tokens = 0

        # Cache collection routing
        cache_key = self.cache_manager.generate_key(self.user_id, original_query, "collection_routing")
        if self.route_collection and collections_names is None:
            if cache_key in self.collection_cache:
                selected_collections = self.collection_cache[cache_key]
            else:
                selected_collections, n_token_route = await self.collection_router.invoke(original_query)
                total_tokens += n_token_route
                self.collection_cache[cache_key] = selected_collections
        else:
            selected_collections = collections_names or self.collection_router.all_collections

        if use_web_search:
            log.color_print(f"<think> Start initial query: {original_query} </think>\n")
            initial_results = await self._retrieve_for_sub_query(original_query, [], 0, selected_collections, use_web_search, original_query)
            all_search_res.extend(initial_results)
            confidence = await self._assess_confidence(original_query, [], all_search_res)
            if confidence >= 0.94:
                log.color_print(f"<think> Confidence {confidence:.2f} >= {self.confidence_threshold:.2f}, stopping early. </think>\n")
                return all_search_res, total_tokens, {"all_sub_queries": []}

        sub_queries, used_token = await self._generate_sub_queries(original_query, history_context)
        total_tokens += used_token
        if not sub_queries:
            return all_search_res, total_tokens, {"all_sub_queries": []}
        all_sub_queries.extend(sub_queries)

        for iter in range(max_iter):
            log.color_print(f"<think> Start Iteration: {iter + 1} </think>\n")
            tasks = [self._retrieve_for_sub_query(query, all_sub_queries, iter + 1, selected_collections, use_web_search, original_query) for query in sub_queries]
            results = await asyncio.gather(*tasks)
            for res in results:
                all_search_res.extend(res)
            all_search_res = deduplicate_results(all_search_res)
            log.color_print(f"<think> Start retrieval assessment: {all_sub_queries} </think>\n")

            confidence = await self._assess_confidence(original_query, all_sub_queries, all_search_res)
            if confidence >= self.confidence_threshold:
                log.color_print(f"<think> Confidence {confidence:.2f} >= {self.confidence_threshold:.2f}, stopping early. </think>\n")
                break
            if iter+1 >= max_iter:
                log.color_print(f"<think> Iteration limit reached </think>\n")
                break
            gap_queries = await self._generate_gap_queries(original_query, all_sub_queries, all_search_res)
            if not gap_queries:
                break
            all_sub_queries.extend(gap_queries)
            sub_queries = gap_queries

        return all_search_res, total_tokens, {"all_sub_queries": all_sub_queries}

    def retrieve(self, original_query: str, history_context: str = "", collections_names: list = None, use_web_search: bool = False, **kwargs) -> Tuple[List[RetrievalResult], int, Dict]:
        start_time = time.time()
        results = asyncio.run(self.async_retrieve(original_query, history_context, collections_names, use_web_search, **kwargs))
        log.color_print(f"<think> Retrieval completed in {time.time() - start_time:.2f} seconds </think>\n")
        return results

    async def async_query(self, query: str, history_context: str = "", collections_names: list = None, use_web_search: bool = False, **kwargs) -> Tuple[str, List[RetrievalResult], int]:
        query_key = self.cache_manager.generate_key(self.user_id, query, collections_names)
        cached_result = self.cache_manager.get(query_key)
        if cached_result:
            log.color_print(f"<think> Using cached response for query: {query} </think>\n")
            return cached_result["answer"], cached_result["results"], cached_result["tokens"]

        all_retrieved_results, n_token_retrieval, additional_info = await self.async_retrieve(query, history_context, collections_names, use_web_search, **kwargs)
        if not all_retrieved_results:
            disclaimer = "\n\n[Disclaimer: No relevant data found! Answer solely based on AI pretrained knowledge and may NOT be valid.]\n\n"
            summary_prompt = SUMMARY_PROMPT.format(question=query, mini_questions="[]", mini_chunk_str="", history_context=history_context)
            chat_response = await self.llm.chat_async([{"role": "user", "content": summary_prompt}])
            final_answer = disclaimer + chat_response.content + disclaimer
            result = (final_answer, [], n_token_retrieval + chat_response.total_tokens)
        else:
            judgment_prompt = f"Analyze the query and history context to determine if a long-form/thesis-level report is expected and necessary.\nQuery: {query}\nHistory Context: {history_context}\nRespond with 'YES' or 'NO'."
            use_final_answer_agent = (await self.llm.chat_async([{"role": "user", "content": judgment_prompt}])).content.strip().upper() == "YES"
            if use_final_answer_agent:
                final_answer = await self.final_answer_agent.generate_response(query=query, retrieved_results=all_retrieved_results, sub_queries=additional_info.get("all_sub_queries", []), history_context=history_context)
                result = (final_answer, all_retrieved_results, n_token_retrieval)
            else:
                summary_prompt = SUMMARY_PROMPT.format(
                    question=query,
                    mini_questions="; ".join(additional_info.get("all_sub_queries", [])),
                    mini_chunk_str=format_retrieved_results(all_retrieved_results),
                    history_context=history_context
                )
                chat_response = await self.highllm.chat_async([{"role": "user", "content": summary_prompt}])
                final_answer = chat_response.content
                result = (final_answer, all_retrieved_results, n_token_retrieval + chat_response.total_tokens)
        self.cache_manager.set(query_key, {"answer": final_answer, "results": all_retrieved_results, "tokens": result[2]})
        return result

    def query(self, query: str, history_context: str = "", collections_names: list = None, use_web_search: bool = False, **kwargs) -> Tuple[str, List[RetrievalResult], int]:
        return asyncio.run(self.async_query(query, history_context, collections_names, use_web_search, **kwargs))

