from typing import List, Tuple, Optional
from deepsearcher.agent.base import RAGAgent
from deepsearcher.vector_db import RetrievalResult
from deepsearcher.llm.base import BaseLLM
from deepsearcher.tools import log
from deepsearcher.loader.web_crawler.ds_crawler import DsCrawler
from deepsearcher.webservice.ddgsearchservice import DuckDuckGoSearchService

import asyncio
from urllib.parse import urlparse
import ast

def get_base_domain(url: str) -> str:
    """Extract the base domain from a URL (e.g., 'example.com' from 'https://www.example.com/path')."""
    parsed = urlparse(url)
    domain = parsed.netloc
    if domain.startswith("www."):
        domain = domain[4:]
    return domain

class WebSearchAgent(RAGAgent):
    def __init__(
        self,
        search_service: DuckDuckGoSearchService,
        web_crawler: DsCrawler,
        llm: BaseLLM,
        max_urls: int = 10,
        max_chunk_size: int = 6000,
        url_relevance_threshold: float = 0.71,
        max_documents: int = 24,  # Increased from 16 for more variety
        **kwargs
    ):
        """
        Initialize the WebSearchAgent without additional scoring, relying on crawler's relevance.

        Args:
            search_service: Service to fetch URLs (expected to have an async search method).
            web_crawler: Crawler to retrieve content from URLs (expected to have an async crawl_urls method).
            llm (BaseLLM): LLM for query breakdown (expected to have chat_async method).
            max_urls (int): Maximum number of URLs to process (default: 10).
            max_chunk_size (int): Maximum characters per chunk (default: 6000).
            url_relevance_threshold (float): Minimum relevance score for documents (default: 0.71).
            max_documents (int): Maximum number of documents to return (default: 20).
            **kwargs: Additional arguments for RAGAgent.
        """
        super().__init__(**kwargs)
        self.search_service = search_service
        self.web_crawler = web_crawler
        self.llm = llm
        self.max_urls = max_urls
        self.max_chunk_size = max_chunk_size
        self.url_relevance_threshold = url_relevance_threshold
        self.max_documents = max_documents

    async def breakdown_query(self, original_query: str, history_context: str = "") -> Tuple[List[str], int]:
        """
        Break down a complex query into up to four sub-queries optimized for web search.

        Args:
            original_query (str): The main query to decompose.
            history_context (str): Previous conversation context (default: "").

        Returns:
            Tuple[List[str], int]: List of sub-queries and token usage.
        """
        prompt = f"""
        You are an expert at optimizing queries for web search. Given the main query and conversation history, decompose the main query into up to six complementary sub-queries. Each sub-query should:
        - Be concise and specific (e.g., suitable for a search engine like Google or DuckDuckGo).
        - Target a distinct aspect or component of the main query to maximize coverage of relevant information.
        - Avoid unnecessary complexity or long phrases that web search engines struggle with.
        - Incorporate context from the conversation history where relevant yet avoid redundancy.

        Main Query: {original_query}
        Conversation History: {history_context}

        Respond with a Python list of strings, e.g., ["sub-query 1", "sub-query 2", ...]. If the query cannot be broken down effectively, return a list with just the original query.
        """
        try:
            chat_response = await self.llm.chat_async(messages=[{"role": "user", "content": prompt}])
            sub_queries = ast.literal_eval(chat_response.content.strip())
            if not isinstance(sub_queries, list) or not sub_queries:
                sub_queries = [original_query]
            log.color_print(f"<think> Generated {len(sub_queries)} sub-queries: {sub_queries} </think>\n")
            return sub_queries, chat_response.total_tokens if hasattr(chat_response, 'total_tokens') else 0
        except Exception as e:
            log.error(f"Failed to break down query: {e}")
            return [original_query], 0

    async def retrieve(self, query: str, history_context: str = "", **kwargs) -> Tuple[List[RetrievalResult], int, dict]:
        """
        Retrieve web content for the query with sub-query breakdown, using crawler's relevance scores.

        Args:
            query (str): The search query.
            history_context (str): Previous conversation context (default: "").

        Returns:
            Tuple[List[RetrievalResult], int, dict]: Retrieved results, token usage, and metadata.
        """
        total_tokens = 0
        try:
            log.info(f"Performing web search for query: {query}")

            # Break down the query into sub-queries
            sub_queries, breakdown_tokens = await self.breakdown_query(query, history_context)
            total_tokens += breakdown_tokens
            search_queries = sub_queries

            # Collect URLs from all sub-queries
            all_urls = set()
            for search_query in search_queries:
                await asyncio.sleep(1)  # Basic rate limiting
                if asyncio.iscoroutinefunction(self.search_service.search):
                    urls = await self.search_service.search(search_query, count=self.max_urls)
                else:
                    urls = await asyncio.to_thread(self.search_service.search, search_query, count=self.max_urls)
                all_urls.update(urls)

            if not all_urls:
                log.warning(f"No URLs found for query: '{query}'")
                return [], total_tokens, {"status": "no_urls"}

            log.info(f"Found {len(all_urls)} unique URLs, crawling content")

            # Crawl URLs with relevance filtering enabled in the crawler
            web_docs = await self.web_crawler.crawl_urls(list(all_urls), query=query)
            if not web_docs:
                log.warning(f"No content retrieved for query: '{query}'")
                return [], total_tokens, {"status": "no_content"}

            # Filter documents based on relevance threshold using crawler's scores
            relevant_docs = [
                (doc, doc.metadata.get("relevance", 0.0))
                for doc in web_docs
                if doc.metadata.get("relevance", 0.0) >= self.url_relevance_threshold
            ]

            # Sort by relevance score in descending order and limit to max_documents
            relevant_docs.sort(key=lambda x: x[1], reverse=True)
            final_docs = relevant_docs[:self.max_documents]

            # Construct retrieval results with enhanced metadata
            retrieval_results = []
            for doc, score in final_docs:
                full_url = doc.metadata.get("reference", "Unknown URL")
                base_domain = get_base_domain(full_url)
                metadata = {
                    "source": "web",
                    "url": full_url,  # Full URL for precise referencing
                    "title": doc.metadata.get("title", "No Title"),  # Fallback if not provided
                    "description": doc.metadata.get("description", "No Description"),  # Fallback
                    "relevance": score,
                    **doc.metadata  # Include all original metadata
                }
                result = RetrievalResult(
                    text=doc.page_content[:self.max_chunk_size] + ("..." if len(doc.page_content) > self.max_chunk_size else ""),
                    reference=full_url,  # Use full URL instead of base domain
                    metadata=metadata,
                    score=score,
                    embedding=None
                )
                retrieval_results.append(result)

            return retrieval_results, total_tokens, {"status": "success"}

        except Exception as e:
            log.error(f"Web search retrieval failed: {e}")
            return [], total_tokens, {"status": "error", "message": str(e)}

    async def query(self, query: str, history_context: str = "", collections_names: list = None, use_web_search: bool = True, **kwargs) -> Tuple[List[RetrievalResult], int, dict]:
        """Retrieve web content without generating an answer."""
        if not use_web_search:
            return [], 0, {"status": "web_search_disabled"}
        return await self.retrieve(query, history_context)

    async def invoke(self, query: str, **kwargs) -> List[RetrievalResult]:
        """Invoke the agent and return retrieval results."""
        results, _, _ = await self.query(query, **kwargs)
        return results