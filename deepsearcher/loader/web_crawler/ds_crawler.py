from __future__ import annotations

import asyncio
import logging
import re
import tempfile
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from deepsearcher import configuration  # Adjust based on your setup
from tenacity import retry, stop_after_attempt, wait_exponential
from trafilatura import extract, extract_metadata
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker import HierarchicalChunker
import ast

logger = logging.getLogger(__name__)

@dataclass(slots=True)
class LinkPolicy:
    follow_types: Set[str] = field(default_factory=lambda: {"pdf", "html"})
    same_domain_only: bool = False
    allow_domains: List[str] = field(default_factory=lambda: [
        "arxiv.org", "acm.org", "ieee.org", "wikipedia.org", "bcg.com",
        "pytorch.org", "mckinsey.com", "huggingface.co", "researchgate.net",
        "springer.com", "nature.com", "science.org", "nasa.gov", "nih.gov",
        "europepmc.org", "nist.gov", "data.gov", "github.com", "gitlab.com",
        "readthedocs.io", "europa.eu", "sec.gov"
    ])
    block_domains: List[str] = field(default_factory=lambda: [
        "facebook.com", "twitter.com", "instagram.com", "tiktok.com", "reddit.com",
        "linkedin.com", "pinterest.com", "yahoo.com", "bing.com",
    ])
    url_regex: Optional[str] = None
    per_page_limit: int = 10
    min_relevance: float = 0.7

    def _type_ok(self, url: str) -> bool:
        url_l = url.lower()
        return (
            ("pdf" in self.follow_types and url_l.endswith(".pdf"))
            or ("html" in self.follow_types and not url_l.endswith(".pdf"))
        )

    def _domain_ok(self, base_dom: str, link_dom: str) -> bool:
        link_dom = link_dom.lower()
        if self.same_domain_only and not link_dom.endswith(base_dom):
            return False
        if any(link_dom.endswith(blocked) for blocked in self.block_domains):
            return False
        if self.allow_domains and not any(link_dom.endswith(allowed) for allowed in self.allow_domains):
            return False
        return True

    def should_follow(self, base_domain: str, url: str) -> bool:
        if not self._type_ok(url):
            return False
        link_dom = urlparse(url).netloc.lower()
        if not self._domain_ok(base_domain, link_dom):
            return False
        if self.url_regex and not re.search(self.url_regex, url, flags=re.I):
            return False
        return True

class DsCrawler:
    def __init__(
        self,
        *,
        min_char_count: int = 160,
        max_concurrent_requests: int = 15,
        verify_ssl: bool = True,
        timeout: int = 16,
        process_pdfs: bool = True,
        max_depth: int = 2,
        max_pages: int = 30,
        link_policy: Optional[Dict] = None,
        query: str = "",
        use_relevance: bool = True,
        max_batch_size: int = 16,
        target_chunk_size: int = 6000,  # Maximum characters per chunk
    ):
        """Initialize the DsCrawler with configurable options."""
        self.min_char_count = min_char_count
        self.max_concurrent_requests = max_concurrent_requests
        self.verify_ssl = verify_ssl
        self.timeout = timeout
        self.process_pdfs = process_pdfs
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.policy = LinkPolicy(**link_policy) if link_policy else LinkPolicy()
        self.query = query
        self.use_relevance = use_relevance
        self.max_batch_size = max_batch_size
        self.target_chunk_size = target_chunk_size  # Store the target chunk size

        self.converter = DocumentConverter()
        self.chunker = HierarchicalChunker()  # Assuming no direct chunk size param

        self.trafilatura_options = {
            "include_tables": True,
            "include_comments": False,
            "favor_precision": True,
            "output_format": "markdown",
        }

        self.llm = configuration.nanollm if use_relevance else None

        self.pdf_loader = configuration.file_loader if process_pdfs else None
        if process_pdfs and (not self.pdf_loader or not hasattr(self.pdf_loader, "load_file")):
            logger.warning("PDF loader invalid or lacks 'load_file'; disabling PDF processing")
            self.process_pdfs = False
            self.pdf_loader = None

        self.relevance_cache: Dict[str, float] = {}
        self.cache_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    def combine_chunks_by_source(self, docs: List[Document], target_size: int) -> List[Document]:
        """Combine chunks from the same source up to target_size, keeping sources separate."""
        source_groups = defaultdict(list)
        
        # Group documents by their source (reference)
        for doc in docs:
            source = doc.metadata.get("reference", "")
            source_groups[source].append(doc)

        combined_docs = []
        for source, group in source_groups.items():
            current_text = ""
            current_metadata = group[0].metadata.copy()  # Base metadata from first chunk

            for doc in group:
                next_text = doc.page_content
                if len(current_text) + len(next_text) + 1 <= target_size:
                    # Combine if within target size
                    if current_text:
                        current_text += " " + next_text
                    else:
                        current_text = next_text
                else:
                    # Start a new chunk if exceeding target size
                    if current_text:
                        combined_docs.append(Document(page_content=current_text, metadata=current_metadata))
                    current_text = next_text if len(next_text) <= target_size else next_text[:target_size]

            # Append any remaining text
            if current_text:
                combined_docs.append(Document(page_content=current_text, metadata=current_metadata))

        return combined_docs

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _download(self, url: str, client: httpx.AsyncClient) -> Tuple[Optional[bytes], str]:
        try:
            r = await client.get(url, follow_redirects=True)
            r.raise_for_status()
            return r.content, r.headers.get("content-type", "")
        except httpx.HTTPStatusError as e:
            logger.error("HTTP error downloading %s: %s", url, e)
            return None, ""
        except httpx.RequestError as e:
            logger.error("Network error downloading %s: %s", url, e)
            return None, ""

    async def _assess_relevance(self, text: str) -> float:
        if not self.use_relevance or not self.llm:
            return 1.0
        if len(text) < self.min_char_count:
            return 0.0
        prompt = f"Given the Query '{self.query}', rate the relevance of this document chunk on a scale from 0 to 1: \n\n{text}\n\nReturn only the numeric score as a float."
        try:
            response = await self.llm.chat_async(messages=[{"role": "user", "content": prompt}])
            score = float(response.content.strip())
            return min(max(score, 0.0), 1.0)
        except Exception as e:
            logger.error(f"Failed to assess relevance for text: {e}")
            return 0.0

    async def _assess_relevance_batch(self, texts: List[str]) -> List[float]:
        if not self.use_relevance or not self.llm:
            return [1.0] * len(texts)
        if not texts:
            return []

        prompt = (
            f"Given the Query '{self.query}', rate the relevance of the following document chunks on a scale from 0 to 1. "
            f"Respond with a valid Python list of floats (e.g., [0.8, 0.3, ...]), containing exactly {len(texts)} scores, "
            f"and ensure the list is fully closed with a ']'. Do not include any text beyond the list.\n\n"
        )
        for i, text in enumerate(texts, 1):
            prompt += f"Chunk {i}: {text}\n\n"

        try:
            response = await self.llm.chat_async(messages=[{"role": "user", "content": prompt}])
            scores_str = response.content.strip()
            try:
                scores = ast.literal_eval(scores_str)
                if not isinstance(scores, list) or len(scores) != len(texts):
                    raise ValueError(f"Expected {len(texts)} scores, got {len(scores)} or invalid format")
                return [min(max(float(s), 0.0), 1.0) for s in scores]
            except (ValueError, SyntaxError) as e:
                logger.warning(f"ast.literal_eval failed: {e}. Attempting regex parsing. Response: {scores_str}")
                scores = re.findall(r'\d+\.\d+', scores_str)
                if len(scores) != len(texts):
                    raise ValueError(f"Regex extracted {len(scores)} scores, expected {len(texts)}")
                return [min(max(float(s), 0.0), 1.0) for s in scores]
        except Exception as e:
            logger.error(f"Batch relevance assessment failed: {e}. Falling back to individual scoring.")
            scores = []
            for text in texts:
                score = await self._assess_relevance(text)
                scores.append(score)
            return scores

    async def _process_batch(self, docs: List[Document]) -> List[Document]:
        if not self.use_relevance:
            return docs

        texts = [doc.page_content for doc in docs]
        scores = await self._assess_relevance_batch(texts)
        filtered_docs = []
        for doc, score in zip(docs, scores):
            if score >= self.policy.min_relevance:
                doc.metadata["relevance"] = score
                filtered_docs.append(doc)
        return filtered_docs

    async def _process_docs_in_batches(self, docs: List[Document]) -> List[Document]:
        if not self.use_relevance:
            return docs

        batches = [docs[i:i + self.max_batch_size] for i in range(0, len(docs), self.max_batch_size)]
        tasks = [self._process_batch(batch) for batch in batches]
        results = await asyncio.gather(*tasks)
        filtered_docs = []
        for result in results:
            filtered_docs.extend(result)
        return filtered_docs

    async def _crawl_one(
        self,
        url: str,
        client: httpx.AsyncClient,
        depth: int,
        visited: Set[str],
        pages_left: List[int],
        relevant_docs_found: List[int],
    ) -> List[Document]:
        if url in visited or pages_left[0] <= 0:
            return []
        visited.add(url)

        raw, ctype = await self._download(url, client)
        if not raw:
            return []

        pages_left[0] -= 1

        docs, links = await self._process_url(url, raw, ctype)

        if docs:
            relevant_docs_found[0] += len(docs)

        if depth > 0 and pages_left[0] > 0:
            tasks = [
                self._crawl_one(link["url"], client, depth - 1, visited, pages_left, relevant_docs_found)
                for link in links if link["score"] >= self.policy.min_relevance and link["url"] not in visited
            ]
            if tasks:
                child_lists = await asyncio.gather(*tasks, return_exceptions=True)
                for lst in child_lists:
                    if isinstance(lst, list):
                        docs.extend(lst)
        return docs

    async def _process_url(self, url: str, content: bytes, content_type: str) -> Tuple[List[Document], List[Dict]]:
        if "html" in content_type.lower():
            docs, links = await self._parse_html(url, content)
        elif self.process_pdfs and url.lower().endswith(".pdf") and self.pdf_loader:
            docs = await self._parse_pdf(url, content)
            links = []
        else:
            docs = await self._parse_with_docling(url, content, content_type)
            links = []

        if self.use_relevance and docs:
            docs = await self._process_docs_in_batches(docs)

        return docs, links

    async def _parse_html(self, url: str, html_b: bytes) -> Tuple[List[Document], List[Dict]]:
        try:
            main = extract(html_b, **self.trafilatura_options)
            if not main or len(main) < self.min_char_count:
                return [], []

            with tempfile.NamedTemporaryFile(delete=True, suffix=".md") as tmp:
                tmp.write(main.encode('utf-8'))
                tmp.flush()
                logger.debug(f"Using temporary file {tmp.name} for Markdown content from {url}")
                conversion_result = self.converter.convert(tmp.name)
                docling_document = conversion_result.document

            chunks = list(self.chunker.chunk(docling_document))
            documents = []
            for chunk in chunks:
                metadata = {"reference": url, "text": chunk.text}
                documents.append(Document(page_content=chunk.text, metadata=metadata))
            documents = self.combine_chunks_by_source(documents, target_size=self.target_chunk_size)
            links = await self._extract_links(url, html_b)
            return documents, links
        except Exception as e:
            logger.error("HTML parse failed %s: %s", url, e)
            return [], []

    async def _parse_pdf(self, url: str, pdf_b: bytes) -> List[Document]:
        if not self.pdf_loader:
            logger.warning("PDF loader not configured; skipping %s", url)
            return []
        try:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            tmp.write(pdf_b)
            tmp.flush()
            tmp.close()

            logger.info(f"Created temporary PDF file: {tmp.name} (size: {os.path.getsize(tmp.name)} bytes) for URL: {url}")
            docs = self.pdf_loader.load_file(tmp.name)
            os.unlink(tmp.name)

            if not docs:
                logger.warning(f"No documents extracted from PDF {url}")
                return []
            logger.info(f"Extracted {len(docs)} documents from PDF {url}")
            for d in docs:
                d.metadata.setdefault("reference", url)
                d.metadata["type"] = "pdf"
            docs = self.combine_chunks_by_source(docs, target_size=self.target_chunk_size)
            return docs
        except Exception as e:
            logger.error(f"PDF parse failed for {url}: {e}")
            return []

    async def _parse_with_docling(self, url: str, content: bytes, content_type: str) -> List[Document]:
        try:
            suffix = ".pdf" if "pdf" in content_type.lower() else ".html"
            with tempfile.NamedTemporaryFile(delete=True, suffix=suffix) as tmp:
                tmp.write(content)
                tmp.flush()
                conversion_result = self.converter.convert(tmp.name)
                docling_document = conversion_result.document
                chunks = list(self.chunker.chunk(docling_document))
                documents = []
                for chunk in chunks:
                    metadata = {"reference": url, "text": chunk.text}
                    documents.append(Document(page_content=chunk.text, metadata=metadata))
                documents = self.combine_chunks_by_source(documents, target_size=self.target_chunk_size)
                return documents
        except Exception as e:
            logger.error(f"Failed to process {url} with Docling: {e}")
            return []

    async def _extract_links(self, url: str, html_b: bytes) -> List[Dict]:
        soup = BeautifulSoup(html_b, "html.parser")
        base_dom = urlparse(url).netloc
        links = []
        for a in soup.find_all("a", href=True):
            href = urljoin(url, a["href"])
            if self.policy.should_follow(base_dom, href):
                anchor_text = a.get_text(strip=True)
                parent = a.find_parent("p")
                context_text = parent.get_text(strip=True) if parent else anchor_text
                links.append({"url": href, "score": 0.0, "context": context_text})
                if len(links) >= self.policy.per_page_limit:
                    break
        if links and self.use_relevance:
            contexts = [link["context"] for link in links]
            scores = await self._assess_relevance_batch(contexts)
            for link, score in zip(links, scores):
                link["score"] = score
        links.sort(key=lambda x: x["score"], reverse=True)
        return links

    async def crawl_urls(self, seed_urls: List[str], query: str) -> List[Document]:
        self.query = query
        limits = httpx.Limits(max_connections=self.max_concurrent_requests)
        async with httpx.AsyncClient(
            verify=self.verify_ssl,
            timeout=self.timeout,
            limits=limits,
            headers={"User-Agent": "DsCrawler/1.0"},
        ) as client:
            pages_left = [self.max_pages]
            visited: Set[str] = set()
            relevant_docs_found = [0]
            logger.info("Starting crawl with %d seed URLs", len(seed_urls))

            async def start(u: str):
                return await self._crawl_one(u, client, self.max_depth, visited, pages_left, relevant_docs_found)

            results = await asyncio.gather(*(start(u) for u in seed_urls), return_exceptions=True)

        docs: List[Document] = []
        for res in results:
            if isinstance(res, list):
                docs.extend(res)
            else:
                logger.error("Unhandled error in crawl: %s", res)
        logger.info("Crawl completed: %d documents found, %d pages visited", len(docs), len(visited))
        return docs

async def main():
    crawler = DsCrawler(
        max_depth=2,
        max_pages=25,
        query="machine learning advancements",
        link_policy={"min_relevance": 0.7},
        use_relevance=True,
        target_chunk_size=6000,  # Set to 6000 as requested
    )
    docs = await crawler.crawl_urls(["https://arxiv.org", "https://wikipedia.org"])
    print(f"Crawled {len(docs)} relevant documents")

if __name__ == "__main__":
    asyncio.run(main())