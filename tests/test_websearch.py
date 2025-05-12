"""
tests/test_websearch.py
Extended async integration test for DuckDuckGoSearchService + DsCrawler.

Diagnostics added:
* Separates PDF‑sourced Documents from HTML.
* Logs how many PDFs were fetched, how many chunks per PDF, rough token count,
  and previews of the first two chunks.
* Fails if a PDF URL is present but no text was extracted.
"""

from __future__ import annotations

import asyncio
import logging
import textwrap
from collections import defaultdict
from typing import Any, Sequence

from deepsearcher.configuration import Configuration, init_config
from deepsearcher import configuration  # populated by init_config

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


async def test_web_search_integration(
    query: str,
    search_service: Any,
    web_crawler: Any,
    count: int = 5,
    preview_chars: int = 250,
) -> bool:
    """
    1) Search URLs        (DuckDuckGoSearchService)
    2) Crawl & parse docs (DsCrawler, including PDF delegation)
    3) Show PDF diagnostics and validate parsed content.
    """
    if search_service is None or web_crawler is None:
        logger.error("Search service or web crawler is not initialized")
        return False

    # ---- Step 1: search -------------------------------------------------- #
    logger.info("Searching for URLs ...")
    try:
        urls: Sequence[str] = await search_service.search(query, count=count)
    except Exception as exc:
        logger.error("Search failed: %s", exc)
        return False

    if not urls:
        logger.warning("No URLs found for query: %r", query)
        return False
    logger.info("Found %d URL(s): %s", len(urls), urls)

    # ---- Step 2: crawl --------------------------------------------------- #
    logger.info("Crawling URLs ...")
    try:
        docs = await web_crawler.crawl_urls(list(urls), query=query)
    except Exception as exc:
        logger.error("Crawler raised an exception: %s", exc)
        return False

    if not docs:
        logger.warning("No content retrieved from URLs")
        return False
    logger.info("Retrieved %d document chunk(s) in total", len(docs))

    # ---- Step 3: classify & preview PDFs -------------------------------- #
    pdf_docs = [
        d for d in docs
        if d.metadata.get("type") == "pdf"
        or d.metadata.get("reference", "").lower().endswith(".pdf")
    ]
    html_docs = [d for d in docs if d not in pdf_docs]

    logger.info("%d chunks from PDF(s)", len(pdf_docs))
    logger.info("%d chunks from HTML pages", len(html_docs))

    pdf_urls = [u for u in urls if u.lower().endswith(".pdf")]
    if pdf_urls and not pdf_docs:
        logger.error("PDF URLs were fetched but no PDF text extracted")
        return False

    chunks_by_pdf: dict[str, list] = defaultdict(list)
    for d in pdf_docs:
        chunks_by_pdf[d.metadata.get("reference", "unknown.pdf")].append(d)

    for ref, chunk_list in chunks_by_pdf.items():
        logger.info("PDF: %s  (chunks: %d)", ref, len(chunk_list))
        for i, chunk in enumerate(chunk_list[:2], 1):
            preview = textwrap.shorten(
                chunk.page_content.replace("\n", " "),
                width=preview_chars,
                placeholder=" …",
            )
            token_estimate = int(len(chunk.page_content) / 4)  # rough
            page = chunk.metadata.get("page", "?")
            logger.info(
                "  chunk %d | page %s | ~%d tokens\n    %s",
                i,
                page,
                token_estimate,
                preview,
            )

    # Basic validation
    for d in docs:
        if not d.page_content.strip():
            logger.error("Empty page_content in a Document")
            return False
        if "reference" not in d.metadata:
            logger.error("Document missing 'reference' metadata")
            return False

    logger.info("All sanity checks passed")
    return True

# --------------------------------------------------------------------------- #
# CLI entry‑point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    cfg = Configuration()
    init_config(cfg)

    search_service = configuration.search_service
    web_crawler = configuration.web_crawler

    query = "stable diffusion denoising diffusion probabilistic models pdf"

    success = asyncio.run(
        test_web_search_integration(query, search_service, web_crawler)
    )

    if success:
        logger.info("Integration test succeeded")
    else:
        logger.error("Integration test FAILED")
