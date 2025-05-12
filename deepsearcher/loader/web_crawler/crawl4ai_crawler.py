import asyncio
from typing import List

from langchain_core.documents import Document

from deepsearcher.loader.web_crawler.base import BaseCrawler
from deepsearcher.tools import log


class Crawl4AICrawler(BaseCrawler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.crawler = None  # Lazy init
        self.browser_config = kwargs.get("browser_config", None)

    def _lazy_init(self):
        from crawl4ai import AsyncWebCrawler, BrowserConfig

        if self.crawler is None:
            config = BrowserConfig.from_kwargs(self.browser_config) if self.browser_config else None
            self.crawler = AsyncWebCrawler(config=config)

    async def _async_crawl(self, url: str) -> Document:
        if self.crawler is None:
            self._lazy_init()

        async with self.crawler as crawler:
            result = await crawler.arun(url)

            markdown_content = result.markdown or ""

            metadata = {
                "reference": url,
                "success": result.success,
                "status_code": result.status_code,
                "media": result.media,
                "links": result.links,
            }

            if hasattr(result, "metadata") and result.metadata:
                metadata["title"] = result.metadata.get("title", "")
                metadata["author"] = result.metadata.get("author", "")

            return Document(page_content=markdown_content, metadata=metadata)

    def crawl_url(self, url: str) -> List[Document]:
        try:
            document = asyncio.run(self._async_crawl(url))
            return [document]
        except Exception as e:
            log.error(f"Error during crawling {url}: {e}")
            return []

    async def _async_crawl_many(self, urls: List[str]) -> List[Document]:
        if self.crawler is None:
            self._lazy_init()
        async with self.crawler as crawler:
            results = await crawler.arun_many(urls)
            documents = []
            for result in results:
                markdown_content = result.markdown or ""
                metadata = {
                    "reference": result.url,
                    "success": result.success,
                    "status_code": result.status_code,
                    "media": result.media,
                    "links": result.links,
                }
                if hasattr(result, "metadata") and result.metadata:
                    metadata["title"] = result.metadata.get("title", "")
                    metadata["author"] = result.metadata.get("author", "")
                documents.append(Document(page_content=markdown_content, metadata=metadata))
            return documents

    def crawl_urls(self, urls: List[str], **crawl_kwargs) -> List[Document]:
        try:
            return asyncio.run(self._async_crawl_many(urls))
        except Exception as e:
            log.error(f"Error during crawling {urls}: {e}")
            return []
