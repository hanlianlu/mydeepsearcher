import os
from typing import List, Optional

from firecrawl import FirecrawlApp
from langchain_core.documents import Document

from deepsearcher.loader.web_crawler.base import BaseCrawler


class FireCrawlCrawler(BaseCrawler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.app = None

    def crawl_url(
        self,
        url: str,
        max_depth: Optional[int] = None,
        limit: Optional[int] = None,
        allow_backward_links: Optional[bool] = None,
    ) -> List[Document]:
        """
        Dynamically crawls a URL using either scrape_url or crawl_url:

        - Uses scrape_url for single-page extraction if no params are provided.
        - Uses crawl_url to recursively gather pages when any param is provided.

        Args:
            url (str): The starting URL to crawl.
            max_depth (Optional[int]): Maximum depth for recursive crawling (default: 2).
            limit (Optional[int]): Maximum number of pages to crawl (default: 20).
            allow_backward_links (Optional[bool]): Allow crawling pages outside the URL's children (default: False).

        Returns:
            List[Document]: List of Document objects with page content and metadata.
        """

        # Lazy init
        self.app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))

        # if user just inputs a single url as param
        # scrape single page
        if max_depth is None and limit is None and allow_backward_links is None:
            scrape_result = self.app.scrape_url(url=url, params={"formats": ["markdown"]})
            markdown_content = scrape_result.get("markdown", "")
            metadata = scrape_result.get("metadata", {})
            metadata["reference"] = url
            return [Document(page_content=markdown_content, metadata=metadata)]

        # else, crawl multiple pages based on users' input params
        # set default values if not provided
        crawl_params = {
            "scrapeOptions": {"formats": ["markdown"]},
            "limit": limit if limit is not None else 20,
            "maxDepth": max_depth if max_depth is not None else 2,
            "allowBackwardLinks": allow_backward_links
            if allow_backward_links is not None
            else False,
        }

        crawl_status = self.app.crawl_url(url=url, params=crawl_params)
        data = crawl_status.get("data", [])

        documents = []
        for item in data:
            markdown_content = item.get("markdown", "")
            metadata = item.get("metadata", {})
            metadata["reference"] = metadata.get("url", url)
            documents.append(Document(page_content=markdown_content, metadata=metadata))

        return documents
