from abc import ABC, abstractmethod
from typing import List

from langchain_core.documents import Document


class BaseCrawler(ABC):
    def __init__(self, **kwargs):
        pass

    def crawl_url(self, url: str, **crawl_kwargs) -> List[Document]:
        pass
        # Return a list of Document objects which contain the markdown format information of the website
        # In the metadata, it's recommended to include the reference to the url.
        # e.g.
        # return [Document(page_content=..., metadata={"reference": "www.abc.com/page1.html"})]

    def crawl_urls(self, urls: List[str], **crawl_kwargs) -> List[Document]:
        """
        Crawl multiple URLs and return a list of Document objects.
        """
        documents = []
        for url in urls:
            documents.extend(self.crawl_url(url, **crawl_kwargs))
        return documents
    
    @abstractmethod
    async def crawl_urls(self, urls: List[str]) -> List[Document]:
        """Crawl the provided URLs and return a list of Document objects."""
        pass