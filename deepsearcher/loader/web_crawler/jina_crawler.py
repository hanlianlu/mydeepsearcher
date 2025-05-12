import os
from typing import List

import requests
from langchain_core.documents import Document

from deepsearcher.loader.web_crawler.base import BaseCrawler


class JinaCrawler(BaseCrawler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.jina_api_token = os.getenv("JINA_API_TOKEN") or os.getenv("JINAAI_API_KEY")
        if not self.jina_api_token:
            raise ValueError("Missing JINA_API_TOKEN environment variable")

    def crawl_url(self, url: str) -> List[Document]:
        jina_url = f"https://r.jina.ai/{url}"
        headers = {
            "Authorization": f"Bearer {self.jina_api_token}",
            "X-Return-Format": "markdown",
        }

        response = requests.get(jina_url, headers=headers)
        response.raise_for_status()

        markdown_content = response.text
        metadata = {
            "reference": url,
            "status_code": response.status_code,
            "headers": dict(response.headers),
        }

        return [Document(page_content=markdown_content, metadata=metadata)]
