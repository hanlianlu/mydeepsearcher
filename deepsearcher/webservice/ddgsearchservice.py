"""
duckduckgo_search_service.py
A modular, extensible search component for retrieving high‑quality URLs
with the duckduckgo-search package, suitable for RAG pipelines.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence, Set
from urllib.parse import urlparse

from duckduckgo_search import DDGS
from tenacity import retry, stop_after_attempt, wait_exponential

# Adjust this import path to match your project layout
from deepsearcher.webservice.base import BaseSearchService  # noqa: E402

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class SearchConfig:
    """Configuration parameters for DuckDuckGoSearchService."""
    max_results: int = 15
    region: str = "wt-wt"
    safesearch: str = "moderate"

    # Domains that always float to the top
    priority_domains: Set[str] = field(
        default_factory=lambda: {
       "arxiv.org", "acm.org", "ieee.org", "wikipedia.org", "bcg.com",
        "pytorch.org", "mckinsey.com", "huggingface.co", "researchgate.net",
        "springer.com", "nature.com", "science.org", "nasa.gov", "nih.gov",
        "europepmc.org", "nist.gov", "data.gov", "github.com", "gitlab.com",
        "readthedocs.io", "europa.eu", "sec.gov"
        }
    )

    # Domain keywords that will be discarded
    blacklist_terms: Set[str] = field(
        default_factory=lambda: {
            "Advameg", 
            "bestgore.com", 
            "Breitbart_News", 
            "Centre_for_Research_on_Globalization", 
            "Examiner.com", 
            "Famous_Birthdays", 
            "Healthline", 
            "InfoWars", 
            "Lenta.ru", 
            "LiveLeak", 
            "Lulu.com",
            "MyLife", 
            "Natural_News", 
            "OpIndia", 
            "The_Points_Guy", 
            "The_Points_Guy_(sponsored_content)", 
            "Swarajya", 
            "Veterans_Today", 
            "ZoomInfo",
        }
    )

    # Legitimate short domains allowed through the short‑domain check
    short_domain_whitelist: Set[str] = field(
        default_factory=lambda: {
            "docs.aws.amazon.com",
            "techcommunity.microsoft.com",
            "alibabacloud.com",
            "msn.com",
            "ibm.com",
            "sec.gov",
        }
    )

    # Maximum length for main domain part to be considered "short"
    short_domain_max_length: int = 4


# --------------------------------------------------------------------------- #
# Quality filtering helpers
# --------------------------------------------------------------------------- #
class QualityFilter:
    """Contains URL/title filtering heuristics."""

    _priority_regex: re.Pattern = re.compile(
        r"\.(edu|gov|govt|mil|ac)\.[a-z]{2,3}$", re.IGNORECASE
    )

    def __init__(self, cfg: SearchConfig) -> None:
        self.cfg = cfg

    # ------------- public API ------------ #

    def filter(self, raw_results: Sequence[dict]) -> List[str]:
        """Return a list of clean URLs in order of quality."""
        accepted: list[str] = []
        for res in raw_results:
            url = res.get("href") or ""
            title = res.get("title") or ""
            if not url:
                continue
            domain = self._extract_domain(url)

            if (
                self._is_blacklisted(url, domain)
                or self._is_suspicious_domain(domain)
                or not self._has_descriptive_title(title)
            ):
                continue

            accepted.append(url)

        # prioritise authoritative domains
        preferred = [
            u for u in accepted if self._is_preferred_domain(self._extract_domain(u))
        ]
        remainder = [u for u in accepted if u not in preferred]
        return preferred + remainder

    # ------------- individual checks ------------ #

    def _extract_domain(self, url: str) -> str:
        try:
            return urlparse(url).netloc.lower()
        except Exception:  # noqa: BLE001
            return url.lower()

    def _is_blacklisted(self, url: str, domain: str) -> bool:
        words = self.cfg.blacklist_terms
        return any(term in url.lower() or term in domain for term in words)

    def _is_suspicious_domain(self, domain: str) -> bool:
        # numeric IP
        if domain.replace(".", "").isdigit():
            return True
        # strip www.
        dom = domain[4:] if domain.startswith("www.") else domain
        # academic/government pass straight through
        if self._priority_regex.search(dom):
            return False
        # very short domains that are not whitelisted
        main = dom.split(".")[0]
        if (
            len(main) <= self.cfg.short_domain_max_length
            and dom not in self.cfg.short_domain_whitelist
        ):
            return True
        return False

    def _has_descriptive_title(self, title: str) -> bool:
        title = title.strip()
        if len(title) < 3:
            return False
        if "." in title and " " not in title:
            return False
        undescriptive = {"404", "403", "page not found", "forbidden", "access denied"}
        return not any(phrase in title.lower() for phrase in undescriptive)

    def _is_preferred_domain(self, domain: str) -> bool:
        if (
            domain in self.cfg.priority_domains
            or any(domain.endswith("." + d) for d in self.cfg.priority_domains)
        ):
            return True
        # academic/government pattern
        return bool(self._priority_regex.search(domain))


# --------------------------------------------------------------------------- #
# Search service
# --------------------------------------------------------------------------- #
class DuckDuckGoSearchService(BaseSearchService):  # type: ignore[misc]
    """High-level async search wrapper suitable for RAG pipelines."""

    def __init__(
        self,
        cfg: SearchConfig | None = None,
        quality_filter: QualityFilter | None = None,
        *,
        searcher: DDGS | None = None,
        logger_: logging.Logger | None = None,
    ) -> None:
        """
        Args:
            cfg: SearchConfig instance. If None, default config is used.
            quality_filter: QualityFilter instance; defaults to one built from cfg.
            searcher: A pre‑configured DDGS searcher (optional).
            logger_: Inject your own logger for testability.
        """
        super().__init__()
        self.cfg = cfg or SearchConfig()
        self.filter = quality_filter or QualityFilter(self.cfg)
        self.searcher = searcher or DDGS()
        self.logger = logger_ or logging.getLogger(__name__)

    # ----------------  public API  ---------------- #

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def search(self, query: str, count: int = 10) -> List[str]:
        """
        Fetch up to `count` high‑quality URLs for *query*.

        The method is fully asynchronous and safe to call from
        any asyncio event loop.
        """
        self.logger.debug("Searching DuckDuckGo for '%s' (count=%d)", query, count)
        try:
            raw: Iterable[dict] = await asyncio.to_thread(
                self.searcher.text,
                query,
                region=self.cfg.region,
                safesearch=self.cfg.safesearch,
                max_results=min(count, self.cfg.max_results),
            )
        except Exception as exc:  # noqa: BLE001
            self.logger.error("DuckDuckGo search failed: %s", exc)
            return []

        urls = self.filter.filter(raw)
        self.logger.info("DuckDuckGo found %d clean URL(s) for '%s'", len(urls), query)
        return urls[:count]  # guarantee caller never sees more than asked


# --------------------------------------------------------------------------- #
# Convenience CLI / quick test (optional)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def _demo() -> None:
        service = DuckDuckGoSearchService()
        links = await service.search("python asyncio tutorial", count=10)
        print("\n".join(links))

    asyncio.run(_demo())