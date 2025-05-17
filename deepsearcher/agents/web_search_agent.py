"""
web_search_retriever.py
=======================

Provides live-web augmentation for RAG pipelines.
Becomes a no-op if `use_web_search` is false in *flow.config*.
"""

from __future__ import annotations
import asyncio, logging
from typing import Any, List

from autogen_agentchat.agents   import AssistantAgent
from autogen_agentchat.messages import TextMessage as Message

from deepsearcher.agent.base    import describe_class
from deepsearcher.vector_db     import RetrievalResult

logger = logging.getLogger(__name__)


@describe_class(
    "WebSearchRetrievalAgent augments local RAG with live web evidence. "
    "Toggled per-request via the `use_web_search` flag in GraphFlow.config."
)
class WebSearchRetrievalAgent(AssistantAgent):
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        *,
        search_service,
        web_crawler,
        llm_client,
        enabled: bool = False,               # global default
        max_urls: int = 10,
        max_chunk_size: int = 6000,
        url_relevance_threshold: float = 0.71,
        name: str = "web_search",
    ):
        # Wrapper node (doesn’t chat itself)
        super().__init__(name=name, model_client=None)

        self._enabled_default = enabled
        self._internal = None               # lazy WebSearchAgent
        self._init_kwargs = dict(
            search_service        = search_service,
            web_crawler           = web_crawler,
            llm                   = llm_client,
            max_urls              = max_urls,
            max_chunk_size        = max_chunk_size,
            url_relevance_threshold = url_relevance_threshold,
        )

    # ------------------------------------------------------------------ #
    # Helper: decide if this turn should hit the web
    # ------------------------------------------------------------------ #
    def _use_web(self) -> bool:
        """
        Look for `use_web_search` in the *group-chat* config.
        Fall back to the constructor’s `enabled` default.
        """
        grp_cfg = getattr(self.group_chat, "config", {}) or {}
        return bool(grp_cfg.get("use_web_search", self._enabled_default))

    # ------------------------------------------------------------------ #
    async def a_receive(self, messages: List[Message]) -> Message:
        payload = messages[-1].content

        if not self._use_web():
            # No-op → just forward the payload unchanged
            return self.send(content=payload, sender=self)

        # ── prepare queries ────────────────────────────────────────────
        sub_qs  = payload.get("sub_queries") or [payload.get("original_query", "")]
        history = payload.get("history", "")

        # ── lazy-instantiate WebSearchAgent ────────────────────────────
        if self._internal is None:
            from deepsearcher.agent.web_search_retriever import WebSearchAgent
            self._internal = WebSearchAgent(**self._init_kwargs)

        # ── execute searches concurrently ─────────────────────────────
        async def _one(q):
            hits, _, _ = await self._internal.retrieve(query=q, history_context=history)
            return hits

        sem = asyncio.Semaphore(5)

        async def _wrapped(q):
            async with sem:
                return await _one(q)

        batches: List[List[RetrievalResult]] = await asyncio.gather(
            *[_wrapped(q) for q in sub_qs]
        )
        raw_hits: List[RetrievalResult] = [h for batch in batches for h in batch]

        # ── build preview for downstream LLM steps ─────────────────────
        preview = [
            {
                "id": i,
                "text": h.text[:150] + ("…" if len(h.text) > 150 else ""),
                "source": h.metadata.get("url") or h.metadata.get("source", ""),
            }
            for i, h in enumerate(raw_hits[:25])   # cap preview length
        ]

        return self.send(
            content  = {**payload, "web_chunks": preview},
            metadata = {"web_chunks_raw": raw_hits},
            sender   = self,
        )


# ---------------------------------------------------------------------- #
# Factory helper
# ---------------------------------------------------------------------- #
def build(ctx, *, enabled: bool | None = None) -> WebSearchRetrievalAgent:
    """
    • ctx : RuntimeContext (search_service, web_crawler, llm_client …)
    • enabled : override global default; if None, fall back to
                ctx.config.query_settings["use_web_search"] or False
    """
    global_flag = bool(ctx.config.query_settings.get("use_web_search", False))
    final_flag  = global_flag if enabled is None else bool(enabled)

    return WebSearchRetrievalAgent(
        search_service = ctx.search_service,
        web_crawler    = ctx.web_crawler,
        llm_client     = ctx.lightllm or ctx.llm_client,
        enabled        = final_flag,
    )
