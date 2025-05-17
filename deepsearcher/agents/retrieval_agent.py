# deepsearcher/agents/retrieval_agent.py
"""
RetrievalAgent
==============

A lightweight AssistantAgent that **does not generate LLM tokens** – it
performs pure vector-DB similarity search, packages the results, and
returns them as a message.  Because it subclasses `AssistantAgent`, you
can insert it directly as a node in GraphFlow and enjoy uniform message
logging / metadata.

The actual DB logic reuses your existing helpers in
`deepsearcher.vector_db` and `deepsearcher.embedding` – no duplication.
"""

from __future__ import annotations
import asyncio, json
from typing import Any, List, Dict

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from deepsearcher.vector_db.base import deduplicate_results, BaseVectorDB
from deepsearcher.vector_db import RetrievalResult
from deepsearcher.embedding.base import BaseEmbedding


class RetrievalAgent(AssistantAgent):
    """Agent node that executes vector-DB searches for given sub-queries."""

    def __init__(
        self,
        vector_db:     BaseVectorDB,
        embed_model:   BaseEmbedding,
        top_k:         int = 9,
        name:          str = "retriever",
    ):
        super().__init__(name=name, model_client=None)  # no LLM needed
        self.vector_db   = vector_db
        self.embed_model = embed_model
        self.top_k       = top_k

    # ------------------------------------------------------------------ #
    # Autogen entry-point: handle incoming message
    # ------------------------------------------------------------------ #
    async def a_receive(
        self,
        messages: List[TextMessage],
        sender:   "AssistantAgent",
        config:   Dict[str, Any] | None = None,
    ) -> TextMessage:
        """
        Expects the latest user message to contain a dict::

            {
                "sub_queries": [...],
                "collections": [...]   # optional
            }

        Returns a message with a preview + stores raw hits in metadata.
        """
        payload = messages[-1].content
        sub_queries: List[str] = payload.get("sub_queries", [])
        collections: List[str] | None = payload.get("collections")

        preview, raw_hits = await self._search(sub_queries, collections)

        return self.send(
            content={"preview": preview, "total_hits": len(raw_hits)},
            metadata={"tool_output": raw_hits},          # keep raw hits
            sender=self,
        )

    # ------------------------------------------------------------------ #
    # Internal async search helper
    # ------------------------------------------------------------------ #
    async def _search(
        self,
        sub_queries: List[str],
        collections: List[str] | None,
    ) -> tuple[list[dict], list[RetrievalResult]]:
        if not sub_queries:
            return [], []

        cols = collections or self.vector_db.list_collections()
        sem  = asyncio.Semaphore(10)

        async def _query_one(coll: str, q: str):
            async with sem:
                return await asyncio.to_thread(
                    self.vector_db.search_data,
                    collection=coll,
                    vector=self.embed_model.embed_query(q),
                    limit=self.top_k,
                )

        batches = await asyncio.gather(
            *[_query_one(c, q) for q in sub_queries for c in cols]
        )
        hits: List[RetrievalResult] = deduplicate_results(
            [h for batch in batches for h in batch]
        )

        preview = [
            {
                "id": i,
                "text": h.text[:3000] + ("…" if len(h.text) > 3000 else ""),
                "score": getattr(h, "distance", None),
                "source": h.metadata.get("source", ""),
            }
            for i, h in enumerate(hits)
        ]
        return preview, hits


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------

def build(cfg) -> RetrievalAgent:
    """Factory for pipeline registry (mirrors build() pattern in other agents)."""
    return RetrievalAgent(
        vector_db   = cfg.vector_db,
        embed_model = cfg.embedding_model,
        top_k       = cfg.config.query_settings.get("top_k", 9),
        name        = getattr(cfg, "agent_name", "retriever"),
    )
