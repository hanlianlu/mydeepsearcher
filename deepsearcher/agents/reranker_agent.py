"""
RerankerAgent
=============

An AssistantAgent that uses an LLM to *filter / score* retrieved chunks.

Input message (from RetrievalAgent or another step) **must** carry:

    content = {
        "original_query": str,
        "sub_queries":    list[str],
    }
    metadata["tool_output"] = List[RetrievalResult]   # raw hits

Output message:

    content = {
        "accepted": List[dict],   # preview of accepted chunks
    }
    metadata["accepted_chunks"] = List[RetrievalResult]  # full objects
"""

from __future__ import annotations
from typing import List, Dict, Any, Tuple

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage        # ← replaces “Message”

from deepsearcher.utils.rag_prompts import RERANK_PROMPT
from deepsearcher.utils.rag_helpers import (
    format_retrieved_results,
    parse_rerank_response,
)
from deepsearcher.vector_db import RetrievalResult


class RerankerAgent(AssistantAgent):
    """
    LLM-based reranker / filter step.
    """

    def __init__(
        self,
        llm_client: Any,
        *,
        name: str = "reranker",
        min_keep_score: float = 0.40,
    ):
        super().__init__(name=name, model_client=llm_client)
        self.min_keep_score = min_keep_score

    # ------------------------------------------------------------------ #
    # Message-driven hook
    # ------------------------------------------------------------------ #
    async def a_receive(
        self,
        messages: List[TextMessage],
        sender:   "AssistantAgent",
        config:   Dict[str, Any] | None = None,
    ) -> TextMessage:
        # ---- Extract payload -------------------------------------------------
        m          = messages[-1]
        payload    = m.content
        raw_hits   : List[RetrievalResult] = m.metadata.get("tool_output", [])

        if not raw_hits:
            # Nothing to rerank – pass through
            return self.send(
                content={"accepted": []},
                metadata={"accepted_chunks": []},
                sender=self,
            )

        original_query: str  = payload.get("original_query", "")
        sub_queries   : List[str] = payload.get("sub_queries", [])

        # ---- Prepare and send prompt ----------------------------------------
        prompt = RERANK_PROMPT.format(
            original_query = original_query,
            sub_queries    = ", ".join(sub_queries),
            retrieved_chunk= format_retrieved_results(raw_hits),
        )

        llm_resp = await self.model_client.chat_async([{"role": "user", "content": prompt}])

        # ---- Parse JSON response --------------------------------------------
        accepted_pairs: List[Tuple[RetrievalResult, float]] = parse_rerank_response(
            raw_hits, llm_resp.content, keep_threshold=self.min_keep_score
        )
        accepted_chunks = [c for c, _ in accepted_pairs]

        preview = [
            {
                "id": i,
                "text": c.text[:3000] + ("…" if len(c.text) > 3000 else ""),
                "score": score,
            }
            for i, (c, score) in enumerate(accepted_pairs)
        ]

        # ---- Return message --------------------------------------------------
        return self.send(
            content  = {"accepted": preview},
            metadata = {"accepted_chunks": accepted_chunks},
            sender   = self,
        )


# ---------------------------------------------------------------------- #
# Factory helper
# ---------------------------------------------------------------------- #

def build(llm_client: Any, *, min_keep_score: float = 0.40) -> RerankerAgent:
    """
    Convenience constructor used by pipeline builders.
    """
    return RerankerAgent(llm_client, min_keep_score=min_keep_score)
