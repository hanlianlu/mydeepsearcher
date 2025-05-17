"""
ReflectionAgent
===============

After each retrieve→rerank pass this agent decides **whether more
search is needed**.  It examines:

    content = {
        "original_query": str,
        "sub_queries":    list[str],     # asked so far
    }

    metadata["accepted_chunks"] = list[RetrievalResult]  # from RerankerAgent

It returns a message whose `content` is either::

    { "new_queries": list[str] }    # loop again
or
    { "new_queries": [] }           # stop & summarise
"""

from __future__ import annotations
import ast
from typing import Any, Dict, List

from autogen_agentchat.agents   import AssistantAgent
from autogen_agentchat.messages import TextMessage as Message
from deepsearcher.utils.rag_prompts  import REFLECT_PROMPT
from deepsearcher.utils.rag_helpers  import format_retrieved_results
from deepsearcher.vector_db          import RetrievalResult


class ReflectionAgent(AssistantAgent):

    def __init__(
        self,
        llm_client: Any,
        *,
        name: str = "reflection",
        max_iter: int = 5,
        confidence_threshold: float = 0.94,   # optional second-stage stop
    ):
        super().__init__(name=name, model_client=llm_client)
        self.max_iter = max_iter
        self.conf_thr = confidence_threshold
        self._iter_seen: Dict[str, int] = {}   # track per-query loops

    # ------------------------------------------------------------------ #
    # message-driven entry-point
    # ------------------------------------------------------------------ #
    async def a_receive(
        self,
        messages: List[Message],
        sender:   "AssistantAgent",
        config:   Dict | None = None,
    ) -> Message:
        pay  = messages[-1].content
        meta = messages[-1].metadata

        orig_q: str        = pay.get("original_query", "")
        sub_qs: List[str]  = pay.get("sub_queries", [])
        chunks: List[RetrievalResult] = meta.get("accepted_chunks", [])

        # Loop-control: how many times have we reflected on *this* query?
        loop_id = orig_q   # could add user id
        self._iter_seen[loop_id] = self._iter_seen.get(loop_id, 0) + 1
        if self._iter_seen[loop_id] > self.max_iter:
            return self._return([], reason="Reached max_iter")

        # Build reflect prompt
        chunk_str = (
            format_retrieved_results(chunks) if chunks
            else "No chunks retrieved."
        )

        prompt = REFLECT_PROMPT.format(
            question       = orig_q,
            mini_questions = ", ".join(sub_qs),
            chunk_str      = chunk_str,
        )

        llm_resp = await self.model_client.chat_async([{"role": "user", "content": prompt}])

        # Parse → list[str]
        try:
            gap_queries = ast.literal_eval(llm_resp.content)
        except Exception:
            gap_queries = []

        if not isinstance(gap_queries, list):
            gap_queries = []
        gap_queries = [str(q).strip() for q in gap_queries if isinstance(q, str)]

        return self._return(gap_queries)

    # ------------------------------------------------------------------ #
    # helper: package outgoing message
    # ------------------------------------------------------------------ #
    def _return(self, new_qs: List[str], *, reason: str | None = None) -> Message:
        return self.send(
            content  = {"new_queries": new_qs},
            metadata = {"reason": reason or ("follow-up" if new_qs else "sufficient")},
            sender   = self,
        )


# ---------------------------------------------------------------------- #
# factory helper
# ---------------------------------------------------------------------- #
def build(
    llm_client: Any,
    *,
    max_iter: int = 5,
    confidence_threshold: float = 0.94,
) -> ReflectionAgent:
    return ReflectionAgent(
        llm_client,
        max_iter=max_iter,
        confidence_threshold=confidence_threshold,
    )
