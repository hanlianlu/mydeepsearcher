"""
confidence_agent.py
===================

Estimates whether the pipeline has *enough* information to craft a
comprehensive answer, outputting a float in [0.0, 1.0].

Typical placement: after each retrieve→rerank loop; ReflectionAgent
reads the score to decide early-stop or gap-search.
"""
from __future__ import annotations
from typing import Any, List, Dict

from autogen_agentchat.agents   import AssistantAgent
from autogen_agentchat.messages import TextMessage as Message

from deepsearcher.utils.rag_prompts import CONFIDENCE_PROMPT
from deepsearcher.utils.rag_helpers  import format_retrieved_results
from deepsearcher.agent.base         import describe_class
from deepsearcher.vector_db          import RetrievalResult
from deepsearcher.utils.autogen_helper import kw_for_assistant_agent


@describe_class(
    "ConfidenceAgent evaluates retrieved evidence and yields a number "
    "between 0 and 1 indicating how certain we can answer the query "
    "without further search."
)
class ConfidenceAgent(AssistantAgent):

    def __init__(
        self,
        llm_client: Any,
        *,
        name: str = "confidence",
    ):
        _super_kwargs = {
            "name"           : name,
            "model_client"   : llm_client,
            kw_for_assistant_agent(): {          # ← one line does the trick
                "cache_seed"  : 42
            },
        }
        super().__init__(**_super_kwargs)

    # ------------------------------------------------------------------ #
    # message-driven hook
    # ------------------------------------------------------------------ #
    async def a_receive(
        self,
        messages: List[Message],
        sender:   "AssistantAgent",
        config:   Dict | None = None,
    ) -> Message:
        """
        Expects latest message.content::

            {
              "original_query": str,
              "sub_queries":    list[str],
              "chunks":         list[RetrievalResult]   # OPTIONAL
            }
        """
        pay = messages[-1].content

        original_q = pay.get("original_query", "")
        sub_qs     = pay.get("sub_queries",   [])
        hits       = pay.get("chunks",        [])

        chunk_str  = (
            format_retrieved_results(hits) if hits
            else "No chunks retrieved."
        )

        prompt = CONFIDENCE_PROMPT.format(
            original_query = original_q,
            sub_queries    = ", ".join(sub_qs),
            chunk_str      = chunk_str,
        )

        reply = await self.model_client.chat_async([{"role":"user","content":prompt}])

        # Parse and clamp
        try:
            score = float(reply.content.strip())
        except Exception:
            score = 0.0
        score = max(0.0, min(1.0, score))

        return self.send(
            content  = score,
            metadata = {"raw_reply": reply.content},
            sender   = self,
        )


# ---------------------------------------------------------------------- #
# factory helper
# ---------------------------------------------------------------------- #
def build(llm_client: Any) -> ConfidenceAgent:
    return ConfidenceAgent(llm_client)
