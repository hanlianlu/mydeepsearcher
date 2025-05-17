"""
NaiveRAGAgent
=============

A thin wrapper around the legacy *NaiveRAG* pipeline so that it can be
selected by RAGRouterAgent via its @describe_class annotation and run as
a single node in modern flows.
"""
from __future__ import annotations
from typing import Any, List, Dict

from autogen_agentchat.agents   import AssistantAgent
from autogen_agentchat.messages import TextMessage as Message

# Re-use the decorator already defined for DeepSearch / ChainOfRAG
from deepsearcher.agent.base    import describe_class


@describe_class(
    "NaiveRAG is the fastest, most lightweight option. "
    "It performs a single vector search and immediately generates an "
    "answer without iterative reasoning or multi-hop decomposition."
    "It is ideal for straightforward factual query or trivial questions where "
    "speed is more important than accuracy and coverage."
)
class NaiveRAGAgent(AssistantAgent):
    """
    Executes the legacy NaiveRAG.query() internally; acts as a single
    AssistantAgent node for GraphFlow or can be called via .run(task=…).
    """

    def __init__(
        self,
        *,
        llm,
        embedding_model,
        vector_db,
        top_k: int = 10,
        route_collection: bool = True,
        text_window_splitter: bool = True,
        name: str = "naive_rag",
    ):
        # No LLM chat inside the wrapper
        super().__init__(name=name, model_client=None)

        self._llm               = llm
        self._embedding_model   = embedding_model
        self._vector_db         = vector_db
        self._top_k             = top_k
        self._route_collection  = route_collection
        self._textwin_splitter  = text_window_splitter
        self._internal          = None   # lazy-loaded NaiveRAG instance

    # ------------------------------------------------------------------ #
    # task-driven entry point (stand-alone or from router)
    # ------------------------------------------------------------------ #
    async def run(
        self,
        task: str | Message,
        **query_kwargs,
    ) -> Message:
        query = task.content if isinstance(task, Message) else task

        # Lazy import to avoid circular dependency
        if self._internal is None:
            from deepsearcher.agent.naive_rag import NaiveRAG

            self._internal = NaiveRAG(
                llm                 = self._llm,
                embedding_model     = self._embedding_model,
                vector_db           = self._vector_db,
                top_k               = self._top_k,
                route_collection    = self._route_collection,
                text_window_splitter= self._textwin_splitter,
            )

        answer, _, _ = await self._internal.async_query(query, **query_kwargs)
        return Message(role="assistant", content=answer)


# ---------------------------------------------------------------------- #
# factory helper
# ---------------------------------------------------------------------- #
def build(cfg) -> NaiveRAGAgent:
    """
    `cfg` is RuntimeContext produced by your configuration bootstrap.
    """
    web_llm = getattr(cfg, "lightllm", None) or cfg.llm_client
    return NaiveRAGAgent(
        llm               = web_llm,
        embedding_model   = cfg.embedding_model,
        vector_db         = cfg.vector_db,
        top_k             = cfg.config.query_settings.get("top_k", 10),
        route_collection  = True,
        text_window_splitter = True,
    )
