# deepsearcher/agents/deepsearch_agent.py
from __future__ import annotations
from typing import Any, List

from autogen_agentchat.agents   import AssistantAgent
from autogen_agentchat.messages import TextMessage as Message
from deepsearcher.agent.base    import describe_class


@describe_class(
    "DeepSearch is the ultimate all-purpose strategy. It decomposes the "
    "query, iteratively retrieves evidence, reflects on information gaps, "
    "and can escalate to a thesis-level FinalPaper when needed."
)
class DeepSearchAgent(AssistantAgent):
    """
    Wrapper around the legacy *DeepSearch* monolith (for now).
    When you’re ready, flip one line to delegate to the new micro-flow.
    """

    def __init__(
        self,
        *,
        llm,
        lightllm,
        highllm,
        embedding_model,
        vector_db,
        max_iter: int = 5,
        name: str = "deepsearch",
    ):
        # wrapper itself never chats
        super().__init__(name=name, model_client=None)

        self._llm        = llm
        self._lightllm   = lightllm
        self._highllm    = highllm
        self._embed      = embedding_model
        self._vectordb   = vector_db
        self._max_iter   = max_iter
        self._internal   = None   # lazy monolith instance

    # ------------------------------------------------------------------ #
    async def run(self, task: str | Message, **kwargs) -> Message:
        query = task.content if isinstance(task, Message) else task

        # ----- Option A: call legacy monolith ---------------------------------
        if self._internal is None:
            from deepsearcher.agent.deep_search import DeepSearch
            self._internal = DeepSearch(
                llm              = self._llm,
                lightllm         = self._lightllm,
                highllm          = self._highllm,
                embedding_model  = self._embed,
                vector_db        = self._vectordb,
                max_iter         = self._max_iter,
                route_collection = True,
                text_window_splitter = True,
            )

        answer, *_ = await self._internal.async_query(query, **kwargs)

        # ----- Option B: switch to micro-flow (uncomment when ready) ----------
        # from deepsearcher.pipelines.deepsearch_flow import build as build_flow
        # flow       = build_flow(kwargs.get("ctx"))  # pass RuntimeContext
        # answer_msg = flow.run(task=query)
        # answer     = answer_msg.content
        # ---------------------------------------------------------------------

        return Message(role="assistant", content=answer)


# ---------------------------------------------------------------------- #
def build(cfg) -> DeepSearchAgent:
    """Factory used by RAGRouter pipeline registry."""
    return DeepSearchAgent(
        llm             = cfg.llm_client,
        lightllm        = getattr(cfg, "lightllm", cfg.llm_client),
        highllm         = getattr(cfg, "highllm",  cfg.llm_client),
        embedding_model = cfg.embedding_model,
        vector_db       = cfg.vector_db,
        max_iter        = cfg.config.query_settings.get("max_iter", 5),
    )
