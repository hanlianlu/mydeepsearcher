"""
chainofrag_agent.py
===================

Single-node wrapper around the monolithic *ChainOfRAG* class.  Keeps the
legacy behaviour but exposes it as an Autogen AssistantAgent so it can
be selected by RAGRouter or called directly.
"""
from __future__ import annotations
from typing import Any, List, Dict

from autogen_agentchat.agents   import AssistantAgent
from autogen_agentchat.messages import TextMessage as Message

from deepsearcher.agent.base     import describe_class


@describe_class(
    "ChainOfRAG is a specialized, precision-focused agent, ideal for queries that demand laser-like accuracy and multi-step reasoning. "
    "It dynamically adapts to gaps in information, crafting reactive subqueries to uncover hidden connections and deliver structured, citation-rich answers. "
    "While not the first choice for broad exploration, it shines in scenarios where precision, transparency, and factual rigor are paramount. "
    "It is NOT suitable for queries requiring critical thinking, comparative studies or long thesis/reports."
)
class ChainOfRAGAgent(AssistantAgent):
    """
    Wrapper: delegates all heavy work to the legacy ChainOfRAG.monolith.

    • Lazy-imports ChainOfRAG to avoid circular imports.
    • .run(task) accepts either raw str or TextMessage.
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
        name: str = "chainofrag",
    ):
        # The wrapper itself never chats → model_client=None
        super().__init__(name=name, model_client=None)

        self._llm            = llm
        self._lightllm       = lightllm
        self._highllm        = highllm
        self._embed          = embedding_model
        self._vectordb       = vector_db
        self._max_iter       = max_iter
        self._internal       = None   # lazy ChainOfRAG instance

    # ------------------------------------------------------------------ #
    # task-driven entry
    # ------------------------------------------------------------------ #
    async def run(
        self,
        task: str | Message,
        **query_kwargs,
    ) -> Message:
        query = task.content if isinstance(task, Message) else task

        # lazy import
        if self._internal is None:
            from deepsearcher.agent.chain_of_rag import ChainOfRAG
            self._internal = ChainOfRAG(
                llm             = self._llm,
                lightllm        = self._lightllm,
                highllm         = self._highllm,
                embedding_model = self._embed,
                vector_db       = self._vectordb,
                max_iter        = self._max_iter,
            )

        answer, *_ = await self._internal.async_query(query, **query_kwargs)
        return Message(role="assistant", content=answer)


# ---------------------------------------------------------------------- #
# factory shortcut
# ---------------------------------------------------------------------- #
def build(cfg) -> ChainOfRAGAgent:
    """
    cfg  : RuntimeContext holding llm, lightllm, highllm, embedding_model, vector_db.
    """
    return ChainOfRAGAgent(
        llm             = cfg.llm_client,
        lightllm        = getattr(cfg, "lightllm", cfg.llm_client),
        highllm         = getattr(cfg, "highllm", cfg.llm_client),
        embedding_model = cfg.embedding_model,
        vector_db       = cfg.vector_db,
        max_iter        = cfg.config.query_settings.get("max_iter", 5),
    )
