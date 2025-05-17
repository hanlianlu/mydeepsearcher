from __future__ import annotations
from typing import Any, List, Optional, Union

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage as Message
from deepsearcher.agent.base import describe_class

@describe_class(
    "Chain-of-RAG is a hybrid agent: it first delegates to the monolithic implementation, "
    "and if the monolith signals low confidence, it falls back to a micro-level loop."
)
class ChainOfRAGAgent(AssistantAgent):
    """AutoGen-compatible wrapper for the monolithic Chain-of-RAG, with optional micro-flow fallback."""

    def __init__(
        self,
        *,
        llm: Any,
        lightllm: Any,
        highllm: Any,
        embedding_model: Any,
        vector_db: Any,
        max_iter: int = 5,
        monolith_conf_tag: str = "FALLBACK:",
        enable_micro: bool = False,
        name: str = "chain_of_rag",
    ):
        # We don't chat directly; the monolith handles all RAG logic
        super().__init__(name=name, model_client=None)
        self._llm = llm
        self._lightllm = lightllm
        self._highllm = highllm
        self._embed = embedding_model
        self._vectordb = vector_db
        self._max_iter = max_iter
        self._conf_tag = monolith_conf_tag
        self._enable_micro = enable_micro
        self._internal: Optional[Any] = None  # lazy monolith instance

    async def run(self, task: Union[str, Message], **kwargs) -> Message:
        query = task.content if isinstance(task, Message) else task

        # Lazy-load the monolithic ChainOfRAG
        if self._internal is None:
            from deepsearcher.agent.chain_of_rag import ChainOfRAG  # monolith entrypoint
            self._internal = ChainOfRAG(
                llm=self._llm,
                lightllm=self._lightllm,
                highllm=self._highllm,
                embedding_model=self._embed,
                vector_db=self._vectordb,
                max_iter=self._max_iter,
            )

        # Delegate to monolith
        answer, *rest = await self._internal.async_query(query, **kwargs)
        # If monolith did NOT flag low confidence, return its full answer
        if not (self._enable_micro and answer.startswith(self._conf_tag)):
            return Message(role="assistant", content=answer)

        # Otherwise, strip the tag and use micro-flow (commented until ready)
        cleaned = answer[len(self._conf_tag):].lstrip()
        # from deepsearcher.pipelines.chainofrag_flow import build_flow
        # flow = build_flow(ctx=kwargs.get("ctx"))
        # micro_msg = flow.run(task=cleaned)
        # return micro_msg

        return Message(role="assistant", content=cleaned)


def build(cfg) -> ChainOfRAGAgent:
    """Factory for ChainOfRAGAgent, used by RAGRouter pipeline registry."""
    return ChainOfRAGAgent(
        llm=cfg.llm_client,
        lightllm=getattr(cfg, "lightllm", cfg.llm_client),
        highllm=getattr(cfg, "highllm", cfg.llm_client),
        embedding_model=cfg.embedding_model,
        vector_db=cfg.vector_db,
        max_iter=cfg.config.query_settings.get("max_iter", 5),
    )
