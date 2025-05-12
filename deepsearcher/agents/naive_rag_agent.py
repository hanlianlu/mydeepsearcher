# deepsearcher/agents/naive_rag_agent.py
"""
Wraps the monolithic NaiveRAG pipeline as a single GraphFlow node,
with lazy import to avoid circular dependencies.
"""
from typing import Any, List, Dict
from autogen_agentchat.agents import AssistantAgent

class NaiveRAGAgent(AssistantAgent):
    """
    Executes a simple retrieve-and-generate RAG process using NaiveRAG.

    Inputs:
      - messages[0].content: the original query (str)
    Returns:
      The generated answer (str).
    """
    def __init__(
        self,
        llm: Any,
        embedding_model: Any,
        vector_db: Any,
        top_k: int = 10,
        route_collection: bool = True,
        text_window_splitter: bool = True,
    ):
        super().__init__(
            name="naive_rag",
            model_client=llm,
            system_message=(
                "Perform a simple RAG: retrieve top-k chunks and answer from them."
            ),
        )
        # Save dependencies for lazy import
        self.llm = llm
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        self.top_k = top_k
        self.route_collection = route_collection
        self.text_window_splitter = text_window_splitter
        self._internal = None

    async def run(
        self,
        messages: List[Any],
        sender: str,
        config: Dict[str, Any],
    ) -> str:
        # Lazy import NaiveRAG to break circular imports
        if self._internal is None:
            from deepsearcher.agent.naive_rag import NaiveRAG
            self._internal = NaiveRAG(
                llm=self.llm,
                embedding_model=self.embedding_model,
                vector_db=self.vector_db,
                top_k=self.top_k,
                route_collection=self.route_collection,
                text_window_splitter=self.text_window_splitter,
            )

        # Delegate to the monolith's query method
        query = messages[0].content
        answer, _, _ = self._internal.query(
            query,
            **config,
        )
        return answer


def build(cfg: Any) -> NaiveRAGAgent:
    """
    Factory: returns a NaiveRAGAgent bound to the given configuration.
    """
    # Choose the light LLM if available for speed
    web_llm = getattr(cfg, 'lightllm', None) or cfg.llm_client
    return NaiveRAGAgent(
        llm=web_llm,
        embedding_model=cfg.embedding_model,
        vector_db=cfg.vector_db,
        top_k=getattr(cfg.query_settings, 'top_k', 10),
        route_collection=True,
        text_window_splitter=True,
    )
