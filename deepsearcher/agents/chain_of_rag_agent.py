# deepsearcher/agents/chainofrag_agent.py
"""
Wraps the existing ChainOfRAG monolith as a single GraphFlow participant,
with lazy import to avoid circular dependencies.
"""
from typing import Any, List, Dict
from autogen_agentchat.agents import AssistantAgent

class ChainOfRAGAgent(AssistantAgent):
    """
    Precision-focused RAG pipeline that dynamically crafts and iterates subqueries.

    Inputs:
      - messages[0].content: the original query (str)
      - config may contain history or other flags if needed

    Returns:
      The final answer (str) from the ChainOfRAG monolith.
    """
    def __init__(
        self,
        llm: Any,
        lightllm: Any,
        highllm: Any,
        embedding_model: Any,
        vector_db: Any,
        max_iter: int,
    ):
        super().__init__(
            name="chainofrag",
            model_client=highllm,
            system_message=(
                "Executing Chain-of-RAG precision pipeline as a single agent."
            ),
        )
        # Save dependencies for lazy instantiation
        self.llm = llm
        self.lightllm = lightllm
        self.highllm = highllm
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        self.max_iter = max_iter
        self._internal = None

    async def run(
        self,
        messages: List[Any],
        sender: str,
        config: Dict[str, Any],
    ) -> str:
        # Lazy import to avoid circular imports at module load
        if self._internal is None:
            from deepsearcher.agent.chain_of_rag import ChainOfRAG
            self._internal = ChainOfRAG(
                llm=self.llm,
                lightllm=self.lightllm,
                highllm=self.highllm,
                embedding_model=self.embedding_model,
                vector_db=self.vector_db,
                max_iter=self.max_iter,
            )

        # Delegate to the monolith's query method
        query = messages[0].content
        answer, _, _ = self._internal.query(
            query,
            **config,
        )
        return answer


def build(
    llm: Any,
    lightllm: Any,
    highllm: Any,
    embedding_model: Any,
    vector_db: Any,
    max_iter: int,
) -> ChainOfRAGAgent:
    """
    Factory: returns a ChainOfRAGAgent bound to the given dependencies.
    """
    return ChainOfRAGAgent(
        llm,
        lightllm,
        highllm,
        embedding_model,
        vector_db,
        max_iter,
    )
