# deepsearcher/agents/finalpaper_agent.py
"""
Wraps the existing FinalPaperAgent monolith as a single GraphFlow participant,
with lazy import to avoid circular dependencies.
"""
from typing import Any, List, Dict
from autogen_agentchat.agents import AssistantAgent

class FinalPaperAgentWrapper(AssistantAgent):
    """
    Produces a thesis-level report by delegating to the monolithic FinalPaperAgent.

    Inputs:
      - messages[0].content: the original query (str)
      - config['retrieved_results']: List[RetrievalResult]
      - config['sub_queries']: List[str]
      - config['history_context']: str

    Returns:
      A comprehensive final answer (str).
    """

    def __init__(
        self,
        lightllm: Any,
        highllm: Any,
    ):
        super().__init__(
            name="finalpaper",
            model_client=highllm,
            system_message=(
                "You are to write a thesis-level report based on retrieved data and context."
            ),
        )
        # Save dependencies, but don't import monolith until run()
        self.lightllm = lightllm
        self.highllm = highllm
        self._internal = None

    async def run(
        self,
        messages: List[Any],
        sender: str,
        config: Dict[str, Any],
    ) -> str:
        # Lazy import FinalPaperAgent to avoid circular import
        if self._internal is None:
            from deepsearcher.agent.final_paper import FinalPaperAgent
            self._internal = FinalPaperAgent(
                lightllm=self.lightllm,
                highllm=self.highllm,
            )

        # Extract inputs
        query = messages[0].content
        retrieved = config.get("retrieved_results", [])
        sub_queries = config.get("sub_queries", [])
        history = config.get("history_context", "")

        # Delegate to original monolith's generate_response
        # Assume generate_response is async
        final = await self._internal.generate_response(
            query=query,
            retrieved_results=retrieved,
            sub_queries=sub_queries,
            history_context=history,
        )
        return final


def build(
    lightllm: Any,
    highllm: Any,
) -> FinalPaperAgentWrapper:
    """
    Factory: returns a FinalPaperAgentWrapper bound to the given LLM clients.
    """
    return FinalPaperAgentWrapper(lightllm, highllm)
