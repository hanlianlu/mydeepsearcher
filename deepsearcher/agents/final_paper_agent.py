"""
finalpaper_agent.py
===================

A wrapper that turns the legacy *FinalPaperAgent* (monolithic, heavy
report generator) into a single message-driven node usable in Autogen
GraphFlows.
"""
from __future__ import annotations
from typing import Any, List, Dict

from autogen_agentchat.agents   import AssistantAgent
from autogen_agentchat.messages import TextMessage as Message

from deepsearcher.agent.base     import describe_class


@describe_class(
    "FinalPaperAgent writes a comprehensive, thesis-level report.  "
    "It weaves together retrieved evidence, sub-query reasoning, and "
    "conversation history into a structured Markdown document suitable "
    "for deep analytical or executive-level consumption."
)
class FinalPaperAgentWrapper(AssistantAgent):
    """
    Message-driven wrapper around *deepsearcher.agent.final_paper.FinalPaperAgent*.
    """

    def __init__(
        self,
        lightllm: Any,
        highllm:  Any,
        *,
        name: str = "finalpaper",
    ):
        # The wrapper itself does NO LLM chat – delegate to monolith
        super().__init__(name=name, model_client=None)

        self._lightllm = lightllm
        self._highllm  = highllm
        self._internal = None  # lazy-loaded FinalPaperAgent instance

    # ------------------------------------------------------------------ #
    # GraphFlow entry-point
    # ------------------------------------------------------------------ #
    async def a_receive(
        self,
        messages: List[Message],
        sender:   "AssistantAgent",
        config:   Dict | None = None,
    ) -> Message:
        """
        Expects latest message.content to carry:

            {
                "original_query":   str,
                "sub_queries":      list[str],
                "chunks":           list[RetrievalResult],
                "history":          str | "",
            }
        """
        pay = messages[-1].content

        # Lazy import to avoid circulars
        if self._internal is None:
            from deepsearcher.agent.final_paper import FinalPaperAgent
            self._internal = FinalPaperAgent(
                lightllm=self._lightllm,
                highllm =self._highllm,
            )

        final_answer = await self._internal.generate_response(
            query             = pay.get("original_query", ""),
            retrieved_results = pay.get("chunks", []),
            sub_queries       = pay.get("sub_queries", []),
            history_context   = pay.get("history", ""),
        )

        return self.send(
            content  = final_answer,                 # Markdown thesis
            metadata = {"generator": "FinalPaperAgent"},
            sender   = self,
        )


# ---------------------------------------------------------------------- #
# factory helper
# ---------------------------------------------------------------------- #
def build(lightllm: Any, highllm: Any) -> FinalPaperAgentWrapper:
    """
    Convenience constructor used by pipeline builders after JudgeAgent
    signals that a long-form report is required.
    """
    return FinalPaperAgentWrapper(lightllm, highllm)
