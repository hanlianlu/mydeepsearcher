# deepsearcher/agents/summarizer_agent.py
"""
SummarizerAgent: uses the conversation history + final retrieval results
and the structured SUMMARY_PROMPT to compose the final answer in Markdown.
"""
from typing import Any
from autogen_agentchat.agents import AssistantAgent
from deepsearcher.utils.rag_prompts import SUMMARY_PROMPT

class SummarizerAgent(AssistantAgent):
    """
    Agent that composes the final answer once no further queries are needed.

    It uses the SUMMARY_PROMPT template to instruct the LLM to synthesize
    sub-queries, retrieved chunks, and history context into a Markdown response.
    """
    def __init__(self, llm_client: Any, temperature: float):
        super().__init__(
            name="summarizer",
            model_client=llm_client,
            system_message=SUMMARY_PROMPT,
        )

        # No custom run(): rely on default AssistantAgent.run(), which
        # feeds the entire conversation (messages) to the LLM along with system_message.


def build(llm_client: Any) -> SummarizerAgent:
    """
    Factory: returns a SummarizerAgent bound to the given LLM client.
    """
    return SummarizerAgent(llm_client)
