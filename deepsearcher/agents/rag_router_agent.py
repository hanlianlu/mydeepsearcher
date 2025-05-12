# deepsearcher/agents/rag_router_agent.py

import ast
from typing import Any, List
from autogen_agentchat.agents import AssistantAgent
from deepsearcher.utils.rag_prompts import RAG_ROUTER_PROMPT

class RAGRouterAgent(AssistantAgent):
    """
    Chooses which RAG pipeline to run:
      1 → DeepSearch pipeline
      2 → NaiveRAG
      3 → ChainOfRAG
    """

    def __init__(
        self,
        llm_client: Any,
        descriptions: List[str],
        max_iter: int = 1,
    ):
        # system_message is our RAG_ROUTER_PROMPT
        super().__init__(
            name="rag_router",
            model_client=llm_client,
            system_message=RAG_ROUTER_PROMPT,
        )
        self.descriptions = descriptions
        self.max_iter = max_iter

    async def run(self, messages: List[Any], sender: str, config: dict) -> int:
        # messages[0].content is the raw user query
        query = messages[0].content
        # Build the “AGENTS:” block
        desc_block = "\n".join(f"[{i+1}]: {d}" for i, d in enumerate(self.descriptions))
        prompt = RAG_ROUTER_PROMPT.format(
            query=query, description_str=desc_block
        )
        # Ask the LLM
        chat_resp = await self.model_client.chat_async([{"role":"user","content":prompt}])
        text = chat_resp.content.strip()
        # Parse the integer
        try:
            idx = int(text) - 1
        except:
            # fallback: pick the first digit you see
            for ch in text:
                if ch.isdigit():
                    idx = int(ch) - 1
                    break
            else:
                idx = 0
        return idx


def build(
    llm_client: Any,
    descriptions: List[str],
    max_iter: int = 1,
) -> RAGRouterAgent:
    """
    Factory: create a RAGRouterAgent bound to the given LLM and agent descriptions.
    """
    return RAGRouterAgent(llm_client, descriptions, max_iter)
