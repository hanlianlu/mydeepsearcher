# deepsearcher/agents/reflection_agent.py

import ast
from typing import Any, List
from autogen_agentchat.agents import AssistantAgent
from deepsearcher.utils.rag_prompts import REFLECT_PROMPT
from deepsearcher.utils.rag_helpers import format_retrieved_results
from deepsearcher.vector_db.base import RetrievalResult


class ReflectionAgent(AssistantAgent):
    """
    Agent that decides whether additional search queries are needed.

    Expects:
      - messages[0].content == original_query (str)
      - config['sub_queries'] == List[str] from the decomposer
      - messages[-1].content is a dict {'accepted': List[RetrievalResult]}
        or directly a List[RetrievalResult]

    Returns:
      A Python list of up to max_iter new sub-queries (or [] if none).
    """

    def __init__(self, llm_client: Any, max_iter: int = 5):
        super().__init__(
            name="reflector",
            model_client=llm_client,
            system_message=(
                "Based on the original query, previous sub-queries, and the retrieved chunks, "
                "determine if more search is needed. "
                "If so, return a Python list of up to 3 new search queries. "
                "If no further research is required, return an empty list."
            ),
        )
        self.max_iter = max_iter
        self._iter_count = 0

    async def run(
        self,
        messages: List[Any],
        sender: str,
        config: dict,
    ) -> List[str]:
        # Prevent infinite loops
        if self._iter_count >= self.max_iter:
            return []
        self._iter_count += 1

        # 1) Original query
        original_query = messages[0].content

        # 2) Sub-queries from config
        sub_queries: List[str] = config.get("sub_queries", [])
        
        # 3) Extract accepted chunks from the previous agent
        last = messages[-1].content
        if isinstance(last, dict) and "accepted" in last:
            accepted: List[RetrievalResult] = last["accepted"]
        elif isinstance(last, list) and all(isinstance(x, RetrievalResult) for x in last):
            accepted = last
        else:
            accepted = []

        # 4) Format into text for the prompt
        chunk_str = (
            format_retrieved_results(accepted)
            if accepted
            else "No chunks retrieved."
        )

        # 5) Build and send the reflection prompt
        prompt = REFLECT_PROMPT.format(
            question=original_query,
            mini_questions=", ".join(sub_queries),
            chunk_str=chunk_str,
        )
        chat_resp = await self.model_client.chat_async(
            [{"role": "user", "content": prompt}]
        )

        # 6) Parse the model's output as a Python list
        try:
            gap_queries = ast.literal_eval(chat_resp.content)
        except Exception:
            gap_queries = []

        # Ensure it's a list of strings
        if not isinstance(gap_queries, list):
            gap_queries = []
        gap_queries = [str(q).strip() for q in gap_queries if isinstance(q, str)]

        return gap_queries


def build(llm_client: Any, max_iter: int = 3) -> ReflectionAgent:
    """
    Factory: create a ReflectionAgent bound to the given LLM client and iteration cap.
    """
    return ReflectionAgent(llm_client, max_iter)
