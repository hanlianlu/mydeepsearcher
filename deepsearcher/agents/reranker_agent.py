# deepsearcher/agents/reranker_agent.py

from typing import List, Tuple, Any
from autogen_agentchat.agents import AssistantAgent
from deepsearcher.utils.rag_prompts import RERANK_PROMPT
from deepsearcher.utils.rag_helpers import (
    format_retrieved_results,
    parse_rerank_response,
)
from deepsearcher.vector_db.base import RetrievalResult

class RerankerAgent(AssistantAgent):
    """
    Agent that re-ranks the retrieved document chunks.

    Expects:
      - messages[0].content == original_query (str)
      - messages[1].metadata["tool_output"]["raw_results"] == List[RetrievalResult]
      - config["sub_queries"] == List[str]

    Returns:
      {"accepted": List[RetrievalResult]}
    """

    def __init__(self, llm_client: Any):
        super().__init__(
            name="reranker",
            model_client=llm_client,
            system_message=(
                "You are a document relevance evaluator. "
                "You will receive the queries/questions and a list of document candicates. "
                "Follow user's instructions to filter and score each document candicate."
            ),
        )

    async def run(
        self,
        messages: List[Any],
        sender: str,
        config: dict,
    ) -> dict:
        # 1) Original query
        original_query = messages[0].content

        # 2) Sub-queries passed along in config
        sub_queries: List[str] = config.get("sub_queries", [])

        # 3) Raw RetrievalResult list from the retrieval tool
        tool_msg = messages[1]
        raw_results: List[RetrievalResult] = (
            tool_msg.metadata.get("tool_output", {})
                         .get("raw_results", [])
        )

        # 4) Format for the prompt
        formatted_chunks = format_retrieved_results(raw_results)

        # 5) Fill and send the rerank prompt
        prompt = RERANK_PROMPT.format(
            original_query=original_query,
            sub_queries=", ".join(sub_queries),
            retrieved_chunk=formatted_chunks,
        )
        chat_resp = await self.model_client.chat_async(
            [{"role": "user", "content": prompt}]
        )

        # 6) Parse the JSON votes back into (result, score)
        accepted_with_scores: List[Tuple[RetrievalResult, float]] = parse_rerank_response(
            raw_results, chat_resp.content
        )

        # 7) Keep only the RetrievalResult objects
        accepted = [chunk for (chunk, _) in accepted_with_scores]

        return {"accepted": accepted}


def build(llm_client: Any) -> RerankerAgent:
    """
    Factory to create a RerankerAgent bound to the given LLM client.
    """
    return RerankerAgent(llm_client)
