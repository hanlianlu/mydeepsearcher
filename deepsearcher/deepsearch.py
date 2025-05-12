# deepsearcher/deepsearch.py
import asyncio
from deepsearcher.orchestration.graph_flow import build_flow
from deepsearcher.utils.rag_helpers import (
    format_retrieved_results,
    parse_rerank_response,
    confidence_prompt,
)

from deepsearcher.utils.rag_prompts import (
    SUB_QUERY_PROMPT, 
    RERANK_PROMPT,
    REFLECT_PROMPT,
    SUMMARY_PROMPT
)

class DeepSearch:
    def __init__(self, user_id="default_user"):
        self.flow = build_flow()

    async def async_query(self, query: str, history_context: str = ""):
        # start the flow with the user's message
        final_msg = await self.flow.arun(
            {"role": "user", "content": query, "history_context": history_context}
        )
        return final_msg.content, [], 0

    def query(self, query: str, history_context: str = ""):
        return asyncio.run(self.async_query(query, history_context))
