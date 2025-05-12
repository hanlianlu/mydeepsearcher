# deepsearcher/agents/confidence_agent.py

import ast
from typing import Any, List, Dict
from autogen_agentchat.agents import AssistantAgent
from deepsearcher.utils.rag_helpers import format_retrieved_results
from deepsearcher.vector_db.base import RetrievalResult

# Prompt template for confidence assessment
CONFIDENCE_PROMPT = """
Based on the original query, previous sub-queries, and the retrieved document chunks, assess your confidence as a number between 0 and 1 that you have enough information to answer the query comprehensively, and able to provide profound insight.

Original Query: {original_query}
Previous Sub-Queries: {sub_queries}
Retrieved Chunks:
{chunk_str}

Respond with a single floating-point number with two decimal places (e.g., 0.85). No additional text.
"""

class ConfidenceAgent(AssistantAgent):
    """
    Agent that evaluates the confidence level of retrieval results.

    Inputs:
      - messages[0].content: the original query (str)
      - config['sub_queries']: List[str]
      - messages[1]: retrieval tool-result Message, with metadata['tool_output'] holding raw RetrievalResult list

    Returns:
      A float between 0.0 and 1.0 indicating confidence.
    """

    def __init__(self, llm_client: Any, temperature: float ):
        super().__init__(
            name="confidence_agent",
            model_client=llm_client,
            system_message=CONFIDENCE_PROMPT,
        )

    async def run(
        self,
        messages: List[Any],
        sender: str,
        config: Dict[str, Any],
    ) -> float:
        # Original user query
        original_query = messages[0].content

        # Sub-queries
        sub_queries: List[str] = config.get("sub_queries", [])

        # Raw retrieval results: try metadata first, then config
        raw_hits: List[RetrievalResult] = []
        # 1) If retrieval tool saved raw objects in metadata
        tool_msg = messages[1]
        if hasattr(tool_msg, 'metadata'):
            out = tool_msg.metadata.get('tool_output')
            if isinstance(out, dict) and 'retrieved' in out:
                # vectordb_search stored preview + raw under the same key
                raw_hits = out.get('raw', []) if 'raw' in out else []
            elif isinstance(out, list):
                raw_hits = out
        # 2) Fallback: config may carry raw results directly
        if not raw_hits and 'results' in config and isinstance(config['results'], list):
            raw_hits = config['results']  # could be RetrievalResult or (RetrievalResult, score)
            # If tuples, extract the first element
            raw_hits = [r if isinstance(r, RetrievalResult) else r[0] for r in raw_hits]

        # Format chunks into a single string
        if raw_hits:
            chunk_str = format_retrieved_results(raw_hits)
        else:
            chunk_str = "No chunks retrieved."

        # Fill and send the prompt
        prompt = CONFIDENCE_PROMPT.format(
            original_query=original_query,
            sub_queries=", ".join(sub_queries),
            chunk_str=chunk_str,
        )
        chat_resp = await self.model_client.chat_async(
            [{"role": "user", "content": prompt}]
        )
        # Parse float
        try:
            val = float(chat_resp.content.strip())
        except Exception:
            val = 0.0
        # Clamp to [0.0, 1.0]
        return max(0.0, min(1.0, val))


def build(llm_client: Any, temperature: float = 0.0) -> ConfidenceAgent:
    """
    Factory: returns a ConfidenceAgent bound to the given LLM client.
    """
    return ConfidenceAgent(llm_client, temperature)
