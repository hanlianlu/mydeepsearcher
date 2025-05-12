# deepsearcher/agents/decomposer_agent.py

import ast
from typing import Any, List, Dict
from autogen_agentchat.agents import AssistantAgent
from deepsearcher.utils.rag_prompts import SUB_QUERY_PROMPT

class DecomposerAgent(AssistantAgent):
    """
    Splits a user query into up to 4 focused sub-questions.
    Overrides run() so that downstream agents receive a real Python list.
    """

    def __init__(self, llm_client: Any):
        super().__init__(
            name="decomposer",
            model_client=llm_client,
            system_message=SUB_QUERY_PROMPT,
            llm_config={
                "cache_seed": 42,
                "temperature": 0.3,     #  splits
            },
        )

    async def run(
        self,
        messages: List[Any],
        sender: str,
        config: Dict[str, Any],
    ) -> List[str]:
        """
        messages[0].content    => the raw user question (string)
        config.get("history_context") => optional history
        Returns List[str]: the sub-questions.
        """
        # 1) Extract inputs
        original_query   = messages[0].content
        history_context  = config.get("history_context", "")

        # 2) Render the prompt template
        prompt_text = SUB_QUERY_PROMPT.format(
            original_query=original_query,
            history_context=history_context,
        )

        # 3) Call the LLM
        chat_resp = await self.model_client.chat_async(
            [{"role": "user", "content": prompt_text}]
        )

        # 4) Parse the LLM’s output into a Python list
        text = chat_resp.content.strip()
        try:
            sub_questions = ast.literal_eval(text)
            if not isinstance(sub_questions, list):
                raise ValueError("Not a list")
            # Ensure every item is a string
            sub_questions = [str(q).strip() for q in sub_questions]
        except Exception:
            # Fallback: treat the entire query as one sub-question
            sub_questions = [original_query]

        return sub_questions


def build(llm_client: Any) -> DecomposerAgent:
    """
    Factory function so you can do:
      from decomposer_agent import build as build_decomposer
      decomposer = build_decomposer(cfg.llm_client)
    """
    return DecomposerAgent(llm_client)
