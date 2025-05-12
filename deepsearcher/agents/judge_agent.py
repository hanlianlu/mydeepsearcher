# deepsearcher/agents/judge_agent.py
"""
FinalPaperJudgeAgent: decides whether to use FinalPaperAgent or SummarizerAgent
based on the original query and history context.
"""
import ast
from typing import Any, List, Dict
from autogen_agentchat.agents import AssistantAgent

# Prompt to choose between detailed thesis-level report or regular summary
JUDGMENT_PROMPT = """
Analyze the query and history context to determine if a long-form/thesis-level report is truely expected and necessary.
Query: {query}
History Context: {history_context}

Respond with exactly 'YES' or 'NO' (case-insensitive), and nothing else.
"""

class FinalPaperJudgeAgent(AssistantAgent):
    def __init__(self, llm_client: Any):
        super().__init__(
            name="finalpaper_judge",
            model_client=llm_client,
            system_message=JUDGMENT_PROMPT,
            llm_config={"temperature": 0.0},  # deterministic choice
        )

    async def run(
        self,
        messages: List[Any],
        sender: str,
        config: Dict[str, Any],
    ) -> bool:
        # messages[0].content: original user query
        query = messages[0].content
        history = config.get("history_context", "")

        # Fill prompt
        prompt = JUDGMENT_PROMPT.format(query=query, history_context=history)
        chat_resp = await self.model_client.chat_async(
            [{"role": "user", "content": prompt}]
        )
        text = chat_resp.content.strip().upper()
        # Return True if 'YES', else False
        return text == "YES"


def build(llm_client: Any) -> FinalPaperJudgeAgent:
    """Factory: create FinalPaperJudgeAgent bound to llm_client."""
    return FinalPaperJudgeAgent(llm_client)
