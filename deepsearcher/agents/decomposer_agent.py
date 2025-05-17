# deepsearcher/agents/decomposer_agent.py
from __future__ import annotations
import ast
from typing import Any, Dict, List

from autogen_agentchat.agents   import AssistantAgent
from autogen_agentchat.messages import TextMessage as Message

from deepsearcher.utils.rag_prompts import SUB_QUERY_PROMPT
from deepsearcher.agent.base        import describe_class
from deepsearcher.utils.autogen_helper import kw_for_assistant_agent


@describe_class(
    "DecomposerAgent splits complex questions into up to four focused "
    "sub-queries, improving retrieval recall while keeping each query "
    "self-contained."
)
class DecomposerAgent(AssistantAgent):
    def __init__(self, llm_client: Any, *, name: str = "decomposer"):
        _kw = kw_for_assistant_agent()          # str | None

        init_kwargs = {
            "name"        : name,
            "model_client": llm_client,
        }
        # only add the temperature/cache-seed block when a keyword exists
        if _kw:
            init_kwargs[_kw] = {"temperature": 0.25, "cache_seed": 42}

        super().__init__(**init_kwargs)

    async def a_receive(
        self,
        messages: List[Message],
        sender:   "AssistantAgent",
        config:   Dict | None = None,
    ) -> Message:
        pay         = messages[-1].content
        original_q  = pay.get("original_query") or messages[-1].content
        history     = pay.get("history", "")

        prompt = SUB_QUERY_PROMPT.format(
            original_query  = original_q,
            history_context = history,
        )

        llm_reply = await self.model_client.chat_async(
            [{"role": "user", "content": prompt}]
        )

        sub_qs = self._parse_list(llm_reply.content, fallback=[original_q])

        return self.send(
            content  = {
                "original_query": original_q,
                "sub_queries":    sub_qs,
                "history":        history,
            },
            metadata = {"total_tokens": llm_reply.total_tokens},
            sender   = self,
        )

    # helper
    @staticmethod
    def _parse_list(text: str, fallback: List[str]) -> List[str]:
        try:
            obj = ast.literal_eval(text.strip())
            if isinstance(obj, list) and all(isinstance(x, str) for x in obj):
                cleaned = [x.strip() for x in obj]
                return cleaned if cleaned else fallback
        except Exception:
            pass
        return fallback


def build(llm_client: Any) -> DecomposerAgent:
    return DecomposerAgent(llm_client)
