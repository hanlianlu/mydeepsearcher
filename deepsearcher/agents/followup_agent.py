from __future__ import annotations
from typing import Union, List, Any
import ast

from autogen_agentchat.agents   import AssistantAgent
from autogen_agentchat.messages import TextMessage as Message
from deepsearcher.agent.base    import describe_class
from deepsearcher.utils.rag_prompts import MONO_FOLLOWUP_PROMPT


@describe_class("Generates up to three follow-up queries to close info gaps.")
class FollowupAgent(AssistantAgent):
    def __init__(self, llm: Any, name: str = "followup"):
        super().__init__(name=name, model_client=None)
        self._llm = llm

    async def run(self, task: Union[dict, Message], **kw) -> Message:
        ctx   = task.content if isinstance(task, Message) else task
        prompt = MONO_FOLLOWUP_PROMPT.format(
            query               = ctx["query"],
            intermediate_context= "\n".join(ctx.get("intermediate_context", [])),
        )
        resp = await self._llm.chat_async([{"role": "user", "content": prompt}])
        try:
            followups: List[str] = ast.literal_eval(resp.content.strip())
            if not isinstance(followups, list):
                followups = []
        except Exception:
            followups = []
        return Message(role="assistant", content={"new_queries": followups})


# ----------------------------------------------------------------------
def build(cfg) -> "FollowupAgent":
    return FollowupAgent(
        llm  = cfg.llm_client,
        name = getattr(cfg, "agent_name", "followup"),
    )
