from __future__ import annotations
from typing import Union, Any

from autogen_agentchat.agents   import AssistantAgent
from autogen_agentchat.messages import TextMessage as Message
from deepsearcher.agent.base    import describe_class
from deepsearcher.utils.rag_prompts import MONO_CONFIDENCE_PROMPT


@describe_class("Estimates a 0-1 confidence that enough info is gathered.")
class ConfidenceAgent(AssistantAgent):
    def __init__(self, llm: Any, name: str = "confidence"):
        super().__init__(name=name, model_client=None)
        self._llm = llm

    async def run(self, task: Union[dict, Message], **kw) -> Message:
        d      = task.content if isinstance(task, Message) else task
        prompt = MONO_CONFIDENCE_PROMPT.format(
            query               = d["query"],
            intermediate_context= "\n".join(d.get("intermediate_context", [])),
        )
        resp = await self._llm.chat_async([{"role": "user", "content": prompt}])
        try:
            score = float(resp.content.strip())
        except Exception:
            score = 0.0
        return Message(role="assistant", content={"confidence": score})


# ----------------------------------------------------------------------
def build(cfg) -> "ConfidenceAgent":
    return ConfidenceAgent(
        llm  = cfg.llm_client,
        name = getattr(cfg, "agent_name", "confidence"),
    )
