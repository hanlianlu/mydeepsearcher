from __future__ import annotations
from typing import Union, Any, List

from autogen_agentchat.agents   import AssistantAgent
from autogen_agentchat.messages import TextMessage as Message
from deepsearcher.agent.base    import describe_class
from deepsearcher.utils.rag_prompts import MONO_FINAL_PROMPT


@describe_class("Produces the two-section Markdown report (factual + insights).")
class FinalAnswerAgent(AssistantAgent):
    def __init__(self, llm: Any, name: str = "final_answer"):
        super().__init__(name=name, model_client=None)
        self._llm = llm

    async def run(self, task: Union[dict, Message], **kw) -> Message:
        d      = task.content if isinstance(task, Message) else task
        prompt = MONO_FINAL_PROMPT.format(
            retrieved_documents = "\n".join(d.get("retrieved_documents", [])),
            intermediate_context= "\n".join(d.get("intermediate_context", [])),
            query               = d.get("query", ""),
            history_context     = d.get("history_context", ""),
        )
        resp = await self._llm.chat_async([{"role": "user", "content": prompt}])
        return Message(role="assistant", content=resp.content)


# ----------------------------------------------------------------------
def build(cfg) -> "FinalAnswerAgent":
    return FinalAnswerAgent(
        llm  = cfg.llm_client,
        name = getattr(cfg, "agent_name", "final_answer"),
    )
