from __future__ import annotations
from typing import Union, List, Any

from autogen_agentchat.agents   import AssistantAgent
from autogen_agentchat.messages import TextMessage as Message
from deepsearcher.agent.base    import describe_class
from deepsearcher.utils.rag_prompts import MONO_INTERMEDIATE_PROMPT


@describe_class("Answers each sub-query strictly from its retrieved documents.")
class IntermediateAnswerAgent(AssistantAgent):
    def __init__(self, llm: Any, name: str = "intermediate_answer"):
        super().__init__(name=name, model_client=None)
        self._llm = llm

    async def run(self, task: Union[dict, Message], **kw) -> Message:
        data   = task.content if isinstance(task, Message) else task
        prompt = MONO_INTERMEDIATE_PROMPT.format(
            retrieved_documents = "\n".join(data.get("docs", [])) or "No documents.",
            sub_query           = data["sub_query"],
        )
        resp = await self._llm.chat_async([{"role": "user", "content": prompt}])
        return Message(role="assistant", content=resp.content.strip())


# ----------------------------------------------------------------------
def build(cfg) -> "IntermediateAnswerAgent":
    return IntermediateAnswerAgent(
        llm  = cfg.llm_client,
        name = getattr(cfg, "agent_name", "intermediate_answer"),
    )
