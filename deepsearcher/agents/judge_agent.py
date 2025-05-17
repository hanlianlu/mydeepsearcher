# deepsearcher/agents/judge_agent.py
"""
FinalPaperJudgeAgent
====================

Inspects the user’s query (+ optional history) and decides whether the
answer should be a *long-form, thesis-style* “Final Paper” or an ordinary
summary.

Returned message
----------------
content  : { "use_final_paper": bool }
metadata : { "raw_reply": str, "total_tokens": int }
"""

from __future__ import annotations
from typing import Any, Dict, List

from autogen_agentchat.agents   import AssistantAgent
from autogen_agentchat.messages import TextMessage as Message

from deepsearcher.utils.autogen_helper import kw_for_assistant_agent
from deepsearcher.agent.base           import describe_class


# ────────────────────────────────────────────────────────────────────
JUDGMENT_PROMPT = """
Analyze whether the user truely expects a **long-form / thesis-level** report
rather than a normal comprehensive answer.

Query:
{query}

History Context:
{history_context}

Respond with **YES** (long report) or **NO** (normal summary)—nothing
else.
""".strip()


@describe_class(
    "FinalPaperJudgeAgent looks at the query and decides if the response "
    "should be a comprehensive, thesis-style report (YES) or a regular "
    "summary (NO)."
)
class FinalPaperJudgeAgent(AssistantAgent):
    """
    The agent simply returns a boolean decision inside `content`.
    """

    def __init__(
        self,
        llm_client: Any,
        *,
        name: str = "judge_finalpaper",
    ):
        # Figure out the correct kw-name for model-config on this Autogen build
        cfg_key = kw_for_assistant_agent()          # str | None

        kwargs = {
            "name"        : name,
            "model_client": llm_client,
        }
        if cfg_key:                                 # only if the backend expects it
            kwargs[cfg_key] = {"temperature": 0.25, "cache_seed": 42}

        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    async def a_receive(               # message-driven callback
        self,
        messages: List[Message],
        sender:   "AssistantAgent",
        config:   Dict | None = None,
    ) -> Message:

        pay     = messages[-1].content
        query   = pay.get("original_query", "") or messages[0].content
        history = pay.get("history", "")

        prompt  = JUDGMENT_PROMPT.format(
            query=query, history_context=history
        )

        llm_rsp = await self.model_client.chat_async(
            [{"role": "user", "content": prompt}]
        )

        decision = llm_rsp.content.strip().upper()
        use_final = decision == "YES"

        return self.send(
            content = {"use_final_paper": use_final},
            metadata= {
                "raw_reply"   : llm_rsp.content,
                "total_tokens": llm_rsp.total_tokens,
            },
            sender  = self,
        )


# ----------------------------------------------------------------------
def build(llm_client: Any) -> FinalPaperJudgeAgent:
    """Factory helper used by pipeline builders."""
    return FinalPaperJudgeAgent(llm_client)
