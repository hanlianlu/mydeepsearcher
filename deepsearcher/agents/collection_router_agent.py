# deepsearcher/agents/collection_router_agent.py
from __future__ import annotations
from typing import Any, Dict, List
from autogen_agentchat.agents   import AssistantAgent
from autogen_agentchat.messages import TextMessage as Message

from deepsearcher.utils.rag_prompts    import COLLECTION_ROUTE_PROMPT
from deepsearcher.vector_db.base       import BaseVectorDB
from deepsearcher.utils.autogen_helper import kw_for_assistant_agent
from deepsearcher.agent.base           import describe_class


@describe_class(
    "CollectionRouterAgent decides which vector-DB collections are most "
    "relevant for the current question.  It returns "
    "`{'collections': List[str]}`."
)
class CollectionRouterAgent(AssistantAgent):

    def __init__(
        self,
        llm_client: Any,
        vector_db:  BaseVectorDB,
        *,
        name: str = "collection_router",
    ):
        _cfg_key = kw_for_assistant_agent()               # str | None

        init_kwargs = {
            "name"        : name,
            "model_client": llm_client,
        }
        if _cfg_key:                                      # Only if supported
            init_kwargs[_cfg_key] = {"temperature": 0.25}

        super().__init__(**init_kwargs)

        self._vector_db = vector_db

    # ------------------------------------------------------------------ #
    async def a_receive(               # (same signature as other agents)
        self,
        messages: List[Message],
        sender:   "AssistantAgent",
        config:   Dict | None = None,
    ) -> Message:

        pay        = messages[-1].content
        question   = pay.get("original_query") or messages[-1].content
        curr_iter  = pay.get("curr_iter", 1)
        iter_str   = "first" if curr_iter <= 1 else "subsequent"

        coll_infos = self._vector_db.list_collections()
        prompt = COLLECTION_ROUTE_PROMPT.format(
            question        = question,
            collection_info = [
                {
                    "collection_name": ci.collection_name,
                    "collection_description": getattr(ci, "description", ""),
                }
                for ci in coll_infos
            ],
            curr_iter_str   = iter_str,
        )

        llm_reply = await self.model_client.chat_async(
            [{"role": "user", "content": prompt}]
        )

        # --- parse ----------------------------------------------------- #
        selected: List[str]
        try:
            import ast
            selected = ast.literal_eval(llm_reply.content)
            if not isinstance(selected, list):
                raise ValueError
        except Exception:
            # fallback: all on first pass, none on later passes
            selected = [ci.collection_name for ci in coll_infos] if iter_str == "first" else []

        # always deduplicate + stringify
        selected = list({str(s).strip() for s in selected})

        return self.send(
            content  = {"collections": selected},
            metadata = {"total_tokens": llm_reply.total_tokens},
            sender   = self,
        )


# ---------------------------------------------------------------------- #
def build(llm_client: Any, vector_db: BaseVectorDB) -> CollectionRouterAgent:
    """Factory helper used by pipeline builders."""
    return CollectionRouterAgent(llm_client, vector_db)
