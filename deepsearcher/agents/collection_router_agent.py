# deepsearcher/agents/router_agent.py

import ast
from typing import Any, List, Dict
from autogen_agentchat.agents import AssistantAgent
from deepsearcher.utils.rag_prompts import COLLECTION_ROUTE_PROMPT
from deepsearcher.vector_db.base import BaseVectorDB

class CollectionRouterAgent(AssistantAgent):
    """
    Chooses which vectorDB collections to query for a given question.
    """

    def __init__(self, llm_client: Any, vector_db: BaseVectorDB):
        super().__init__(
            name="router",
            model_client=llm_client,
            system_message=COLLECTION_ROUTE_PROMPT,
        )
        self.vector_db = vector_db

    async def run(
        self,
        messages: List[Any],
        sender: str,
        config: Dict[str, Any],
    ) -> Dict[str, List[str]]:
        # 1) Extract the original question
        question = messages[0].content

        # 2) Determine iteration string
        curr_iter = config.get("curr_iter", 1)
        curr_iter_str = "first" if curr_iter <= 1 else "subsequent"

        # 3) Gather available collections info
        infos = self.vector_db.list_collections()
        coll_info = [
            {
                "collection_name": ci.collection_name,
                "collection_description": getattr(ci, "description", ""),
            }
            for ci in infos
        ]

        # 4) Fill the prompt
        prompt = COLLECTION_ROUTE_PROMPT.format(
            question=question,
            collection_info=coll_info,
            curr_iter_str=curr_iter_str,
        )

        # 5) Call the LLM synchronously
        chat_resp = await self.model_client.chat_async(
            [{"role": "user", "content": prompt}]
        )

        # 6) Parse the Python list of names
        try:
            selected = ast.literal_eval(chat_resp.content)
        except Exception:
            # Fallback: first‐iteration → all; else none
            if curr_iter_str == "first":
                selected = [ci["collection_name"] for ci in coll_info]
            else:
                selected = []
        # Ensure it’s a list of strings
        selected = [str(x) for x in selected if isinstance(x, str)]

        return {"collections": selected}


def build(llm_client: Any, vector_db: BaseVectorDB) -> CollectionRouterAgent:
    """
    Factory: create a RouterAgent bound to the given LLM and vectorDB.
    """
    return CollectionRouterAgent(llm_client, vector_db)
