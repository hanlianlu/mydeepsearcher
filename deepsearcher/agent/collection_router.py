"""
CollectionRouter  -- robust, dual-mode (sync + async) version
-------------------------------------------------------------

Will return a tuple: (selected_collection_names: list[str], token_cost: int)
no matter how you call it:

    # ❶ async context
    selected, tokens = await router.invoke(question)

    # ❷ sync context
    selected, tokens = router.invoke(question)
"""
from __future__ import annotations

import asyncio, logging
from typing import List, Tuple

from deepsearcher.agent.base   import BaseAgent
from deepsearcher.llm.base     import BaseLLM
from deepsearcher.vector_db.base import BaseVectorDB
from deepsearcher.tools        import log      # keeps your colour print

logger = logging.getLogger(__name__)

COLLECTION_ROUTE_PROMPT = """
You are provided with a QUESTION, a COLLECTION_INFO containing collection_name(s)
and their corresponding collection_description(s), and the Current Iteration.

Your task is to:
1. Identify all collection_name(s) relevant to answering the QUESTION.
2. Strictly IGNORE any information provided by the current iteration UNLESS
   Current Iteration is exactly "first".

Rules you MUST follow:
- If Current Iteration is "first" AND you find NO relevant collections to answer
  the question, you MUST return ALL available collection_names as a *Python list
  of strings*.

Return **only** that Python list: no prose, no extra keys.

QUESTION: {question}
COLLECTION_INFO: {collection_info}
Current Iteration: {curr_iter_str}

Your selected collection name list is:
"""


class CollectionRouter(BaseAgent):
    def __init__(self, llm: BaseLLM, vector_db: BaseVectorDB,*_, **__):
        self.llm        = llm
        self.vector_db  = vector_db

        # Preserve original list for fast fallback
        try:
            self.all_collections = [
                c.collection_name for c in self.vector_db.list_collections()
            ]
        except Exception as e:
            logger.warning("VectorDB list_collections() failed: %s", e)
            self.all_collections = []

    # ------------------------------------------------------------------ #
    # Public entry-point – works sync OR async
    # ------------------------------------------------------------------ #
    def invoke(
        self,
        query: str,
        collections_names: list[str] | None = None,
        curr_iter: int = 1,
    ) -> Tuple[List[str], int] | "asyncio.Future":
        """
        • If called inside an event-loop, returns a *coroutine* that you must await.
        • If called from plain sync code, runs the coroutine to completion and
          returns the result directly.
        """
        coro = self._ainvoke(query, collections_names, curr_iter)

        try:
            loop = asyncio.get_running_loop()
            # we're already in async code → give coroutine back to caller
            return coro  # type: ignore[return-value]
        except RuntimeError:
            # no running loop → run to completion ourselves
            return asyncio.run(coro)

    # ------------------------------------------------------------------ #
    # True async implementation
    # ------------------------------------------------------------------ #
    async def _ainvoke(
        self,
        query: str,
        collections_names: list[str] | None,
        curr_iter: int,
    ) -> Tuple[List[str], int]:

        curr_iter_str = "first" if curr_iter <= 1 else "subsequent"
        tok_cost      = 0

        # --- determine candidate collections --------------------------------
        try:
            if collections_names:
                infos = self.vector_db.list_collections(collections_names)
            else:
                infos = self.vector_db.list_collections()
        except Exception as e:
            logger.warning("VectorDB access failed (%s). Falling back to []", e)
            infos = []

        if not infos:                       # none at all → hard-fail graceful
            log.warning("No collections found in VectorDB; returning empty list")
            return [], 0

        if len(infos) == 1:                 # only one choice → short-circuit
            name = infos[0].collection_name
            log.color_print(f"<think> Only one collection [{name}] available. </think>\n")
            return [name], 0

        # --- build LLM prompt -----------------------------------------------
        prompt = COLLECTION_ROUTE_PROMPT.format(
            question        = query,
            collection_info = [
                {
                    "collection_name": ci.collection_name,
                    "collection_description": getattr(ci, "description", ""),
                }
                for ci in infos
            ],
            curr_iter_str   = curr_iter_str,
        )

        # --- call LLM (async if possible) -----------------------------------
        if hasattr(self.llm, "chat_async"):
            resp = await self.llm.chat_async([{"role": "user", "content": prompt}])
        else:
            resp = await asyncio.to_thread(
                self.llm.chat, messages=[{"role": "user", "content": prompt}]
            )
        tok_cost += getattr(resp, "total_tokens", 0)

        # --- parse output ----------------------------------------------------
        try:
            selected = eval(resp.content)
            if not isinstance(selected, list):
                raise ValueError
        except Exception:
            # fallback behaviour copied from your original code
            selected = (
                [ci.collection_name for ci in infos]
                if curr_iter_str == "first"
                else []
            )

        # Ensure unique, valid strings
        selected = list({str(c).strip() for c in selected if str(c).strip()})

        # Always add collections with empty description or default collection
        for ci in infos:
            if not getattr(ci, "description", "") or \
               ci.collection_name == getattr(self.vector_db, "default_collection", None):
                selected.append(ci.collection_name)

        selected = list(dict.fromkeys(selected))     # preserve order, dedupe

        log.color_print(
            f"<think> CollectionRouter → {selected} (iter: {curr_iter_str}) </think>\n"
        )
        return selected, tok_cost
