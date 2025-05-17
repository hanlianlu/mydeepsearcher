"""
RAGRouterAgent
==============

Top-level *meta-orchestrator* that chooses among three RAG pipelines:

  1 → DeepSearchGraphFlow
  2 → NaiveRAGGraphFlow
  3 → ChainOfRAGGraphFlow

It is **task-driven** (implements `run(task=...)`) because it will be
invoked directly by your CLI / server, not as a node inside another
GraphFlow.
"""
from __future__ import annotations
import importlib, logging, re
from typing import Any, List

from autogen_agentchat.agents   import AssistantAgent
from autogen_agentchat.messages import TextMessage   # v0.5.7+

from deepsearcher.utils.rag_prompts import RAG_ROUTER_PROMPT

logger = logging.getLogger(__name__)

_PIPELINE_REGISTRY = {
    "1": "deepsearcher.pipelines.deepsearch_flow.build",
    "2": "deepsearcher.pipelines.naiverag_flow.build",
    "3": "deepsearcher.pipelines.chainofrag_flow.build",
}

_DESCRIPTION_CACHE: dict[str, str] = {}   # filled by caller


class RAGRouterAgent(AssistantAgent):
    def __init__(
        self,
        llm_client: Any,
        *,
        descriptions: List[str],
        name: str = "rag_router",
    ):
        super().__init__(name=name, model_client=llm_client)
        # stash descriptions for prompt render
        for i, desc in enumerate(descriptions, start=1):
            _DESCRIPTION_CACHE[str(i)] = desc

    # ------------------------------------------------------------------ #
    # task-driven entry point
    # ------------------------------------------------------------------ #
    async def run(self, task: str, ctx, **_) -> str:
        """
        Parameters
        ----------
        task : str    # the user query
        ctx  : object # RuntimeContext (llm, vector_db, embedding, config)

        Returns
        -------
        str : final answer produced by the chosen pipeline
        """
        choice_idx = await self._choose_pipeline(task)
        build_fn   = self._import_build(_PIPELINE_REGISTRY[choice_idx])

        # Build / run selected GraphFlow
        flow = build_fn(ctx)
        result_msg = flow.run(task=TextMessage(role="user", content=task))
        return result_msg.content   # Markdown answer

    # ------------------------------------------------------------------ #
    # internal helpers
    # ------------------------------------------------------------------ #
    async def _choose_pipeline(self, query: str) -> str:
        desc_block = "\n".join(f"[{k}]: {v}" for k, v in _DESCRIPTION_CACHE.items())
        prompt = RAG_ROUTER_PROMPT.format(query=query, description_str=desc_block)

        reply = await self.model_client.chat_async([{"role": "user", "content": prompt}])
        text  = reply.content.strip()

        # First digit 1-3 present in LLM response wins
        match = re.search(r"[1-3]", text)
        choice = match.group(0) if match else "1"
        logger.info("RAGRouter chose pipeline %s", choice)
        return choice

    def _import_build(self, dotted_path: str):
        mod_name, _, func_name = dotted_path.rpartition(".")
        module = importlib.import_module(mod_name)
        return getattr(module, func_name)


# ---------------------------------------------------------------------- #
# convenience factory
# ---------------------------------------------------------------------- #
def build(llm_client: Any, agent_descriptions: List[str]) -> RAGRouterAgent:
    """
    Factory used by `orchestration.master_flow`.
    `agent_descriptions` should come from the `__description__`
    attribute of DeepSearch, NaiveRAG, ChainOfRAG.
    """
    return RAGRouterAgent(llm_client, descriptions=agent_descriptions)
