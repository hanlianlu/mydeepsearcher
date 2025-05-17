"""
SummarizerAgent
===============

Terminal node that turns everything collected by upstream agents
(sub-queries → retrieval → rerank → reflection) into a **Markdown**
answer.

Features
--------
* Works both in *micro-agent* GraphFlow and when called stand-alone.
* Graceful fall-back: if **zero** chunks arrive, the agent still answers
  (adds the classic “[Disclaimer: …]” banner).
* All runtime knobs are constructor params / factory kwargs – no magic
  globals.
"""
from __future__ import annotations

import asyncio, json
from typing import Any, Dict, List

from autogen_agentchat.agents   import AssistantAgent
from autogen_agentchat.messages import TextMessage            # official class

from deepsearcher.utils.rag_prompts import SUMMARY_PROMPT
from deepsearcher.agent.base        import describe_class
from deepsearcher.vector_db.base    import RetrievalResult


# ╭────────────────────────────────────────────────────────────────────╮
# │  Constants                                                         │
# ╰────────────────────────────────────────────────────────────────────╯
DISCLAIMER = (
    "\n\n[Disclaimer: No relevant data found! Answer solely by "
    "AI's prior knowledge and may be inaccurate.]\n\n"
)

# ╭────────────────────────────────────────────────────────────────────╮
# │  Helper – format chunks for the prompt                             │
# ╰────────────────────────────────────────────────────────────────────╯
def _format_chunks(chunks: List[RetrievalResult] | List[Dict]) -> str:
    """
    Accepts either the real `RetrievalResult` objects **or** the light-weight
    preview dicts produced by RetrievalAgent / WebSearchRetrievalAgent.
    """
    out = []
    for i, ch in enumerate(chunks):
        if isinstance(ch, dict):
            txt   = ch.get("text", "")
            meta  = ch.get("metadata", {})
        else:                                       # RetrievalResult
            txt   = ch.text
            meta  = ch.metadata
        out.append(
            f"<Document {i}>\n"
            f"Text: {txt}\n"
            f"Metadata: {json.dumps(meta, ensure_ascii=False)}\n"
            f"</Document {i}>"
        )
    return "\n".join(out)


# ╭────────────────────────────────────────────────────────────────────╮
# │  Agent implementation                                              │
# ╰────────────────────────────────────────────────────────────────────╯
@describe_class(
    "SummarizerAgent writes the final report.  If it receives no retrieved "
    "documents it still answers, but surrounds the response with a disclaimer."
)
class SummarizerAgent(AssistantAgent):
    """
    Expected **input message** (`messages[-1].content`)::

        {
            "original_query": str,
            "sub_queries":    list[str],
            "chunks":         list[RetrievalResult|dict],   # may be empty
            "history":        str | "",
        }

    Returned message’s **content** is Markdown; **metadata** includes
    ``{"total_tokens": int, "num_chunks": int}``.
    """

    # ──────────────────────────────────────────────────────────────────
    def __init__(
        self,
        llm_client: Any,
        *,
        name: str = "summarizer",
        top_n_chunks: int = 30
    ):
        super().__init__(name=name, model_client=llm_client)
        self.top_n_chunks = top_n_chunks
        self.llm_cfg      = {}

    # ──────────────────────────────────────────────────────────────────
    async def a_receive(                       # message-driven API
        self,
        messages: List[TextMessage],
        sender:   "AssistantAgent",
        config:   Dict | None = None,
    ) -> TextMessage:

        payload: Dict = messages[-1].content
        orig_q   = payload.get("original_query", "")
        sub_qs   = payload.get("sub_queries", [])
        chunks   = (payload.get("chunks") or [])[: self.top_n_chunks]
        history  = payload.get("history", "")

        # ── No chunks → fallback answer with disclaimer ────────────────
        if not chunks:
            prompt = (
                "You have *no* external documents.  Answer the question with "
                "your own knowledge.\n\n"
                f"Question: {orig_q}"
            )
            raw = await self.model_client.chat_async(
                [{"role": "user", "content": prompt}],
                **self.llm_cfg,
            )
            answer = f"{DISCLAIMER}{raw.content.strip()}{DISCLAIMER}"
            return self.send(
                content=answer,
                metadata={"total_tokens": raw.total_tokens, "num_chunks": 0},
                sender=self,
            )

        # ── Normal summarisation path ──────────────────────────────────
        prompt = SUMMARY_PROMPT.format(
            question        = orig_q,
            mini_questions  = "; ".join(sub_qs),
            mini_chunk_str  = _format_chunks(chunks),
            history_context = history,
        )

        raw = await self.model_client.chat_async(
            [{"role": "user", "content": prompt}],
            **self.llm_cfg,
        )

        return self.send(
            content = raw.content,
            metadata={
                "total_tokens": raw.total_tokens,
                "num_chunks":   len(chunks),
            },
            sender=self,
        )


# ╭────────────────────────────────────────────────────────────────────╮
# │  Factory helper                                                    │
# ╰────────────────────────────────────────────────────────────────────╯
def build(
    llm_client: Any,
    *,
    top_n_chunks: int = 30 ) -> SummarizerAgent:
    """
    Example
    -------
    ```python
    from deepsearcher.agents.summarizer_agent import build as build_summ
    summ = build_summ(cfg.llm_client, top_n_chunks=15)
    ```
    """
    return SummarizerAgent(
        llm_client     = llm_client,
        top_n_chunks   = top_n_chunks
    )
