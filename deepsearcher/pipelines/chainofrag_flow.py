# deepsearcher/pipelines/chainofrag_flow.py
# ---------------------------------------------------------------------------
# Chain-of-RAG flow – AutoGen v0.5.7
# Steps:
#   1. Decompose → 2. Retrieve → 3. (WebSearch*) → 4. (Rerank*) 
#   → 5. IntermediateAnswer → 6. Followup → 7. Confidence → 8. FinalAnswer
# ---------------------------------------------------------------------------

from autogen_agentchat.teams import DiGraphBuilder, GraphFlow

# Agent factories (each ends in build(cfg))
from deepsearcher.agents.decomposer_agent          import build as build_decomposer
from deepsearcher.agents.retrieval_agent          import build as build_retrieval
from deepsearcher.agents.web_search_agent         import build as build_websearch
from deepsearcher.agents.reranker_agent           import build as build_reranker
from deepsearcher.agents.intermediate_answer_agent import build as build_intermediate
from deepsearcher.agents.followup_agent           import build as build_followup
from deepsearcher.agents.cor_confidence_agent         import build as build_confidence
from deepsearcher.agents.cor_summarizer_agent       import build as build_final_answer


def build(ctx, use_web: bool = False, use_rerank: bool = False) -> GraphFlow:
    """
    Build a Chain-of-RAG GraphFlow with precise monolith-style steps:

      1. Decompose
      2. Retrieve
      3. Optional WebSearch
      4. Optional Rerank
      5. IntermediateAnswer (per sub-query)
      6. Followup (generate new sub-queries)
      7. Confidence (early-stop check)
      8. FinalAnswer (two-section Markdown)

    Args:
      ctx:         Runtime context (llm_client, vector_db, embedding_model, config, etc.)
      use_web:     If True, include the WebSearch step after Retrieval.
      use_rerank:  If True, include the Reranker step after (WebSearch or Retrieval).

    Returns:
      Configured GraphFlow.
    """

    # 1️⃣ Instantiate each agent via its build(cfg)
    dec         = build_decomposer(ctx)
    ret         = build_retrieval(ctx)
    web         = build_websearch(ctx)    if use_web   else None
    rer         = build_reranker(ctx)     if use_rerank else None
    inter_ans   = build_intermediate(ctx)
    follow      = build_followup(ctx)
    confidence  = build_confidence(ctx)
    final       = build_final_answer(ctx)

    # 2️⃣ Early-stop threshold from your config (monolith uses ~0.91 by default)
    threshold = ctx.config.query_settings.get("confidence_threshold", 0.91)

    def low_confidence(msg):
        return msg.content.get("confidence", 0.0) < threshold

    def high_confidence(msg):
        return msg.content.get("confidence", 0.0) >= threshold

    # 3️⃣ Build the directed graph
    builder = DiGraphBuilder()

    # Add every node
    builder.add_node(dec)
    builder.add_node(ret)
    if web:        builder.add_node(web)
    if rer:        builder.add_node(rer)
    builder.add_node(inter_ans)
    builder.add_node(follow)
    builder.add_node(confidence)
    builder.add_node(final)

    # Edges: Decompose → Retrieve
    builder.add_edge(dec, ret)

    # Retrieve → [WebSearch →] [Rerank →] IntermediateAnswer
    prev = ret
    if web:
        builder.add_edge(prev, web); prev = web
    if rer:
        builder.add_edge(prev, rer); prev = rer
    builder.add_edge(prev, inter_ans)

    # IntermediateAnswer → Followup
    builder.add_edge(inter_ans, follow)

    # Followup → Confidence → (loop or finish)
    builder.add_edge(follow, confidence)
    builder.add_edge(confidence, ret,   condition=low_confidence)  # Loop for more info
    builder.add_edge(confidence, final, condition=high_confidence) # Done when confident

    # 4️⃣ Final assembly
    graph = builder.build()
    participants = [dec, ret] \
                 + ([web] if web else []) \
                 + ([rer] if rer else []) \
                 + [inter_ans, follow, confidence, final]

    return GraphFlow(participants=participants, graph=graph)
