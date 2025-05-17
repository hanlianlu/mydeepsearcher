"""
Chain-of-RAG flow – Autogen 0.5.7
Decompose → Retrieve → WebSearch* → Rerank → Reflect(loop) → Summarize
"""
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow

from deepsearcher.agents.decomposer_agent  import DecomposerAgent
from deepsearcher.agents.retrieval_agent   import RetrievalAgent
from deepsearcher.agents.web_search_agent  import build as build_web
from deepsearcher.agents.reranker_agent    import RerankerAgent
from deepsearcher.agents.reflection_agent  import ReflectionAgent
from deepsearcher.agents.summarizer_agent  import SummarizerAgent


# -------------------------------------------------------------------- #
# Helper names referenced by DiGraphBuilder (must be **global strings**)
# -------------------------------------------------------------------- #
def _has_new(msg): return bool(msg.content.get("new_queries"))
def _no_new(msg):  return not msg.content.get("new_queries")


# -------------------------------------------------------------------- #
# Factory
# -------------------------------------------------------------------- #
def build(ctx) -> GraphFlow:
    """Return a ready-to-run GraphFlow instance."""
    dec  = DecomposerAgent(ctx.llm_client)

    ret  = RetrievalAgent(
        vector_db   = ctx.vector_db,
        embed_model = ctx.embedding_model,
        top_k       = 9,
    )

    web  = build_web(ctx)                 # might be a No-Op stub
    rer  = RerankerAgent(ctx.llm_client)

    refl = ReflectionAgent(
        ctx.llm_client,
        max_iter             = ctx.config.query_settings.get("max_iter", 6),
        confidence_threshold = 0.92,
    )

    summ = SummarizerAgent(ctx.llm_client)

    # ----------------- build directed graph -------------------------- #
    b = DiGraphBuilder()
    for agent in (dec, ret, web, rer, refl, summ):
        b.add_node(agent)

    b.add_edge(dec, ret)
    b.add_edge(ret, web)
    b.add_edge(web, rer)
    b.add_edge(rer, refl)

    b.add_edge(refl, ret,  condition="_has_new")   # loop
    b.add_edge(refl, summ, condition="_no_new")    # finish

    graph = b.build()

    # Autogen requires *list*, not dict_values
    return GraphFlow(participants=[dec, ret, web, rer, refl, summ],
                     graph=graph)
