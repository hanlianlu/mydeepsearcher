"""
Naive-RAG flow – Autogen 0.5.7
Retrieve → WebSearch* → Summarize
"""
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow

from deepsearcher.agents.retrieval_agent   import RetrievalAgent
from deepsearcher.agents.web_search_agent  import build as build_web
from deepsearcher.agents.summarizer_agent  import SummarizerAgent


# -------------------------------------------------------------------- #
def build(ctx) -> GraphFlow:
    """One-shot RAG: DB retrieval (+optional web) then summarise."""
    ret  = RetrievalAgent(
        vector_db   = ctx.vector_db,
        embed_model = ctx.embedding_model,
        top_k       = 10,
    )

    web  = build_web(ctx)             # may be a No-Op stub if disabled
    summ = SummarizerAgent(ctx.llm_client)

    # ----------------- build directed graph -------------------------- #
    b = DiGraphBuilder()
    for agent in (ret, web, summ):
        b.add_node(agent)

    b.add_edge(ret, web)
    b.add_edge(web, summ)

    graph = b.build()

    return GraphFlow(participants=[ret, web, summ],
                     graph=graph)
