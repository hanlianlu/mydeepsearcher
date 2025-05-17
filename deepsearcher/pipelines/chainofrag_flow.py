"""
Chain-of-RAG flow – Autogen 0.5.7
Decompose → Retrieve → WebSearch* → Rerank → Reflect(loop) → Summarize
"""
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow

from deepsearcher.agents.decomposer_agent import DecomposerAgent
from deepsearcher.agents.retrieval_agent import RetrievalAgent
from deepsearcher.agents.web_search_agent import build as build_web
from deepsearcher.agents.reranker_agent import RerankerAgent
from deepsearcher.agents.reflection_agent import ReflectionAgent
from deepsearcher.agents.summarizer_agent import SummarizerAgent

# Condition functions for edge transitions
def has_new_queries(msg):
    """Check if the message content has new queries."""
    return bool(msg.content.get("new_queries"))

def no_new_queries(msg):
    """Check if the message content has no new queries."""
    return not msg.content.get("new_queries")

# Factory
def build(ctx, use_web: bool = False) -> GraphFlow:
    """Build and return a Chain-of-RAG GraphFlow with optional web search.

    Args:
        ctx: Runtime context containing LLM client, vector DB, etc.
        use_web: Whether to include web search in the flow.

    Returns:
        Configured GraphFlow instance.
    """
    # Initialize agents with context parameters
    dec = DecomposerAgent(ctx.llm_client)
    ret = RetrievalAgent(
        vector_db=ctx.vector_db,
        embed_model=ctx.embedding_model,
        top_k=9,
    )
    rer = RerankerAgent(ctx.llm_client)
    refl = ReflectionAgent(
        ctx.llm_client,
        max_iter=ctx.config.query_settings.get("max_iter", 6),
        confidence_threshold=0.92,
    )
    summ = SummarizerAgent(ctx.llm_client)

    # Build the directed graph
    b = DiGraphBuilder()
    for agent in (dec, ret, rer, refl, summ):
        b.add_node(agent)

    # Define the flow sequence
    b.add_edge(dec, ret)
    if use_web:
        web = build_web(ctx)
        b.add_node(web)
        b.add_edge(ret, web)
        b.add_edge(web, rer)
    else:
        b.add_edge(ret, rer)

    b.add_edge(rer, refl)
    # Note: Ensure ReflectionAgent sets "new_queries" in msg.content
    b.add_edge(refl, ret, condition=has_new_queries)  # Loop back if new queries
    b.add_edge(refl, summ, condition=no_new_queries)  # Finish if no new queries

    # Build and return the GraphFlow
    graph = b.build()
    participants = [dec, ret, rer, refl, summ]
    if use_web:
        participants.append(web)
    return GraphFlow(participants=participants, graph=graph)