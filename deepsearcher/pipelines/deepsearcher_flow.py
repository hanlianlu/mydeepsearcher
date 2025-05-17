"""
DeepSearcher micro-flow (Autogen 0.5.7 compliant)
-------------------------------------------------
Decompose → Router → Retrieve → WebSearch* → Rerank
         → Reflect(loop) → Judge → {Summarize | FinalPaper}
"""
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow

from deepsearcher.agents.decomposer_agent import DecomposerAgent
from deepsearcher.agents.collection_router_agent import CollectionRouterAgent
from deepsearcher.agents.retrieval_agent import RetrievalAgent
from deepsearcher.agents.web_search_agent import build as build_web
from deepsearcher.agents.reranker_agent import RerankerAgent
from deepsearcher.agents.reflection_agent import ReflectionAgent
from deepsearcher.agents.judge_agent import FinalPaperJudgeAgent
from deepsearcher.agents.summarizer_agent import SummarizerAgent
from deepsearcher.agents.final_paper_agent import build as build_fp

# Condition functions for edge transitions
def has_new_queries(msg):
    return bool(msg.content.get("new_queries"))

def no_new_queries(msg):
    return not msg.content.get("new_queries")

def needs_final_paper(msg):
    return msg.content.get("use_final_paper", False)

def no_final_paper(msg):
    return not msg.content.get("use_final_paper", False)

def build(ctx, use_web: bool = False):
    # Initialize agents with context parameters
    dec = DecomposerAgent(ctx.llm_client)
    router = CollectionRouterAgent(ctx.llm_client, ctx.vector_db)
    ret_db = RetrievalAgent(ctx.vector_db, ctx.embedding_model, top_k=9)
    rer = RerankerAgent(ctx.llm_client)
    refl = ReflectionAgent(
        ctx.llm_client,
        max_iter=ctx.config.query_settings.get("max_iter", 6),
        confidence_threshold=0.92,
    )
    judge = FinalPaperJudgeAgent(ctx.llm_client)
    summ = SummarizerAgent(ctx.llm_client)
    fp = build_fp(ctx.lightllm or ctx.llm_client, ctx.backupllm or ctx.llm_client)

    # Build the directed graph
    b = DiGraphBuilder()
    # Add core agents to the graph
    for a in (dec, router, ret_db, rer, refl, judge, summ, fp):
        b.add_node(a)

    # Define the flow sequence
    b.add_edge(dec, router)
    b.add_edge(router, ret_db)

    # Conditionally include WebSearchAgent
    if use_web:
        web = build_web(ctx)
        b.add_node(web)
        b.add_edge(ret_db, web)
        b.add_edge(web, rer)
    else:
        b.add_edge(ret_db, rer)

    b.add_edge(rer, refl)
    # Conditional edges from ReflectionAgent
    b.add_edge(refl, router, condition=has_new_queries)  # Loop back if new queries
    b.add_edge(refl, judge, condition=no_new_queries)    # Proceed if no new queries
    # Conditional edges from FinalPaperJudgeAgent
    b.add_edge(judge, fp, condition=needs_final_paper)   # Final paper if needed
    b.add_edge(judge, summ, condition=no_final_paper)    # Summarize otherwise

    # Build and return the GraphFlow
    graph = b.build()
    return GraphFlow(participants=[dec, router, ret_db, rer, refl, judge, summ, fp], graph=graph)