"""
DeepSearcher micro-flow (Autogen 0.5.7 compliant)
-------------------------------------------------
Decompose → Router → Retrieve → WebSearch* → Rerank
         → Reflect(loop) → Judge → {Summarize | FinalPaper}
"""
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow

from deepsearcher.agents.decomposer_agent        import DecomposerAgent
from deepsearcher.agents.collection_router_agent import CollectionRouterAgent
from deepsearcher.agents.retrieval_agent         import RetrievalAgent
from deepsearcher.agents.web_search_agent        import build as build_web
from deepsearcher.agents.reranker_agent          import RerankerAgent
from deepsearcher.agents.reflection_agent        import ReflectionAgent
from deepsearcher.agents.judge_agent             import FinalPaperJudgeAgent
from deepsearcher.agents.summarizer_agent        import SummarizerAgent
from deepsearcher.agents.final_paper_agent       import build as build_fp


# ------------------------------------------------------------------ #
# string conditions — must reference callables available at runtime
# ------------------------------------------------------------------ #
def _has_new(msg):      return bool(msg.content.get("new_queries"))
def _no_new(msg):       return not msg.content.get("new_queries")
def _need_fp(msg):      return     msg.content["use_final_paper"]
def _no_fp(msg):        return not msg.content["use_final_paper"]


def build(ctx):
    dec    = DecomposerAgent(ctx.llm_client)
    router = CollectionRouterAgent(ctx.llm_client, ctx.vector_db)
    ret_db = RetrievalAgent(ctx.vector_db, ctx.embedding_model, top_k=9)
    web    = build_web(ctx)
    rer    = RerankerAgent(ctx.llm_client)
    refl   = ReflectionAgent(
        ctx.llm_client,
        max_iter=ctx.config.query_settings.get("max_iter", 6),
        confidence_threshold=0.92,
    )
    judge  = FinalPaperJudgeAgent(ctx.llm_client)
    summ   = SummarizerAgent(ctx.llm_client)
    fp     = build_fp(ctx.lightllm or ctx.llm_client,
                      ctx.backupllm or ctx.llm_client)

    # -------- graph ---------------------------------------------------
    b = DiGraphBuilder()
    for a in (dec, router, ret_db, web, rer, refl, judge, summ, fp):
        b.add_node(a)

    b.add_edge(dec,    router)
    b.add_edge(router, ret_db)
    b.add_edge(ret_db, web)
    b.add_edge(web,    rer)
    b.add_edge(rer,    refl)

    b.add_edge(refl, router, condition="_has_new")
    b.add_edge(refl, judge,  condition="_no_new")

    b.add_edge(judge, fp,   condition="_need_fp")   # FP leaf
    b.add_edge(judge, summ, condition="_no_fp")     # Summ leaf
    graph = b.build()
    return GraphFlow(participants=[dec, router, ret_db, web, rer, refl,
                                   judge, fp, summ],
                     graph=graph)
