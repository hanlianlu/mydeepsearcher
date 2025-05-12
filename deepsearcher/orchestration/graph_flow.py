# deepsearcher/orchestration/graph_flow.py

from typing import Any, List
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import GraphFlow

# RAG Router to choose pipeline
from deepsearcher.agents.rag_router_agent import build as build_rag_router

# DeepSearcher micro-agents
from deepsearcher.agents.decomposer_agent import build as build_decomposer
from deepsearcher.agents.collection_router_agent import build as build_collection_router
from deepsearcher.agents.retrieval_tool import make_retrieval_tool
from deepsearcher.agents.reranker_agent import build as build_reranker
from deepsearcher.agents.reflection_agent import build as build_reflection
from deepsearcher.agents.judge_agent import build as build_judge
from deepsearcher.agents.summarizer_agent import build as build_summarizer
from deepsearcher.agents.final_paper_agent import build as build_finalpaper

# Legacy pipelines
from deepsearcher.agents.naive_rag_agent import build as build_naive_agent
from deepsearcher.agents.chain_of_rag_agent import build as build_chain_agent

# Configuration
from deepsearcher.light_configuration import light_cfg as cfg


def build_flow() -> GraphFlow:
    """
    Overall RAG workflow:

      User
        → RAGRouterAgent (select 0/1/2)
           ├─ 0 → DeepSearcher multi-agent pipeline
           ├─ 1 → NaiveRAG monolith
           └─ 2 → ChainOfRAG monolith
    """
    # 1) RAGRouter selection
    descriptions = [
        "DeepSearcher multi-agent and all-around pipeline",
        "NaiveRAG simple pipeline",
        "ChainOfRAG precision pipeline",
    ]
    rag_router = build_rag_router(
        llm_client=cfg.llm_client,
        descriptions=descriptions,
        max_iter=cfg.max_iter,
    )

    # 2) Prepare DeepSearcher agents
    decomposer = build_decomposer(cfg.llm_client)
    collection_router = build_collection_router(cfg.llm_client, cfg.vector_db)
    vectordb_tool = make_retrieval_tool(cfg.vector_db, cfg.embedding_model)
    retrieval_caller = AssistantAgent(
        name="retrieval_caller",
        model_client=cfg.llm_client,
        tools=[vectordb_tool],
        system_message=(
            "Receive `sub_queries` and `collections`, call vectordb_search and return JSON."
        ),
        reflect_on_tool_use=False,
    )
    reranker = build_reranker(cfg.llm_client)
    reflection = build_reflection(cfg.llm_client, cfg.max_iter)
    judge = build_judge(cfg.llm_client)
    summarizer = build_summarizer(cfg.llm_client)
    finalpaper = build_finalpaper(cfg.lightllm, cfg.backupllm)

    # 3) Legacy pipelines
    naive_agent = build_naive_agent(cfg)
    chain_agent = build_chain_agent(
        cfg.llm_client, cfg.lightllm, cfg.backupllm, cfg.embedding_model, cfg.vector_db, cfg.max_iter
    )

    # 4) Assemble GraphFlow
    flow = GraphFlow(
        participants=[
            rag_router,
            # DeepSearcher branch
            decomposer, collection_router, retrieval_caller, reranker, reflection, judge, summarizer, finalpaper,
            # Legacy branches
            naive_agent, chain_agent,
        ]
    )

    @flow.on_start
    def start(msg: Any, state: Any):
        return rag_router

    # 5) Branch to the selected pipeline
    flow.connect(rag_router, decomposer,   condition=lambda msg: msg.content == 0)
    flow.connect(rag_router, naive_agent,  condition=lambda msg: msg.content == 1)
    flow.connect(rag_router, chain_agent,  condition=lambda msg: msg.content == 2)

    # 6) DeepSearcher subgraph
    flow.connect(decomposer,       collection_router)
    flow.connect(collection_router, retrieval_caller)
    flow.connect(retrieval_caller,  reranker)
    flow.connect(reranker,          reflection)

    # Loop back if new sub-queries
    flow.connect(
        reflection, collection_router, condition=lambda msg: bool(msg.content)
    )

    # After reflection, decide finalpaper vs summarizer
    flow.connect(
        reflection, judge, condition=lambda msg: not msg.content
    )
    flow.connect(judge, finalpaper,   condition=lambda msg: msg.content is True)
    flow.connect(judge, summarizer,   condition=lambda msg: msg.content is False)

    # End of DeepSearcher branch
    flow.connect(summarizer, flow.END)
    flow.connect(finalpaper, flow.END)

    # 7) Legacy pipelines terminate immediately
    flow.connect(naive_agent, flow.END)
    flow.connect(chain_agent, flow.END)

    return flow
