"""
master graph_flow
=================

Entry workflow for *mydeepsearcher*:

    User → RAGRouterAgent
           ├─ 1 → DeepSearchAgent  (balanced, all-purpose)
           ├─ 2 → NaiveRAGAgent    (fast, one-shot)
           └─ 3 → ChainOfRAGAgent  (precision, multi-hop)
"""

from autogen_agentchat.teams  import GraphFlow
from autogen_agentchat.messages import TextMessage as Message

# 1) Meta-router
from deepsearcher.agents.rag_router_agent import build as build_router

# 2) Three wrapper pipelines
from deepsearcher.agents.deep_seacher_agent   import build as build_deep_agent
from deepsearcher.agents.naive_rag_agent    import build as build_naive_agent
from deepsearcher.agents.chain_of_rag_agent   import build as build_chain_agent

# 3) Runtime context (lightweight configuration)
from deepsearcher.configuration import Configuration as cfg


def build_flow( default_cfg: dict | None = None) -> GraphFlow:
    """
    Returns a GraphFlow whose single branching point is RAGRouterAgent.

    *Participants*
        0: RAGRouterAgent   — decides strategy
        1: DeepSearchAgent  — legacy DeepSearch monolith
        2: NaiveRAGAgent    — fast one-shot RAG
        3: ChainOfRAGAgent  — precision multi-hop RAG
    """
    default_cfg = default_cfg or {"use_web_search": False}

    # --- instantiate agents --------------------------------------------------
    deep_agent  = build_deep_agent(cfg)
    naive_agent = build_naive_agent(cfg)
    chain_agent = build_chain_agent(cfg)

    descriptions = [
        deep_agent.__description__,
        naive_agent.__description__,
        chain_agent.__description__,
    ]

    rag_router = build_router(cfg.model_client, descriptions)

    # --- GraphFlow skeleton ---------------------------------------------------
    flow = GraphFlow([rag_router, deep_agent, naive_agent, chain_agent])
    flow.default_config = default_cfg

    # Start node
    @flow.on_start
    def _start(msg: Message, _state):
        return rag_router

    # Branching: router outputs an *int* index 0-2
    flow.connect(rag_router, deep_agent,  condition=lambda m: m.content == 0)
    flow.connect(rag_router, naive_agent, condition=lambda m: m.content == 1)
    flow.connect(rag_router, chain_agent, condition=lambda m: m.content == 2)

    # All leaf agents finish the dialogue
    flow.connect(deep_agent,  flow.END)
    flow.connect(naive_agent, flow.END)
    flow.connect(chain_agent, flow.END)

    return flow
