deepsearcher/
├─ orchestration/
    ├─ gragh_flow.py           # Use legacy agents with autogen style agents to build graph.
├─ agents/                     # new micro-agents, as well as legacy-pipeline-agents
│   ├─ decomposer_agent.py
│   ├─ retrieval_agent.py
│   ├─ reranker_agent.py
│   ├─ reflection_agent.py
│   ├─ summarizer_agent.py
│   └─ ...
├─ pipelines/                  # one graphflow per micro based pipeline
│   ├─ deepsearch_flow.py      # DeepSearchGraphFlow.build(ctx)
│   ├─ chainofrag_flow.py      # ChainOfRAGGraphFlow.build(ctx)
│   └─ naiverag_flow.py        # NaiveRAGGraphFlow.build(ctx)
├─ router/                     # meta-orchestrator
│   └─ rag_router_agent.py     # picks which pipeline to run
├─ configuration.py, light_configuration.py   # (keep / thin-out)
└─ ...
