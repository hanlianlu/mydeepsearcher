from typing import List, Optional, Tuple
from deepsearcher.agent import RAGAgent
from deepsearcher.llm.base import BaseLLM
from deepsearcher.tools import log
from deepsearcher.vector_db import RetrievalResult

RAG_ROUTER_PROMPT = """Given a list of agent indexes and corresponding descriptions, each agent has a specific function. 
Given a query, select only one agent that best matches the agent handling the query, and return the index without any other information.
There might be information about max iteration of query provided for consideration.

## Question
{query}

## Agent Indexes and Descriptions
{description_str}

## Max Iteration
{max_iter}

Only return one agent index number that best matches the agent handling the query:
"""

class RAGRouter(RAGAgent):
    def __init__(
        self,
        llm: BaseLLM,
        lightllm: BaseLLM,
        rag_agents: List[RAGAgent],
        agent_descriptions: Optional[List[str]] = None,
    ):
        self.llm = llm
        self.lightllm = lightllm
        self.rag_agents = rag_agents
        self.agent_descriptions = agent_descriptions
        self.max_iter = 1
        if not self.agent_descriptions:
            try:
                self.agent_descriptions = [
                    agent.__class__.__description__ for agent in self.rag_agents
                ]
            except Exception:
                raise AttributeError(
                    "Please provide agent descriptions or set __description__ attribute for each agent class."
                )

    def _route(self, query: str, **kwargs) -> Tuple[RAGAgent, int]:
        max_iter = kwargs.pop("max_iter", self.max_iter)
        description_str = "\n".join(
            [f"[{i + 1}]: {description}" for i, description in enumerate(self.agent_descriptions)]
        )
        prompt = RAG_ROUTER_PROMPT.format(query=query, description_str=description_str, max_iter=max_iter)
        try:
            chat_response = self.lightllm.chat(messages=[{"role": "user", "content": prompt}])
        except Exception as e:
            log.warning(f"Light LLM failed: {e}, falling back to LLM.")
            chat_response = self.llm.chat(messages=[{"role": "user", "content": prompt}])
            
        try:
            selected_agent_index = int(chat_response.content) - 1
        except ValueError:
            log.warning(
                "Parse int failed in RAGRouter, but will try to find the last digit as fallback."
            )
            selected_agent_index = int(self.find_last_digit(chat_response.content)) - 1

        selected_agent = self.rag_agents[selected_agent_index]
        log.color_print(
            f"<think> Select agent [{selected_agent.__class__.__name__}] to answer the query [{query}] </think>\n"
        )
        return self.rag_agents[selected_agent_index], chat_response.total_tokens

    def retrieve(self, query: str, collections_names: list=None, history_context: str = "", **kwargs) -> Tuple[List[RetrievalResult], int, dict]:
        agent, n_token_router = self._route(query, **kwargs)
        retrieved_results, n_token_retrieval, metadata = agent.retrieve(query, collections_names=collections_names, history_context=history_context, **kwargs)
        return retrieved_results, n_token_router + n_token_retrieval, metadata

    def query(self, query: str, history_context: str = "", collections_names: list = None, use_web_search: bool = False, **kwargs) -> Tuple[str, List[RetrievalResult], int]:
        agent, n_token_router = self._route(query, **kwargs)
        answer, retrieved_results, n_token_retrieval = agent.query(
            query,
            history_context,
            collections_names=collections_names,
            use_web_search=use_web_search,
            **kwargs
        )
        return answer, retrieved_results, n_token_router + n_token_retrieval

    def find_last_digit(self, string):
        for char in reversed(string):
            if char.isdigit():
                return char
        raise ValueError("No digit found in the string")