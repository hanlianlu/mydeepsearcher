from abc import ABC
from typing import Any, List, Tuple

from deepsearcher.vector_db import RetrievalResult


def describe_class(description):
    def decorator(cls):
        cls.__description__ = description
        return cls

    return decorator


class BaseAgent(ABC):
    def __init__(self, **kwargs):
        pass

    def invoke(self, query: str, **kwargs) -> Any:
        """
        Invoke the agent and return the result.
        Args:
            query: The query string.

        """


class RAGAgent(BaseAgent):
    def __init__(self, **kwargs):
        pass

    def retrieve(self, query: str, history_context: str, collections_names: list, **kwargs) -> Tuple[List[RetrievalResult], int, dict]:
        """
        Retrieve document results from the knowledge base.

        Args:
            query: The query string.

        Returns:
            A tuple containing:
                - the retrieved results
                - the total number of token usages of the LLM
                - any additional metadata, which can be an empty dictionary
        """

    def query(self, query: str, history_context :str = "", collections_names: list = None, use_web_search:bool = False, **kwargs) -> Tuple[str, List[RetrievalResult], int]:
        """
        Query the agent and return the answer.

        Args:
            query: The current query string
            history_context: default empty, could be users previous Q-A ish conversation

        Returns:
            A tuple containing:
                - the result generated from LLM
                - the retrieved document results
                - the total number of token usages of the LLM
        """
    def _checker_perform_retrieval(self, query: str) -> str:
        """
        Use a lightweight check (via LLM or heuristic rules) to decide if the query needs local RAG retrieval.
        Returns True if retrieval is needed, False if the LLM can answer directly.
        """
        """# Example: you might call your LLM  with a prompt like:
        prompt = (
            f"Analyze the following query and decide if retrieving local context is ever needed."
            f"Answer with 'yes' if retrieval is necessary or preferred, otherwise 'no' if the query is self-contained for LLM to resolve: {query}."
            f"Respond exclusively in valid str format without any other text"
        )
        decision = self.llm.chat(prompt)
        return decision.strip().lower().startswith('yes')"
        """

