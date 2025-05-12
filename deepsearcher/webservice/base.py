from abc import ABC, abstractmethod
from typing import List

class BaseSearchService(ABC):
    @abstractmethod
    def search(self, query: str, count: int = 10) -> List[str]:
        """
        Search for relevant URLs based on the query.

        Args:
            query (str): The search query.
            count (int): The maximum number of URLs to return.

        Returns:
            List[str]: A list of relevant URLs.
        """
        pass