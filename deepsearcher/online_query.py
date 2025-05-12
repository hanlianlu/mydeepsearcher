from typing import List, Tuple
from deepsearcher import configuration
from deepsearcher.vector_db.base import RetrievalResult


def query(original_query: str, history_context: str = "", max_iter: int = 4, collections_names: list = None, use_web_search: bool = False) -> Tuple[str, List[RetrievalResult], int]:
    """
    Execute a query through the default searcher, optionally using web search.

    Args:
        original_query (str): The user's query.
        history_context (str): Previous conversation context.
        max_iter (int): Maximum iterations for query processing.
        collections_names (list): List of collection names to search.
        use_web_search (bool): Whether to include web search results.

    Returns:
        Tuple[str, List[RetrievalResult], int]: Answer, retrieved results, and token count.
    """
    default_searcher = configuration.default_searcher
    return default_searcher.query(
        original_query,
        history_context,
        max_iter=max_iter,
        collections_names=collections_names,
        use_web_search=use_web_search
    )


def retrieve(
    original_query: str, collections_names: list = None, history_context: str = "", max_iter: int = 4 ) -> Tuple[List[RetrievalResult], List[str], int]:
    default_searcher = configuration.default_searcher
    retrieved_results, consume_tokens, metadata = default_searcher.retrieve(
        original_query, collections_names = collections_names, history_context= history_context, max_iter=max_iter
    )
    return retrieved_results, [], consume_tokens


def naive_retrieve(query: str, collections_names: list = None, top_k=10) -> List[RetrievalResult]:
    naive_rag = configuration.naive_rag
    all_retrieved_results, consume_tokens, _ = naive_rag.retrieve(query, collections_names=collections_names)
    return all_retrieved_results


def naive_rag_query(
    query: str, collection: str = None, top_k=10
) -> Tuple[str, List[RetrievalResult]]:
    naive_rag = configuration.naive_rag
    answer, retrieved_results, consume_tokens = naive_rag.query(query)
    return answer, retrieved_results


def delete_document_by_reference(reference: str, collection: str = None) -> bool:
    """
    根据给定的 reference 删除 Milvus 中对应的文档。
    
    Args:
        reference: 文档的引用标识（例如 blob 名称）。
        collection: (可选) 指定的集合名称。如果不指定，则使用配置中的默认集合。
        
    Returns:
        如果删除成功，返回 True，否则返回 False。
    """
    vector_db = configuration.vector_db
    try:
        # 调用 Milvus 类中你实现的 delete_document 方法
        vector_db.delete_document(collection=collection, reference=reference)
        print(f"Successfully deleted document with reference: {reference}")
        return True
    except Exception as e:
        print(f"Error deleting document with reference {reference}: {e}")
        return False