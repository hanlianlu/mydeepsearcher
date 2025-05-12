# deepsearcher/utils/rag_helpers.py
from typing import List, Tuple
from deepsearcher.vector_db import RetrievalResult
import json

def format_retrieved_results(
    retrieved_results: List[RetrievalResult],
    text_window_splitter: bool = True,
) -> str:
    """Return a pretty <Document i> ... string used by prompts."""
    formatted = []
    for i, result in enumerate(retrieved_results):
        text = (
            result.metadata.get("wider_text", result.text)
            if text_window_splitter
            else result.text
        )
        formatted.append(
            f"<Document {i}>\nText: {text}\n"
            f"Reference: {json.dumps(result.reference)}\n"
            f"Metadata: {json.dumps(result.metadata)}\n</Document {i}>"
        )
    return "\n".join(formatted)



def parse_rerank_response(
    chunks: List[RetrievalResult],
    raw_json: str,
) -> List[Tuple[RetrievalResult, float]]:
    """
    Given the JSON string the LLM returned (list of ["YES"/"NO", score]),
    pair each accepted chunk with its score.
    """
    import json as _json, logging
    try:
        votes = _json.loads(raw_json)
    except _json.JSONDecodeError as e:
        logging.warning("Invalid rerank JSON: %s", e)
        return []

    accepted = []
    for chunk, (yn, score) in zip(chunks, votes):
        if str(yn).upper() == "YES":
            accepted.append((chunk, float(score)))
    return accepted


def confidence_prompt(original_query, sub_queries, chunk_str):
    return (
        "Based on the original query, previous sub-queries, and the retrieved "
        "chunks, assess your confidence (as a number between 0 and 1) that you "
        "have enough information to answer the query comprehensively with "
        "profound insights.\n\n"
        f"Original Query: {original_query}\n"
        f"Previous Sub Queries: {', '.join(sub_queries)}\n"
        f"Retrieved Chunks: {chunk_str}\n\n"
        "Respond with a single number with double decimal precision between 0.00 and 1.00."
    )
