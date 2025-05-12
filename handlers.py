# handlers.py
from typing import List, Tuple, Callable
from deep_searcher_chat.prod_chat_handler import (
    process_query_stream,
    revectorize_private_data,
    list_collections,
)


def chat(
    query: str,
    history: str,
    collections: List[str],
    use_web: bool,
    max_iter: int,
    think_callback: Callable[[str], None] | None = None,
) -> Tuple[str, List[dict], int]:
    return process_query_stream(
        query_str=query,
        history_context=history,
        think_callback=think_callback,
        maxiter=max_iter,
        collections=collections,
        use_web_search=use_web,
    )


def revectorize() -> Tuple[bool, str]:
    return revectorize_private_data()


def collections_list() -> List[str]:
    return list_collections()

# query_stream.py
import sys
import json
from handlers import chat

if __name__ == "__main__":
    payload = json.load(sys.stdin)
    answer, docs, tokens = chat(
        payload['question'],
        payload.get('history', ''),
        payload.get('collections', []),
        payload.get('use_web', False),
        payload.get('max_iter', 2),
        think_callback=None,
    )
    print(json.dumps({'answer': answer, 'docs': docs, 'tokens': tokens}), flush=True)