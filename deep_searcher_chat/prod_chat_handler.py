"""
Middleware between Streamlit UI and DeepSearcher backend
─────────────────────────────────────────────────────────
Adds a global lock for safe re-vectorization so that only one
process runs at a time – all other callers get an immediate
“already in progress” warning.
"""
from __future__ import annotations

import logging
import os
import json
import subprocess
import threading
from collections import defaultdict
from typing import List, Tuple

import streamlit as st
from deepsearcher.configuration import Configuration, init_config
from deepsearcher import configuration
from deepsearcher.online_query import query

# ───────────────────────── Logging ──────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────── Cached Milvus connection ───────────────
@st.cache_resource
def get_milvus_client():
    cfg = Configuration()
    init_config(cfg)
    return configuration.vector_db

milvus_client = get_milvus_client()

# ──────────────────────  THINK LOGGER  ───────────────────────
class ThinkLoggerHandler(logging.Handler):
    """Relays <think> messages from the agent to a Streamlit callback."""

    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def emit(self, record):
        msg = self.format(record)
        if "<think>" in msg:
            cleaned = msg.replace("<think>", "").replace("</think>", "").strip()
            if cleaned:
                self.callback(cleaned)


# ────────────────────  Global helpers  ───────────────────────
def initialize_deepsearcher() -> None:
    if "ds_initialized" not in st.session_state:
        st.session_state.ds_initialized = True


def list_collections() -> List[str]:
    """Return all collection names in Milvus."""
    try:
        infos = milvus_client.list_collections()
        return [col.collection_name for col in infos]
    except Exception as exc:
        logger.error("Failed to list collections: %s", exc)
        return []


def _metadata_to_dict(md):
    if isinstance(md, dict):
        return md
    if isinstance(md, str):
        try:
            return json.loads(md)
        except json.JSONDecodeError:
            logger.warning("Bad metadata JSON: %s", md)
    return {}


def get_display_text(doc):
    """Prefer metadata['wider_text'] over doc.text for display."""
    meta = _metadata_to_dict(doc.metadata)
    return meta.get("wider_text", doc.text)


def process_retrieved_results(retrieved):
    """Deduplicate references and build nice excerpts for UI."""
    doc_map = defaultdict(list)
    for idx, doc in enumerate(retrieved):
        ref = doc.reference
        excerpt = get_display_text(doc)[:200] + ("…" if len(doc.text) > 200 else "")
        meta = _metadata_to_dict(doc.metadata)
        source = meta.get("source", "local")
        full_url = meta.get("reference", "") if source == "web" else ""
        doc_map[ref].append((idx, excerpt, source, full_url))

    deduped = []
    for ref, items in doc_map.items():
        lines = []
        for idx, exc, src, url in items:
            link = f'<a href="{url}" target="_blank" rel="noopener noreferrer">web</a>' if src == "web" and url else f"({src})"
            lines.append(f"[Doc {idx}] {link}: {exc}")
        deduped.append({"reference": ref, "excerpt": "\n".join(lines)})
    return deduped


def process_query_stream(
    query_str: str,
    history_context: str = "",
    think_callback=None,
    maxiter: int = 4,
    collections: list | None = None,
    use_web_search: bool = False,
):
    initialize_deepsearcher()

    handler = None
    if think_callback:
        handler = ThinkLoggerHandler(think_callback)
        handler.setFormatter(logging.Formatter("%(message)s"))
        prog_logger = logging.getLogger("progress")
        prog_logger.addHandler(handler)
        prog_logger.setLevel(logging.INFO)

    try:
        final_answer, retrieved, tokens = query(
            query_str,
            history_context,
            max_iter=maxiter,
            collections_names=collections,
            use_web_search=use_web_search,
        )
        docs = process_retrieved_results(retrieved)
    finally:
        if handler:
            prog_logger.removeHandler(handler)

    return final_answer, docs, tokens


# ──────────────────  SAFE REVECTORIZE API  ───────────────────
_revectorize_lock = threading.Lock()

def revectorize_private_data() -> Tuple[bool, str]:
    """
    Launch load_azure_private_data.py in a subprocess.

    Returns
    -------
    (True, stdout) on success;
    (False, message) on failure or if another revectorization
    is already running.
    """
    if not _revectorize_lock.acquire(blocking=False):
        # Someone else is already revectorizing
        return False, "Revectorization already in progress."

    try:
        logger.info("Starting private-data revectorization …")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(script_dir, "..", "load_azure_private_data.py")
        env = os.environ.copy()

        result = subprocess.run(
            ["python", script_path],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        logger.info("Revectorization finished successfully.")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        err = (
            f"Script failed and exit {e.returncode}\nSTDERR: {e.stderr}\nSTDOUT: {e.stdout}"
        )
        logger.error(err)
        return False, err
    except Exception as exc:
        logger.exception("Unexpected error during revectorization")
        return False, str(exc)
    finally:
        _revectorize_lock.release()