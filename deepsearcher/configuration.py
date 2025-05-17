# deepsearcher/configuration.py
"""
deepsearcher.configuration  –  fault-tolerant bootstrap
=======================================================

• Each component is initialised in its own try/except block.
• On failure a lightweight dummy object is injected and a warning logged.
• Legacy globals (vector_db, llm, …) are still exported for
  backwards-compatibility, while new code can use the returned
  `RuntimeContext` instance.
"""
from __future__ import annotations
import os, logging, yaml
from typing import Any, Dict, Literal
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --------------------------------------------------------------------- #
# Interfaces (imported lazily where possible)
# --------------------------------------------------------------------- #
from deepsearcher.vector_db.base       import BaseVectorDB
from deepsearcher.embedding.base       import BaseEmbedding
from deepsearcher.llm.base             import BaseLLM
from deepsearcher.loader.file_loader.base import BaseLoader
from deepsearcher.loader.web_crawler.base import BaseCrawler
from deepsearcher.webservice.base      import BaseSearchService

from deepsearcher.agent                import (
    DeepSearch, ChainOfRAG, NaiveRAG, WebSearchAgent
)
from deepsearcher.agent.rag_router     import RAGRouter

# compatibility helper for AutoGen 0.5.7
from deepsearcher.utils.autogen_helper import ensure_autogen_llm_compat

# --------------------------------------------------------------------- #
DEFAULT_YAML = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "config.yaml"
)

FeatureType = Literal[
    "llm", "embedding", "lightllm", "backupllm", "nanollm",
    "file_loader", "web_crawler", "vector_db", "search_service"
]

# --------------------------------------------------------------------- #
# Dummy shims
# --------------------------------------------------------------------- #
class NullVectorDB(BaseVectorDB):
    """In-memory do-nothing stand-in so imports never fail."""
    def __init__(self, dim: int = 1536):
        self._dim = dim
        self._store: Dict[str, list] = {}

    @property
    def has_connection(self) -> bool:
        return False

    def list_collections(self):
        return []

    def init_collection(self, name: str, dim: int | None = None):
        self._store[name] = []
        self._dim = dim or self._dim

    def insert_data(self, collection: str, embeddings, metadatas):
        return

    def clear_db(self):
        self._store.clear()

    def search_data(self, collection: str, vector, limit: int = 4):
        return []

    @property
    def dimension(self) -> int:
        return self._dim


class DummySearch(BaseSearchService):
    async def search(self, *_):
        return []


class _NoOp:
    """Signals missing component."""
    def __getattr__(self, item):
        raise RuntimeError(f"{item} unavailable")


# --------------------------------------------------------------------- #
# Configuration loader
# --------------------------------------------------------------------- #
class Configuration:
    def __init__(self, path: str = DEFAULT_YAML):
        self.provide_settings: Dict[str,Any] = {}
        self.query_settings:   Dict[str,Any] = {}
        self.load_settings:    Dict[str,Any] = {}
        self._load_yaml(path)

    def _load_yaml(self, path: str):
        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f) or {}
            self.provide_settings = data.get("provide_settings", {})
            self.query_settings   = data.get("query_settings", {})
            self.load_settings    = data.get("load_settings", {})
            logger.info("Loaded configuration %s", path)
        except Exception as e:
            logger.warning("Config load failed (%s) – using empty defaults", e)

        # inject any API keys we find in the environment
        for api_var in ("AZURE_OPENAI_API_KEY","OPENAI_API_KEY"):
            if val := os.getenv(api_var):
                for ft in ("llm","backupllm","lightllm","nanollm","embedding"):
                    node = self.provide_settings.setdefault(ft, {})
                    node.setdefault("config", {})["api_key"] = val

    def get_provider_config(self, feature: FeatureType) -> Dict[str,Any]:
        return self.provide_settings.get(feature, {})


# --------------------------------------------------------------------- #
# Module factory
# --------------------------------------------------------------------- #
class ModuleFactory:
    def __init__(self, cfg: Configuration):
        self.cfg = cfg

    def _create(self, feature: FeatureType, mod_path: str):
        prov = self.cfg.get_provider_config(feature)
        cls  = prov.get("provider")
        kw   = prov.get("config", {}) or {}
        if not cls:
            raise ImportError(f"No provider configured for '{feature}'")
        try:
            module = __import__(mod_path, fromlist=[cls])
            return getattr(module, cls)(**kw)
        except Exception as e:
            # fallback path
            try:
                module = __import__(f"{mod_path}.{cls.lower()}", fromlist=[cls])
                return getattr(module, cls)(**kw)
            except Exception:
                raise e

    def create_llm(self)         -> BaseLLM:       return self._create("llm",         "deepsearcher.llm")
    def create_backupllm(self)   -> BaseLLM:       return self._create("backupllm",   "deepsearcher.llm")
    def create_lightllm(self)    -> BaseLLM:       return self._create("lightllm",    "deepsearcher.llm")
    def create_nanollm(self)     -> BaseLLM:       return self._create("nanollm",     "deepsearcher.llm")
    def create_embedding(self)   -> BaseEmbedding: return self._create("embedding",   "deepsearcher.embedding")
    def create_file_loader(self) -> BaseLoader:    return self._create("file_loader", "deepsearcher.loader.file_loader")
    def create_web_crawler(self) -> BaseCrawler:   return self._create("web_crawler", "deepsearcher.loader.web_crawler")
    def create_vector_db(self)   -> BaseVectorDB:  return self._create("vector_db",   "deepsearcher.vector_db")
    def create_search_service(self)->BaseSearchService: return self._create("search_service","deepsearcher.webservice")


# --------------------------------------------------------------------- #
# Runtime context used by new pipelines / agents
# --------------------------------------------------------------------- #
class RuntimeContext:
    def __init__(self, **entries):
        self.__dict__.update(entries)


# --------------------------------------------------------------------- #
# Initialise everything
# --------------------------------------------------------------------- #
def init_config(cfg: Configuration) -> RuntimeContext:
    factory = ModuleFactory(cfg)

    # --- helper: treat None or exceptions as fallback --------------- #
    def _safe(fn, default):
        try:
            result = fn()
            return result if result is not None else default
        except Exception as e:
            logger.warning("%s init failed: %s", fn.__name__, e)
            return default

    # --- instantiate LLMs & wrap for AutoGen ------------------------ #
    raw_llm      = _safe(factory.create_llm,          _NoOp())
    raw_backup   = _safe(factory.create_backupllm,    raw_llm)
    raw_light    = _safe(factory.create_lightllm,     raw_llm)
    raw_nano     = _safe(factory.create_nanollm,      raw_llm)

    # wrap all four so they support .create(...) and .model_info
    llm          = ensure_autogen_llm_compat(raw_llm)
    backupllm    = ensure_autogen_llm_compat(raw_backup)
    lightllm     = ensure_autogen_llm_compat(raw_light)
    nanollm      = ensure_autogen_llm_compat(raw_nano)

    # must have at least one
    if isinstance(llm, _NoOp) and isinstance(backupllm, _NoOp):
        raise RuntimeError("No LLM backend initialised.")

    # --- other components ------------------------------------------- #
    embedding    = _safe(factory.create_embedding,   _NoOp())
    vectordb     = _safe(factory.create_vector_db,   NullVectorDB())
    file_loader  = _safe(factory.create_file_loader, _NoOp())
    crawler      = _safe(factory.create_web_crawler, _NoOp())
    search_srv   = _safe(factory.create_search_service, DummySearch())

    # --- WebSearchAgent --------------------------------------------- #
    try:
        web_search = WebSearchAgent(
            search_service=search_srv,
            web_crawler=crawler,
            llm=lightllm if not isinstance(lightllm, _NoOp) else backupllm,
            max_urls=10,
            max_chunk_size=6000,
            url_relevance_threshold=0.71,
        )
    except Exception as e:
        logger.warning("WebSearchAgent init failed: %s (dummy)", e)
        web_search = DummySearch()

    # --- RAGRouter (fallback to Naive) ------------------------------ #
    try:
        default_search = RAGRouter(
            llm=llm,
            lightllm=lightllm,
            rag_agents=[
                DeepSearch(
                    llm=llm, lightllm=lightllm, highllm=backupllm,
                    embedding_model=embedding, vector_db=vectordb,
                    max_iter=cfg.query_settings.get("max_iter", 5),
                ),
                ChainOfRAG(
                    llm=llm, lightllm=lightllm, highllm=backupllm,
                    embedding_model=embedding, vector_db=vectordb,
                    max_iter=cfg.query_settings.get("max_iter", 5),
                ),
            ],
        )
    except Exception as e:
        logger.warning("RAGRouter init failed: %s (using NaiveRAG)", e)
        default_search = NaiveRAG(
            llm=lightllm, embedding_model=embedding, vector_db=vectordb
        )

    # --- Build context object --------------------------------------- #
    ctx = RuntimeContext(
        llm_client       = llm,
        backupllm        = backupllm,
        lightllm         = lightllm,
        nanollm          = nanollm,
        # alias for pipelines expecting ctx.highllm
        highllm          = backupllm,
        embedding_model  = embedding,
        vector_db        = vectordb,
        file_loader      = file_loader,
        web_crawler      = crawler,
        search_service   = search_srv,
        web_search_agent = web_search,
        default_searcher = default_search,
        config           = cfg,
    )

    # --- Backwards-compat globals ---------------------------- #
    globals().update(
        module_factory   = factory,
        llm              = llm,
        backupllm        = backupllm,
        lightllm         = lightllm,
        nanollm          = nanollm,
        highllm          = backupllm,
        embedding_model  = embedding,
        vector_db        = vectordb,
        file_loader      = file_loader,
        web_crawler      = crawler,
        search_service   = search_srv,
        web_search_agent = web_search,
        default_searcher = default_search,
        naive_rag        = _NoOp(),          # still created lazily if needed
        max_iter         = cfg.query_settings.get("max_iter", 5),
    )

    return ctx
