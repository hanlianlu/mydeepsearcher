import os
import logging
from typing import Dict, Any, Literal
import yaml
from dotenv import load_dotenv

# Interfaces
from deepsearcher.vector_db.base import BaseVectorDB
from deepsearcher.embedding.base import BaseEmbedding
from deepsearcher.llm.base import BaseLLM
from deepsearcher.loader.file_loader.base import BaseLoader
from deepsearcher.loader.web_crawler.base import BaseCrawler
from deepsearcher.webservice.base import BaseSearchService

load_dotenv()

# Configure logging
tlogging = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default config path
dir_path = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_YAML_PATH = os.path.join(dir_path, '..', 'config.yaml')

FeatureType = Literal[
    'llm', 'embedding', 'lightllm', 'backupllm', 'nanollm',
    'file_loader', 'web_crawler', 'vector_db', 'search_service'
]

class Configuration:
    """
    Load settings from a YAML file and .env, with overrides.
    """
    def __init__(self, config_path: str = DEFAULT_CONFIG_YAML_PATH):
        self.provide_settings: Dict[str, Any] = {}
        self.query_settings:   Dict[str, Any] = {}
        self.load_settings:    Dict[str, Any] = {}
        self._load_yaml(config_path)

    def _load_yaml(self, path: str):
        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f) or {}
            self.provide_settings = data.get('provide_settings', {})
            self.query_settings   = data.get('query_settings', {})
            self.load_settings    = data.get('load_settings', {})
            logger.info(f"Loaded configuration from {path}")
        except Exception as e:
            logger.warning(f"Could not load config at {path}: {e}")
            self.provide_settings = {}
            self.query_settings   = {}
            self.load_settings    = {}

    def set_provider_config(self, feature: FeatureType, provider: str, config: Dict[str, Any]):
        if feature not in self.provide_settings:
            raise ValueError(f"Unsupported feature: {feature}")
        self.provide_settings[feature] = {'provider': provider, 'config': config}

    def get_provider_config(self, feature: FeatureType) -> Dict[str, Any]:
        cfg = self.provide_settings.get(feature)
        if cfg is None:
            raise ValueError(f"No provider configured for: {feature}")
        return cfg

class ModuleFactory:
    """
    Instantiate modules for each feature based on Configuration.
    """
    def __init__(self, config: Configuration):
        self.config = config

    def _instantiate(self, module_path: str, class_name: str, kwargs: Dict[str, Any]):
        module = __import__(module_path, fromlist=[class_name])
        cls = getattr(module, class_name)
        return cls(**kwargs)

    def _create(self, feature: FeatureType, module_path: str):
        prov = self.config.get_provider_config(feature)
        cls_name = prov['provider']
        cfg_kwargs = prov.get('config', {}) or {}
        try:
            return self._instantiate(module_path, cls_name, cfg_kwargs)
        except Exception:
            # Fallback to lowercase module
            fallback = f"{module_path}.{cls_name.lower()}"
            return self._instantiate(fallback, cls_name, cfg_kwargs)

    def create_llm(self) -> BaseLLM:
        return self._create('llm', 'deepsearcher.llm')
    def create_backupllm(self) -> BaseLLM:
        return self._create('backupllm', 'deepsearcher.llm')
    def create_lightllm(self) -> BaseLLM:
        return self._create('lightllm', 'deepsearcher.llm')
    def create_nanollm(self) -> BaseLLM:
        return self._create('nanollm', 'deepsearcher.llm')
    def create_embedding(self) -> BaseEmbedding:
        return self._create('embedding', 'deepsearcher.embedding')
    def create_file_loader(self) -> BaseLoader:
        return self._create('file_loader', 'deepsearcher.loader.file_loader')
    def create_web_crawler(self) -> BaseCrawler:
        return self._create('web_crawler', 'deepsearcher.loader.web_crawler')
    def create_vector_db(self) -> BaseVectorDB:
        return self._create('vector_db', 'deepsearcher.vector_db')
    def create_search_service(self) -> BaseSearchService:
        return self._create('search_service', 'deepsearcher.webservice')

# Dummy fallbacks
class DummyVectorDB(BaseVectorDB):
    def list_collections(self): return []
    def search_data(self, *args, **kwargs): return []
class DummyWebSearch:
    def retrieve(self, *args, **kwargs): return [], 0, {}
class DummyRAG:
    def retrieve(self, *args, **kwargs): return [], 0, {}
    def query(self, *args, **kwargs): return "", [], 0

# Global placeholders (populated in init_config)
module_factory = None
llm = backupllm = lightllm = nanollm = None
embedding_model = file_loader = web_crawler = None
vector_db = search_service = web_search_agent = None
default_searcher = naive_rag = None

# Initialization
def init_config(cfg: Configuration):
    global module_factory, llm, backupllm, lightllm, nanollm
    global embedding_model, file_loader, web_crawler, vector_db
    global search_service, web_search_agent, default_searcher, naive_rag

    module_factory = ModuleFactory(cfg)

    # LLMs
    try:
        llm = module_factory.create_llm()
        backupllm = module_factory.create_backupllm()
        lightllm = module_factory.create_lightllm()
        nanollm = module_factory.create_nanollm()
    except Exception as e:
        logger.warning(f"LLM init failed: {e}")
        llm = backupllm = lightllm = nanollm = None

    # Embedding
    try:
        embedding_model = module_factory.create_embedding()
    except Exception as e:
        logger.warning(f"Embedding init failed: {e}")
        embedding_model = None

    # File loader
    try:
        file_loader = module_factory.create_file_loader()
    except Exception as e:
        logger.warning(f"File loader init failed: {e}")
        file_loader = None

    # Web crawler
    try:
        web_crawler = module_factory.create_web_crawler()
    except Exception as e:
        logger.warning(f"Web crawler init failed: {e}")
        web_crawler = None

    # Vector DB
    try:
        vector_db = module_factory.create_vector_db()
    except Exception as e:
        logger.warning(f"VectorDB init failed: {e}")
        vector_db = DummyVectorDB()

    # Search service
    try:
        search_service = module_factory.create_search_service()
    except Exception as e:
        logger.warning(f"Search service init failed: {e}")
        search_service = None

    # WebSearchAgent
    try:
        from deepsearcher.agent import WebSearchAgent
        web_search_agent = WebSearchAgent(
            search_service=search_service,
            web_crawler=web_crawler,
            llm=lightllm or llm or backupllm,
            max_urls=10,
            max_chunk_size=6000,
            url_relevance_threshold=0.71,
        )
    except Exception as e:
        logger.warning(f"WebSearchAgent init failed: {e}")
        web_search_agent = DummyWebSearch()

    # RAG pipelines
    try:
        from deepsearcher.agent import DeepSearch, NaiveRAG, RAGRouter
        default_searcher = RAGRouter(
            llm=llm,
            lightllm=lightllm,
            rag_agents=[
                DeepSearch(
                    llm=llm, lightllm=lightllm, highllm=backupllm,
                    embedding_model=embedding_model, vector_db=vector_db,
                    max_iter=cfg.query_settings.get('max_iter', 5),
                ),
            ],
        )
        naive_rag = NaiveRAG(
            llm=lightllm or backupllm,
            embedding_model=embedding_model,
            vector_db=vector_db,
            top_k=cfg.query_settings.get('top_k', 10),
        )
    except Exception as e:
        logger.warning(f"RAG init failed: {e}")
        default_searcher = naive_rag = DummyRAG()

    logger.info("Configuration initialization complete.")
