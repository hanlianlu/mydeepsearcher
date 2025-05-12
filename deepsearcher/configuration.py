import os
from typing import Dict, Any, Literal
import yaml
import logging
from deepsearcher.vector_db.base import BaseVectorDB
from deepsearcher.agent import ChainOfRAG, DeepSearch, NaiveRAG, WebSearchAgent
from deepsearcher.agent.rag_router import RAGRouter
from deepsearcher.embedding.base import BaseEmbedding
from deepsearcher.llm.base import BaseLLM
from deepsearcher.loader.file_loader.base import BaseLoader
from deepsearcher.loader.web_crawler.base import BaseCrawler
from deepsearcher.webservice.base import BaseSearchService
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
current_dir = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_YAML_PATH = os.path.join(current_dir, "..", "config.yaml")

FeatureType = Literal["llm", "embedding", "lightllm", "file_loader", "web_crawler", "vector_db", "search_service", "backupllm", "nanollm"]

class Configuration:
    def __init__(self, config_path: str = DEFAULT_CONFIG_YAML_PATH):
        self.provide_settings: Dict[str, Any] = {}
        self.query_settings: Dict[str, Any] = {}
        self.load_settings: Dict[str, Any] = {}
        self._load(config_path)

    def _load(self, config_path: str):
        logger.info(f"Loading configuration from {config_path}")
        try:
            with open(config_path, "r") as file:
                config_data = yaml.safe_load(file)
            self.provide_settings = config_data.get("provide_settings", {})
            self.query_settings = config_data.get("query_settings", {})
            self.load_settings = config_data.get("load_settings", {})
            logger.debug(f"Loaded provide_settings: {self.provide_settings.keys()}")
        except Exception as e:
            logger.error(f"Failed to load config file: {e}")
            raise

        # Override sensitive information with environment variables
        if 'llm' in self.provide_settings and 'config' in self.provide_settings['llm']:
            llm_config = self.provide_settings['llm']['config']
            azure_openai_key = os.getenv('AZURE_OPENAI_API_KEY')
            if azure_openai_key:
                llm_config['api_key'] = azure_openai_key
        
        if 'backupllm' in self.provide_settings and 'config' in self.provide_settings['backupllm']:
            llm_config = self.provide_settings['backupllm']['config']
            azure_openai_key = os.getenv('AZURE_OPENAI_API_KEY')
            if azure_openai_key:
                llm_config['api_key'] = azure_openai_key

        if 'lightllm' in self.provide_settings and 'config' in self.provide_settings['lightllm']:
            lightllm_config = self.provide_settings['lightllm']['config']
            azure_openai_key = os.getenv('AZURE_OPENAI_API_KEY')
            if azure_openai_key:
                lightllm_config['api_key'] = azure_openai_key

        if 'nanollm' in self.provide_settings and 'config' in self.provide_settings['nanollm']:
            lightllm_config = self.provide_settings['nanollm']['config']
            azure_openai_key = os.getenv('AZURE_OPENAI_API_KEY')
            if azure_openai_key:
                lightllm_config['api_key'] = azure_openai_key

        if 'embedding' in self.provide_settings and 'config' in self.provide_settings['embedding']:
            embedding_config = self.provide_settings['embedding']['config']
            openai_api_key = os.getenv('OPENAI_API_KEY')
            if openai_api_key:
                embedding_config['api_key'] = openai_api_key

    def set_provider_config(self, feature: FeatureType, provider: str, provider_configs: Dict[str, Any]):
        if feature not in self.provide_settings:
            logger.warning(f"Attempting to set unsupported feature: {feature}")
            raise ValueError(f"Unsupported feature: {feature}")
        self.provide_settings[feature]["provider"] = provider
        self.provide_settings[feature]["config"] = provider_configs
        logger.info(f"Set provider config for {feature}: {provider}")

    def get_provider_config(self, feature: FeatureType) -> Dict[str, Any]:
        if feature not in self.provide_settings:
            logger.error(f"Requested unsupported feature: {feature}")
            raise ValueError(f"Unsupported feature: {feature}")
        return self.provide_settings[feature]

class ModuleFactory:
    def __init__(self, config: Configuration):
        self.config = config

    def _create_module_instance(self, feature: FeatureType, module_name: str):
        """
        Dynamically import the requested `provider` class for *feature* and
        return an instantiated object.

        Special case:
            ─ DuckDuckGoSearchService now expects a `SearchConfig` object
              via the keyword argument `cfg=` instead of loose **kwargs.
              The branch below converts any YAML‑supplied `config:` dict into
              that object transparently.
        """
        provider_config = self.config.get_provider_config(feature)
        provider = provider_config.get("provider")
        class_name = provider
        cfg_kwargs = provider_config.get("config", {}) or {}
        logger.debug("Provider='%s', raw‑config=%s", provider, cfg_kwargs)

        # ------------------------------------------------------------------ #
        # Helper: instantiate, handling the DuckDuckGoSearchService special case
        # ------------------------------------------------------------------ #
        def _instantiate(module):
            cls = getattr(module, class_name)
            if class_name == "DuckDuckGoSearchService":
                # Local import to avoid touching unrelated code paths
                from deepsearcher.webservice.ddgsearchservice import SearchConfig

                search_cfg = SearchConfig(**cfg_kwargs) if cfg_kwargs else SearchConfig()
                return cls(cfg=search_cfg)
            return cls(**cfg_kwargs)

        # ------------------------------------------------------------------ #
        # Primary import path: deepsearcher.webservice (or similar top‑level)
        # ------------------------------------------------------------------ #
        try:
            module = __import__(module_name, fromlist=[class_name])
            instance = _instantiate(module)
            return instance

        except (ImportError, AttributeError, TypeError) as e:
            logger.warning("Primary import failed for %s: %s", class_name, e)

            # -------------------------------------------------------------- #
            # Fallback path: deepsearcher.webservice.<provider‑lowercase>
            # -------------------------------------------------------------- #
            full_module_name = f"{module_name}.{provider.lower()}"
            logger.info("Trying fallback import path '%s'", full_module_name)
            try:
                module = __import__(full_module_name, fromlist=[class_name])
                instance = _instantiate(module)
                logger.info("Successfully created %s (fallback path)", class_name)
                return instance

            except (ImportError, AttributeError, TypeError) as inner_e:
                logger.error("Fallback import failed for %s: %s", class_name, inner_e)
                raise ImportError(
                    f"Failed to instantiate '{class_name}' for feature '{feature}' from "
                    f"'{module_name}' or '{full_module_name}'. First error: {e}; "
                    f"fallback error: {inner_e}"
                ) from inner_e


    def create_llm(self) -> BaseLLM:
        return self._create_module_instance("llm", "deepsearcher.llm")
    
    def create_backupllm(self) -> BaseLLM:
        return self._create_module_instance("backupllm", "deepsearcher.llm")

    def create_lightllm(self) -> BaseLLM:
        return self._create_module_instance("lightllm", "deepsearcher.llm")
    
    def create_nanollm(self) -> BaseLLM:
        return self._create_module_instance("nanollm", "deepsearcher.llm")

    def create_embedding(self) -> BaseEmbedding:
        return self._create_module_instance("embedding", "deepsearcher.embedding")

    def create_file_loader(self) -> BaseLoader:
        return self._create_module_instance("file_loader", "deepsearcher.loader.file_loader")

    def create_web_crawler(self) -> BaseCrawler:
        return self._create_module_instance("web_crawler", "deepsearcher.loader.web_crawler")

    def create_vector_db(self) -> BaseVectorDB:
        return self._create_module_instance("vector_db", "deepsearcher.vector_db")

    def create_search_service(self) -> BaseSearchService:
        return self._create_module_instance("search_service", "deepsearcher.webservice")

# Global objects initialized later
module_factory: ModuleFactory = None
llm: BaseLLM = None
backupllm: BaseLLM = None
lightllm: BaseLLM = None
nanollm: BaseLLM = None
embedding_model: BaseEmbedding = None
file_loader: BaseLoader = None
vector_db: BaseVectorDB = None
web_crawler: BaseCrawler = None
default_searcher = None
naive_rag = None
search_service: BaseSearchService = None
web_search_agent: WebSearchAgent = None  # Add global variable for WebSearchAgent

def init_config(config: Configuration):
    global module_factory, llm, backupllm, embedding_model, lightllm, file_loader, vector_db, web_crawler, default_searcher, naive_rag, search_service, web_search_agent, nanollm
    logger.info("Starting configuration initialization")
    module_factory = ModuleFactory(config)
    try:
        llm = module_factory.create_llm()
        backupllm = module_factory.create_backupllm()
        lightllm = module_factory.create_lightllm()
        nanollm = module_factory.create_nanollm()
        embedding_model = module_factory.create_embedding()
        file_loader = module_factory.create_file_loader()
        web_crawler = module_factory.create_web_crawler()
        vector_db = module_factory.create_vector_db()
        search_service = module_factory.create_search_service()
        
        if search_service is None:
            logger.error("Failed to initialize search_service: returned None")
            raise ValueError("Search service initialization failed")
        
        # Initialize WebSearchAgent with fallback mechanism for llm
        web_llm = lightllm if lightllm is not None else llm if llm is not None else backupllm
        highllm = backupllm if backupllm is not None else llm
        if web_llm is None:
            logger.error("No LLM available for WebSearchAgent initialization")
            raise ValueError("Failed to initialize WebSearchAgent: No LLM available")
        
        web_search_agent = WebSearchAgent(
            search_service=search_service,
            web_crawler=web_crawler,
            llm=web_llm,
            max_urls = 10,
            max_chunk_size=6000,
            url_relevance_threshold=0.71
        )
        if web_search_agent is None:
            logger.error("Failed to initialize web_search_agent: returned None")
            raise ValueError("Web search agent initialization failed")
        
        default_searcher = RAGRouter(
            llm=llm,
            lightllm=web_llm,
            rag_agents=[
                DeepSearch(
                    llm=llm,
                    lightllm=web_llm,
                    highllm=highllm,
                    embedding_model=embedding_model,
                    vector_db=vector_db,
                    max_iter=config.query_settings.get("max_iter", 5),
                    route_collection=True,
                    text_window_splitter=True,
                ),
                ChainOfRAG(
                    llm=llm,
                    lightllm=web_llm,
                    highllm=highllm,
                    embedding_model=embedding_model,
                    vector_db=vector_db,
                    max_iter=config.query_settings.get("max_iter", 5),
                    route_collection=True,
                    text_window_splitter=True,
                ),
            ],
        )
        naive_rag = NaiveRAG(
            llm=web_llm,
            embedding_model=embedding_model,
            vector_db=vector_db,
            top_k=10,
            route_collection=True,
            text_window_splitter=True,
        )
        logger.info("Configuration initialization completed")
    except Exception as e:
        logger.error(f"Configuration initialization failed: {e}")
        raise