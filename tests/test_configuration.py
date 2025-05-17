"""
tests/test_configuration.py
===========================

Smoke-tests deepsearcher.configuration *after* the NullVectorDB patch.
Run with:  python tests/test_configuration.py
"""
from pathlib import Path
import sys, logging, asyncio

repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))

from deepsearcher.configuration import Configuration, init_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("config-test")

# ------------------------------------------------------------------ #
# 1. bootstrap
cfg_obj = Configuration()     # reads config.yaml + .env overrides
init_config(cfg_obj)          # populates globals inside the module

from deepsearcher import configuration as cfg

# ------------------------------------------------------------------ #
# 2. probe *one* available LLM
llm_client = cfg.lightllm or cfg.llm or cfg.backupllm
if llm_client:
    async def _ping():
        resp = await llm_client.chat_async([{"role": "user", "content": "ping"}])
        log.info("LLM responded: %s", resp.content.strip()[:999])
    asyncio.run(_ping())
else:
    log.warning("No LLM backend initialised; skipping chat probe.")

# ------------------------------------------------------------------ #
# 3. run a tiny query if default_searcher exists
QUESTION = "Briefly describe Milvus."
if getattr(cfg, "default_searcher", None):
    answer, *_ = cfg.default_searcher.query(QUESTION, use_web_search=False)
    log.info("default_searcher → %s", answer)
else:
    log.warning("default_searcher unavailable – skipping RAG probe.")

log.info("configuration.py smoke-test finished ✅")
