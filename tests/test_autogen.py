import asyncio
import logging
from typing import Dict, Any

from autogen_agentchat.messages import TextMessage
from deepsearcher.configuration import Configuration, init_config
from deepsearcher.pipelines.deepsearcher_flow import build_deepsearcher_flow
from deepsearcher.pipelines.chainofrag_flow import build_chainofrag_flow
from deepsearcher.pipelines.naiverag_flow import build_naiverag_flow

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("micro-flow-test")

async def run_flow(flow, question: str, use_web: bool) -> str:
    try:
        msg = TextMessage(content=question, source="user")
        reply_msg = await flow.run(task=msg)
        if not isinstance(reply_msg, str) or not reply_msg.strip():
            logger.error(f"Flow {flow.__class__.__name__} with use_web={use_web} returned empty or invalid response")
            return ""
        return reply_msg
    except Exception as e:
        logger.error(f"Error running flow {flow.__class__.__name__} with use_web={use_web}: {str(e)}")
        return ""

async def main():
    config = Configuration()
    ctx = init_config(config)
    logger.info("Configuration bootstrapped ✅")
    
    QUESTION = "Briefly describe Milvus and give one practical use-case."
    flow_configs = [
        (build_deepsearcher_flow(ctx, use_web=False), "DeepSearch", False),
        (build_deepsearcher_flow(ctx, use_web=True), "DeepSearch", True),
        (build_chainofrag_flow(ctx, use_web=False), "ChainOfRAG", False),
        (build_chainofrag_flow(ctx, use_web=True), "ChainOfRAG", True),
        (build_naiverag_flow(ctx, use_web=False), "NaiveRAG", False),
        (build_naiverag_flow(ctx, use_web=True), "NaiveRAG", True),
    ]
    
    results: Dict[str, Dict[bool, str]] = {}
    for flow, name, use_web in flow_configs:
        if name not in results:
            results[name] = {}
        logger.info(f"Running {name} | web={use_web} …")
        ans = await run_flow(flow, QUESTION, use_web)
        results[name][use_web] = ans
        logger.info(f"{name} | web={use_web} → {ans[:50]}…")
    
    for name, res in results.items():
        for flag, ans in res.items():
            assert isinstance(ans, str) and ans.strip(), f"{name} with web={flag} failed to return a valid response"

if __name__ == "__main__":
    asyncio.run(main())