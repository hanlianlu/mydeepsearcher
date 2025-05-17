"""
tests/test_micro_pipelines.py
=============================

Smoke-tests the three Autogen micro-agent pipelines **individually**:

    • deepsearcher_flow
    • chainofrag_flow
    • naiverag_flow

For every flow we execute the query twice:
    – with {"use_web_search": False}
    – with {"use_web_search": True }

The test passes when all six calls finish without raising and each
returns a non-empty string.
"""
from __future__ import annotations
from pathlib import Path
import sys, asyncio, logging, textwrap

# ── 1. Bootstrap project & configuration ──────────────────────────────
repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))                       # add repo to PYTHONPATH

from deepsearcher.configuration import Configuration, init_config

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s"
)
log = logging.getLogger("micro-flow-test")

cfg_obj = Configuration()        # loads config.yaml + .env overrides
ctx     = init_config(cfg_obj)   # fault-tolerant bootstrap
log.info("Configuration bootstrapped ✅")

# ── 2. Build the three micro-flows ────────────────────────────────────
from deepsearcher.pipelines.deepsearcher_flow import build as build_deep
from deepsearcher.pipelines.chainofrag_flow   import build as build_chain
from deepsearcher.pipelines.naiverag_flow     import build as build_naive

flows = {
    "DeepSearch" : build_deep(ctx),
    "ChainOfRAG" : build_chain(ctx),
    "NaiveRAG"   : build_naive(ctx),
}

# ── 3. Helper: run one flow with a runtime flag ───────────────────────
from autogen_agentchat.messages import TextMessage

async def run_flow(flow, *, question: str, use_web: bool) -> str:
    """
    • Attaches the per-run flag to the flow (**GraphFlow.run() accepts no
      kwargs in Autogen 0.5.7** – settings must live on `flow.config`).
    • Sends a user message and awaits the final assistant reply.
    """
    flow.config = {"use_web_search": use_web}    # ← key line

    msg = TextMessage(
        role    = "user",
        content = question,
        source  = "user",                        # mandatory field
    )
    reply_msg = await flow.run(task=msg)
    return reply_msg.content.strip()

# ── 4. Execute all flows (sync wrapper for convenience) ───────────────
QUESTION = "Briefly describe Milvus and give one practical use-case."
answers  = []

async def main():
    for name, flow in flows.items():
        for flag in (False, True):
            label = f"{name} | web={flag}"
            log.info("Running %s …", label)
            ans = await run_flow(flow, question=QUESTION, use_web=flag)
            print(
                f"\n── {label} ──\n"
                f"{textwrap.shorten(ans, width=300, placeholder='…')}\n"
            )
            answers.append(ans)

asyncio.run(main())

# ── 5. Simple assertions for CI / automated test runners ──────────────
assert all(isinstance(a, str) and a for a in answers), \
       "One pipeline returned an empty answer!"
log.info("All micro-agent pipelines executed successfully ✅")
