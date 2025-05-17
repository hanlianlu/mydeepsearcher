from __future__ import annotations

from inspect import signature, getsource
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Detect the correct *_config keyword for AssistantAgent
def kw_for_assistant_agent() -> Optional[str]:
    """Return the keyword that the current Autogen build expects for the
    model‑config dict in `AssistantAgent.__init__`.
    """
    sig = signature(AssistantAgent.__init__)
    for cand in ("model_config", "llm_config", "config"):
        if cand in sig.parameters:
            return cand
    for name in sig.parameters:
        if name.endswith("_config"):
            return name
    return None

# Convert Python lambdas → string for DiGraphBuilder.add_edge() (0.5.7)
def edge_cond(fn: Callable | str | None) -> str | None:
    """Return `fn` unchanged if it is already a string; otherwise return the
    source code of the callable so Autogen 0.5.7 can eval it."""
    if callable(fn):
        return getsource(fn).strip()
    return str(fn) if fn is not None else None

# LLM compatibility wrapper
class AutogenLLMAdapter:
    """Expose `.create()` / `.stream()` and `.model_info` for Autogen."""
    def __init__(self, raw_llm: Any):
        self._raw = raw_llm
        self.model_info = getattr(raw_llm, "model_info", {"vision": False})

    @staticmethod
    def _to_openai_dict(msg: Any) -> Dict[str, str]:
        """Convert an Autogen or LangChain message into `{"role":.., "content":..}`."""
        if isinstance(msg, SystemMessage):
            return {"role": "system", "content": msg.content}
        elif isinstance(msg, HumanMessage):
            return {"role": "user", "content": msg.content}
        elif isinstance(msg, AIMessage):
            return {"role": "assistant", "content": msg.content}
        elif isinstance(msg, dict):
            return msg
        elif isinstance(msg, TextMessage):
            return {"role": msg.role, "content": msg.content}
        role = getattr(msg, "role", None) or getattr(msg, "source", None)
        content = getattr(msg, "content", None)
        if role is None or content is None:
            raise RuntimeError(f"Cannot convert {type(msg).__name__}: {msg!r} to chat format")
        return {"role": role, "content": content}

    async def create(self, messages: List[Any], **kwargs) -> Any:
        converted = [self._to_openai_dict(m) for m in messages]
        return await self._raw.chat_async(converted, **kwargs)

    def stream(self, messages: List[Any], **kwargs) -> AsyncIterator[Any]:
        converted = [self._to_openai_dict(m) for m in messages]
        return self._raw.stream_chat(converted, **kwargs)

    def __getattr__(self, item):
        return getattr(self._raw, item)

def ensure_autogen_llm_compat(raw_llm: Any) -> AutogenLLMAdapter:
    """Return *raw_llm* if it exposes a `.create` coroutine, else wrap it."""
    if callable(getattr(raw_llm, "create", None)):
        return raw_llm
    return AutogenLLMAdapter(raw_llm)