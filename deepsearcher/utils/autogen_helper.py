from __future__ import annotations

from inspect import signature, getsource
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from langchain_core.messages import SystemMessage

# ----------------------------------------------------------------------------
# 1. Detect the correct *_config keyword for AssistantAgent
# ----------------------------------------------------------------------------

def kw_for_assistant_agent() -> Optional[str]:
    """Return the keyword that the current Autogen build expects for the
    model‑config dict in `AssistantAgent.__init__`.
    """
    sig = signature(AssistantAgent.__init__)
    # Preferred explicit names first
    for cand in ("model_config", "llm_config", "config"):
        if cand in sig.parameters:
            return cand
    # Fallback: anything ending with *_config
    for name in sig.parameters:
        if name.endswith("_config"):
            return name
    return None

# ----------------------------------------------------------------------------
# 2. Convert Python lambdas → string for DiGraphBuilder.add_edge() (0.5.7)
# ----------------------------------------------------------------------------

def edge_cond(fn: Callable | str | None) -> str | None:  # noqa: D401 – simple helper
    """Return `fn` unchanged if it is already a string; otherwise return the
    source code of the callable so Autogen 0.5.7 can eval it."""
    if callable(fn):
        return getsource(fn).strip()
    return str(fn) if fn is not None else None

# ----------------------------------------------------------------------------
# 3. LLM compatibility wrapper
# ----------------------------------------------------------------------------

class AutogenLLMAdapter:  # noqa: D101 – internal helper
    """Expose `.create()` / `.stream()` and `.model_info` for Autogen."""

    # We convert *each* message only when needed, so store
    # a reference to the raw client and forward everything else.
    def __init__(self, raw_llm: Any):
        self._raw = raw_llm
        # Autogen checks «model_info["vision"]» – provide a safe default.
        self.model_info = getattr(raw_llm, "model_info", {"vision": False})

    # ---- helpers ---------------------------------------------------------
    @staticmethod
    def _to_openai_dict(msg: Any) -> Dict[str, str]:
        """Convert an Autogen message *or* a plain dict into
        ``{"role":.., "content":..}``.
        Raises ``RuntimeError`` if conversion is impossible.
        """
        if isinstance(msg, dict):
            # Already in the desired schema – trust the caller.
            return msg

        # Handle LangChain SystemMessage
        if isinstance(msg, SystemMessage):
            return {"role": "system", "content": msg.content}

        # TextMessage covers user/assistant/system inside agentchat.
        if isinstance(msg, TextMessage):
            return {"role": msg.role, "content": msg.content}

        # Generic fallback: look for .role/.source + .content attrs.
        role = getattr(msg, "role", None) or getattr(msg, "source", None)
        content = getattr(msg, "content", None)
        if role is None or content is None:
            raise RuntimeError(f"Cannot convert {msg!r} to chat format")
        return {"role": role, "content": content}

    # ---- public API expected by Autogen ----------------------------------
    async def create(self, messages: List[Any], **kwargs) -> Any:
        converted = [self._to_openai_dict(m) for m in messages]
        return await self._raw.chat_async(converted, **kwargs)

    # Autogen sometimes calls `.stream` rather than `.create_stream`.
    def stream(self, *, messages: List[Any], **kwargs) -> AsyncIterator[Any]:
        converted = [self._to_openai_dict(m) for m in messages]
        return self._raw.stream_chat(converted, **kwargs)

    # Delegate every other attribute access to the wrapped object.
    def __getattr__(self, item):  # noqa: D401 – passthrough
        return getattr(self._raw, item)


# ---- public helper --------------------------------------------------------

def ensure_autogen_llm_compat(raw_llm: Any) -> AutogenLLMAdapter:
    """Return *raw_llm* if it already exposes a `.create` coroutine, else wrap
    it in :class:`AutogenLLMAdapter`. This should be called on every LLM
    instance before passing it into Autogen agents or teams."""
    if callable(getattr(raw_llm, "create", None)):
        return raw_llm  # already compatible
    return AutogenLLMAdapter(raw_llm)