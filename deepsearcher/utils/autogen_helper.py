from inspect import signature, getsource
from typing import Any, AsyncIterator, List, Dict, Callable, Optional

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage


def kw_for_assistant_agent() -> Optional[str]:
    """
    Return the correct keyword for the model-configuration dict expected by
    AssistantAgent.__init__ in the installed Autogen 0.5.7 build.
    """
    sig = signature(AssistantAgent.__init__)

    # Preferred explicit names in Autogen 0.5.7
    for cand in ("model_config", "llm_config", "config"):
        if cand in sig.parameters:
            return cand

    # Fallback: any param ending in "_config"
    for name in sig.parameters:
        if name.endswith("_config"):
            return name

    return None


def edge_cond(fn: Callable) -> str:
    """
    Wrap a python callable so it can be passed as a string `condition`
    in DiGraphBuilder.add_edge for Autogen 0.5.7.
    """
    if callable(fn):
        return getsource(fn).strip()
    return str(fn)


class AutogenLLMAdapter:
    """
    Adapter to wrap an existing LLM client (with .chat_async()/.stream_chat())
    so that Autogen 0.5.7 can call .create(...) or .stream(...).
    """
    def __init__(self, raw_llm: Any):
        self._raw = raw_llm
        # signal no vision features
        self.model_info = {"vision": False}

    async def create(self, messages: List[Any], **kwargs) -> Any:
        # Convert Autogen message objects into dicts for chat_async
        converted: List[Dict[str, Any]] = []
        for m in messages:
            if isinstance(m, dict):
                converted.append(m)
            elif isinstance(m, TextMessage):
                converted.append({"role": m.role, "content": m.content})
            else:
                role = getattr(m, "role", None) or getattr(m, "source", None)
                content = getattr(m, "content", None)
                if role is None or content is None:
                    raise RuntimeError(f"Cannot convert {m!r} to chat format")
                converted.append({"role": role, "content": content})

        return await self._raw.chat_async(converted, **kwargs)

    def stream(self, messages: List[Any], **kwargs) -> AsyncIterator[Any]:
        converted: List[Dict[str, Any]] = []
        for m in messages:
            if isinstance(m, dict):
                converted.append(m)
            elif isinstance(m, TextMessage):
                converted.append({"role": m.role, "content": m.content})
            else:
                role = getattr(m, "role", None) or getattr(m, "source", None)
                content = getattr(m, "content", None)
                if role is None or content is None:
                    raise RuntimeError(f"Cannot convert {m!r} to chat format")
                converted.append({"role": role, "content": content})

        return self._raw.stream_chat(converted, **kwargs)


def ensure_autogen_llm_compat(raw_llm: Any) -> AutogenLLMAdapter:
    """
    Wrap any LLM client in an adapter so Autogen 0.5.7 can call .create()/.stream().
    """
    return AutogenLLMAdapter(raw_llm)
