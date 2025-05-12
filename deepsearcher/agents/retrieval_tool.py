from autogen_agentchat.tools import AgentTool
import asyncio, json
from typing import List, Dict, Any
from deepsearcher.vector_db import RetrievalResult
from deepsearcher.vector_db.base import deduplicate_results

def make_retrieval_tool(vector_db, embed_model, top_k: int = 9) -> AgentTool:
    
    async def _vectordb_search(
        sub_queries: List[str],
        collections: List[str] | None = None,
    ) -> Dict[str, Any]:
        try:
            if not sub_queries:
                return {"preview": [], "total_hits": 0}
            cols = collections or vector_db.list_collections()
            sem = asyncio.Semaphore(10)
            async def _search_one(coll, q):
                async with sem:
                    return await asyncio.to_thread(
                        vector_db.search_data,
                        collection=coll,
                        vector=embed_model.embed_query(q),
                        limit=top_k,
                    )
            batches = await asyncio.gather(*[ _search_one(c,q) for q in sub_queries for c in cols ])
            hits = deduplicate_results([h for batch in batches for h in batch])
            preview = [
                {
                    "id": i,
                    "text": h.text[:3000] + ("…" if len(h.text) > 3000 else ""),
                    "score": getattr(h, "distance", None),
                    "source": h.metadata.get("source", ""),
                }
                for i,h in enumerate(hits)
            ]
            return {"preview": preview, "total_hits": len(hits)}
        except Exception:
            return {"preview": [], "total_hits": 0}

    return AgentTool(
        name="vectordb_search",
        description="Vector‐DB similarity search tool",
        function=_vectordb_search,
        result_converter=lambda res: json.dumps(res, ensure_ascii=False),
        save_output_in_message_metadata=True,
    )
