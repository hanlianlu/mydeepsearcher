# app.py
import json
import threading
import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from handlers import chat, revectorize, collections_list

app = FastAPI()

class QueryRequest(BaseModel):
    question: str
    history: str = ""
    collections: list[str] = []
    use_web: bool = False
    max_iter: int = 2

@app.post("/query")
async def query_stream(req: QueryRequest):
    queue: asyncio.Queue = asyncio.Queue()

    def think_cb(msg: str):
        queue.put_nowait({'type': 'think', 'data': msg})

    def run_query():
        answer, docs, tokens = chat(
            req.question,
            req.history,
            req.collections,
            req.use_web,
            req.max_iter,
            think_callback=think_cb,
        )
        queue.put_nowait({'type': 'answer', 'data': answer})
        queue.put_nowait({'type': 'done', 'data': {'docs': docs, 'tokens': tokens}})

    threading.Thread(target=run_query, daemon=True).start()

    async def event_generator():
        while True:
            ev = await queue.get()
            data = json.dumps(ev['data'], ensure_ascii=False)
            yield f"event: {ev['type']}\ndata: {data}\n\n"
            if ev['type'] == 'done':
                break

    return EventSourceResponse(event_generator())

@app.get("/collections")
async def get_collections():
    return {"collections": collections_list()}

@app.post("/revectorize")
async def post_revectorize():
    ok, msg = revectorize()
    return {"ok": ok, "msg": msg}