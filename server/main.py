"""FastAPI server entry point for the Memory Environment."""

from __future__ import annotations
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.routes.memory import router

app = FastAPI(
    title="Memory Environment",
    description="OpenEnv-compatible RL environment for LLM memory management benchmarking.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


if __name__ == "__main__":
    uvicorn.run("server.main:app", host="0.0.0.0", port=8004, reload=False)
