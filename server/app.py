"""Entry point alias — re-exports the FastAPI app for multi-mode deployment."""

from server.main import app, main

__all__ = ["app", "main"]
