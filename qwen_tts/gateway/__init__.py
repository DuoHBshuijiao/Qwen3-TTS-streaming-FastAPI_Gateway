# coding=utf-8
"""HTTP gateway (FastAPI) for Qwen3 TTS — optional entrypoint, no Gradio required."""

from .app import create_app

__all__ = ["create_app"]
