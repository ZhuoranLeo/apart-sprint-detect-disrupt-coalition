"""Shim: the game code asks for LocalLLMClient; send it to OpenRouter instead."""
from src.llm.llm_client import LLMClient as LocalLLMClient  # noqa: F401
