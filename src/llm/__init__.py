"""Integracao com LLMs locais usados para formatar a resposta final."""

from .tinyllama import GenerationResult, TinyLlamaClient, TinyLlamaConfig

__all__ = ["GenerationResult", "TinyLlamaClient", "TinyLlamaConfig"]
