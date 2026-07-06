"""Sentence embeddings baseados em sentence-transformers."""

from __future__ import annotations

import os

import numpy as np
from sentence_transformers import SentenceTransformer


class SentenceEmbeddingPipeline:
    """
    Wrapper simples para o modelo da Dyesi.

    Modelo padrao:
    - `all-MiniLM-L6-v2`
    - dimensao 384
    - licenca Apache 2.0
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        cache_dir = os.getenv("PAA_HF_CACHE_DIR") or os.getenv("HF_HOME") or ".cache/huggingface"
        local_files_only = os.getenv("PAA_HF_LOCAL_ONLY", "0").strip().lower() in {"1", "true", "yes"}
        self.model = SentenceTransformer(
            model_name,
            cache_folder=cache_dir,
            local_files_only=local_files_only,
        )

    def get_embedding(self, text: str) -> np.ndarray:
        return self.model.encode(text, convert_to_numpy=True).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.get_embedding(text)


def load_sentence_embedder(model_name: str = "all-MiniLM-L6-v2") -> SentenceEmbeddingPipeline:
    return SentenceEmbeddingPipeline(model_name=model_name)
