"""
cosine_search.py

Busca exata por similaridade de cosseno, usada como ground truth para
calcular recall@k do HNSW e como backend de retrieval no sistema Q&A.

Funciona com qualquer matriz de embeddings (Word2Vec, Sentence Embeddings, etc.)
desde que query e base compartilhem a mesma dimensão d.

Complexidade: O(N * d) por consulta, onde N = número de filmes e
d = dimensão dos vetores.
"""

from __future__ import annotations

import numpy as np


def exact_cosine_search(
    query_vector: np.ndarray,
    embeddings: np.ndarray,
    top_k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Busca exata por similaridade de cosseno (força bruta).

    Parameters
    ----------
    query_vector : np.ndarray, shape (d,)
        Vetor da pergunta já embedado.
    embeddings : np.ndarray, shape (N, d)
        Matriz com os vetores de todas as sinopses da base.
    top_k : int
        Quantidade de resultados mais similares a retornar.

    Returns
    -------
    indices : np.ndarray, shape (top_k,)
        Índices (linhas de `embeddings`) dos filmes mais similares,
        em ordem decrescente de similaridade.
    scores : np.ndarray, shape (top_k,)
        Similaridade de cosseno correspondente a cada índice.
    """
    query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-12)
    emb_norms = embeddings / (
        np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    )

    similarities = emb_norms @ query_norm

    top_k = min(top_k, len(similarities))
    partial_idx = np.argpartition(-similarities, top_k - 1)[:top_k]
    order = np.argsort(-similarities[partial_idx])
    indices = partial_idx[order]
    scores = similarities[indices]

    return indices, scores


def batch_exact_cosine_search(
    query_vectors: np.ndarray,
    embeddings: np.ndarray,
    top_k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Busca exata para várias queries de uma vez.

    Returns
    -------
    indices : np.ndarray, shape (n_queries, top_k)
    scores  : np.ndarray, shape (n_queries, top_k)
    """
    emb_norms = embeddings / (
        np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    )
    q_norms = query_vectors / (
        np.linalg.norm(query_vectors, axis=1, keepdims=True) + 1e-12
    )

    sims = q_norms @ emb_norms.T

    top_k = min(top_k, embeddings.shape[0])
    idx_part = np.argpartition(-sims, top_k - 1, axis=1)[:, :top_k]

    n_queries = query_vectors.shape[0]
    indices = np.empty((n_queries, top_k), dtype=np.int64)
    scores = np.empty((n_queries, top_k), dtype=np.float32)
    for i in range(n_queries):
        row_idx = idx_part[i]
        row_scores = sims[i, row_idx]
        order = np.argsort(-row_scores)
        indices[i] = row_idx[order]
        scores[i] = row_scores[order]

    return indices, scores
