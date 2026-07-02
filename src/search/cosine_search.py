"""
cosine_search.py

STUB TEMPORÁRIO — este módulo é responsabilidade do Welder.
Implementa busca exata por similaridade de cosseno, usada como "gabarito"
(ground truth) para calcular o recall@k do HNSW.

Assim que o Welder entregar a versão final, basta substituir este arquivo
pelo dele — a interface (nomes de função e parâmetros) deve ser mantida
para não quebrar o restante do código do Henrique.

Complexidade: O(N * d) por consulta, onde N = número de filmes e
d = dimensão dos vetores (compara a query contra todos os N vetores).
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
        Vetor da pergunta do usuário já embedado.
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
    # Normaliza para usar produto escalar como similaridade de cosseno
    query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-12)
    emb_norms = embeddings / (
        np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    )

    similarities = emb_norms @ query_norm  # shape (N,)

    # Pega os top_k maiores sem ordenar tudo (O(N) + O(k log k))
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
    Mesma busca, mas para várias queries de uma vez (usado nos benchmarks
    para gerar o ground truth de recall@k de forma mais rápida).

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

    sims = q_norms @ emb_norms.T  # shape (n_queries, N)

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
