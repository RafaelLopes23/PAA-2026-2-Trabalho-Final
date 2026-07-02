"""
hnsw_search.py

Responsável: Henrique
Implementa busca aproximada de vizinhos mais próximos usando HNSW
(Hierarchical Navigable Small World), via a biblioteca `hnswlib`.

HNSW constrói um grafo em camadas sobre os vetores. A busca desce pelas
camadas (do topo, mais esparsa, até a base, mais densa) fazendo saltos
"gulosos" em direção ao vizinho mais próximo, evitando comparar a query
contra todos os N pontos da base.

Complexidade aproximada:
- Construção do índice: O(N log N)
- Busca:                O(log N)  (aproximado, não garantido)

Isso contrasta com a busca exata por cosseno (O(N * d) por consulta),
implementada em `cosine_search.py`.
"""

from __future__ import annotations
import time
import os
import json
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import hnswlib


@dataclass
class HNSWParams:
    """Parâmetros de construção e busca do índice HNSW."""
    M: int = 16
    ef_construction: int = 200
    ef_search: int = 100
    space: str = "cosine"  # hnswlib suporta: 'l2', 'ip', 'cosine'


class HNSWSearch:
    """
    Wrapper em torno do hnswlib.Index com uma API simples de
    construir / salvar / carregar / consultar, compatível com a
    interface comum que o Rafael vai definir na integração.
    """

    def __init__(self, dim: int, params: Optional[HNSWParams] = None):
        self.dim = dim
        self.params = params or HNSWParams()
        self.index: Optional[hnswlib.Index] = None
        self.n_elements = 0
        self.build_time_seconds: Optional[float] = None

    # ------------------------------------------------------------------
    # Construção
    # ------------------------------------------------------------------
    def build(self, embeddings: np.ndarray, ids: Optional[np.ndarray] = None) -> float:
        """
        Constrói o índice HNSW a partir da matriz de embeddings.

        Parameters
        ----------
        embeddings : np.ndarray, shape (N, dim)
        ids : np.ndarray, shape (N,), opcional
            IDs inteiros dos itens (default: 0..N-1)

        Returns
        -------
        build_time_seconds : float
        """
        n = embeddings.shape[0]
        if ids is None:
            ids = np.arange(n)

        self.index = hnswlib.Index(space=self.params.space, dim=self.dim)
        self.index.init_index(
            max_elements=n,
            M=self.params.M,
            ef_construction=self.params.ef_construction,
        )

        start = time.perf_counter()
        self.index.add_items(embeddings, ids)
        elapsed = time.perf_counter() - start

        self.index.set_ef(self.params.ef_search)
        self.n_elements = n
        self.build_time_seconds = elapsed
        return elapsed

    # ------------------------------------------------------------------
    # Consulta
    # ------------------------------------------------------------------
    def query(self, query_vector: np.ndarray, top_k: int = 5):
        """
        Busca os top_k vizinhos mais próximos de um único vetor de consulta.

        Returns
        -------
        labels : np.ndarray, shape (top_k,)
        distances : np.ndarray, shape (top_k,)
            Para space='cosine', hnswlib retorna (1 - similaridade_cosseno).
        """
        if self.index is None:
            raise RuntimeError("Índice ainda não foi construído/carregado. Chame build() ou load().")

        query_vector = np.asarray(query_vector).reshape(1, -1)
        labels, distances = self.index.knn_query(query_vector, k=top_k)
        return labels[0], distances[0]

    def batch_query(self, query_vectors: np.ndarray, top_k: int = 5):
        """Consulta em lote — usada nos benchmarks para medir tempo médio."""
        if self.index is None:
            raise RuntimeError("Índice ainda não foi construído/carregado.")
        labels, distances = self.index.knn_query(query_vectors, k=top_k)
        return labels, distances

    def set_ef_search(self, ef_search: int) -> None:
        """Permite trocar ef_search sem reconstruir o índice (só afeta busca)."""
        if self.index is None:
            raise RuntimeError("Índice ainda não foi construído/carregado.")
        self.index.set_ef(ef_search)
        self.params.ef_search = ef_search

    # ------------------------------------------------------------------
    # Persistência
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Salva o índice binário + um .json com metadados (params, dim, n)."""
        if self.index is None:
            raise RuntimeError("Nada para salvar: índice não construído.")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.index.save_index(path)

        meta_path = path + ".meta.json"
        meta = {
            "dim": self.dim,
            "n_elements": self.n_elements,
            "build_time_seconds": self.build_time_seconds,
            "params": asdict(self.params),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "HNSWSearch":
        """Carrega um índice salvo anteriormente, junto com seus metadados."""
        meta_path = path + ".meta.json"
        with open(meta_path) as f:
            meta = json.load(f)

        params = HNSWParams(**meta["params"])
        obj = cls(dim=meta["dim"], params=params)
        obj.index = hnswlib.Index(space=params.space, dim=meta["dim"])
        obj.index.load_index(path, max_elements=meta["n_elements"])
        obj.index.set_ef(params.ef_search)
        obj.n_elements = meta["n_elements"]
        obj.build_time_seconds = meta["build_time_seconds"]
        return obj

    # ------------------------------------------------------------------
    # Métricas auxiliares
    # ------------------------------------------------------------------
    def index_memory_bytes(self) -> int:
        """
        Estimativa de memória usada pelo índice.
        hnswlib não expõe isso diretamente, então aproximamos pelo tamanho
        do arquivo salvo em disco (proxy razoável e simples de justificar
        nos slides).
        """
        tmp_path = "/tmp/_hnsw_mem_probe.bin"
        self.save(tmp_path)
        size = os.path.getsize(tmp_path)
        os.remove(tmp_path)
        os.remove(tmp_path + ".meta.json")
        return size


def recall_at_k(
    hnsw_indices: np.ndarray,
    exact_indices: np.ndarray,
) -> float:
    """
    Calcula recall@k médio entre resultados do HNSW e da busca exata.

    Parameters
    ----------
    hnsw_indices : np.ndarray, shape (n_queries, k)
    exact_indices : np.ndarray, shape (n_queries, k)
        Ground truth vindo de cosine_search.batch_exact_cosine_search.

    Returns
    -------
    recall : float
        Fração média de itens do top_k exato que também aparecem
        no top_k retornado pelo HNSW.
    """
    n_queries, k = exact_indices.shape
    recalls = np.empty(n_queries)
    for i in range(n_queries):
        hits = len(set(hnsw_indices[i].tolist()) & set(exact_indices[i].tolist()))
        recalls[i] = hits / k
    return float(recalls.mean())
