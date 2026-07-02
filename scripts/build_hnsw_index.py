"""
build_hnsw_index.py

Entregável: scripts/build_hnsw_index.py

Lê os embeddings (por padrão, Sentence Embeddings da Dyesi) e constrói
o índice HNSW, salvando o binário em artifacts/hnsw_index.bin.

Uso:
    python scripts/build_hnsw_index.py \
        --embeddings artifacts/sentence_embeddings.npy \
        --output artifacts/hnsw_index.bin \
        --M 16 --ef_construction 200 --ef_search 100
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.search.hnsw_search import HNSWSearch, HNSWParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", default="artifacts/sentence_embeddings.npy",
                         help="Caminho do .npy com os embeddings a indexar")
    parser.add_argument("--output", default="artifacts/hnsw_index.bin",
                         help="Caminho de saída do índice HNSW")
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--ef_construction", type=int, default=200)
    parser.add_argument("--ef_search", type=int, default=100)
    args = parser.parse_args()

    print(f"Carregando embeddings de {args.embeddings} ...")
    embeddings = np.load(args.embeddings).astype(np.float32)
    n, dim = embeddings.shape
    print(f"  -> {n} vetores de dimensão {dim}")

    params = HNSWParams(
        M=args.M,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
    )

    index = HNSWSearch(dim=dim, params=params)
    print(f"Construindo índice HNSW (M={params.M}, ef_construction={params.ef_construction}) ...")
    elapsed = index.build(embeddings)
    print(f"  -> construído em {elapsed:.4f} segundos")

    index.save(args.output)
    print(f"Índice salvo em {args.output} (+ {args.output}.meta.json)")


if __name__ == "__main__":
    main()
