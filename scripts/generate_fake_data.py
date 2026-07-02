"""
generate_fake_data.py

Script AUXILIAR (não é entregável oficial) para gerar dados falsos
compatíveis com o formato esperado do Welder (movies.parquet) e da
Dyesi (sentence_embeddings.npy), para o Henrique poder desenvolver e
testar hnsw_search.py / build_hnsw_index.py / run_benchmarks.py sem
precisar esperar os outros módulos ficarem prontos.

Assim que os arquivos reais existirem, é só apagar/ignorar este script
e apontar os outros scripts para os arquivos de verdade.

Uso:
    python scripts/generate_fake_data.py --n 5000 --dim 384
"""

import argparse
import os
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000, help="Número de filmes falsos")
    parser.add_argument("--dim", type=int, default=384, help="Dimensão dos embeddings (384 é comum p/ MiniLM)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # --- movies.parquet (simula entregável do Welder) ---
    os.makedirs("data/processed", exist_ok=True)
    df = pd.DataFrame({
        "movie_id": np.arange(args.n),
        "title": [f"Fake Movie {i}" for i in range(args.n)],
        "synopsis": [f"This is a fake synopsis for movie number {i}." for i in range(args.n)],
    })
    df.to_parquet("data/processed/movies.parquet", index=False)
    print(f"[ok] data/processed/movies.parquet criado com {args.n} filmes")

    # --- sentence_embeddings.npy (simula entregável da Dyesi) ---
    os.makedirs("artifacts", exist_ok=True)
    sentence_embeddings = rng.normal(size=(args.n, args.dim)).astype(np.float32)
    np.save("artifacts/sentence_embeddings.npy", sentence_embeddings)
    print(f"[ok] artifacts/sentence_embeddings.npy criado com shape {sentence_embeddings.shape}")

    # --- word2vec_embeddings.npy (simula entregável do Pedro, dimensão menor) ---
    w2v_dim = 100
    word2vec_embeddings = rng.normal(size=(args.n, w2v_dim)).astype(np.float32)
    np.save("artifacts/word2vec_embeddings.npy", word2vec_embeddings)
    print(f"[ok] artifacts/word2vec_embeddings.npy criado com shape {word2vec_embeddings.shape}")

    # --- algumas queries falsas pra testar consultas ---
    n_queries = 50
    fake_queries = rng.normal(size=(n_queries, args.dim)).astype(np.float32)
    np.save("artifacts/fake_queries.npy", fake_queries)
    print(f"[ok] artifacts/fake_queries.npy criado com shape {fake_queries.shape} (só para teste)")


if __name__ == "__main__":
    main()
