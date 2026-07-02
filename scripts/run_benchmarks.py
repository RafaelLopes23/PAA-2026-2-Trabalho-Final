"""
run_benchmarks.py

Entregável: scripts/run_benchmarks.py

Roda um grid de parâmetros do HNSW (M, ef_construction, ef_search),
compara cada configuração com a busca exata por cosseno (baseline/gabarito)
e salva:

  - results/benchmark_results.csv  (uma linha por configuração testada)
  - results/recall_vs_ef_search.png
  - results/query_time_hnsw_vs_exact.png
  - results/build_time_vs_M.png

Uso:
    python scripts/run_benchmarks.py \
        --embeddings artifacts/sentence_embeddings.npy \
        --queries artifacts/fake_queries.npy \
        --top_k 5
"""

import argparse
import os
import sys
import time
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.search.hnsw_search import HNSWSearch, HNSWParams, recall_at_k
from src.search.cosine_search import batch_exact_cosine_search


# Grid de parâmetros a testar. Ajuste conforme o tempo disponível:
# comece pequeno para garantir que roda, depois expanda se sobrar tempo.
M_VALUES = [8, 16, 32]
EF_CONSTRUCTION_VALUES = [100, 200]
EF_SEARCH_VALUES = [50, 100, 200]


def measure_exact_search(embeddings: np.ndarray, queries: np.ndarray, top_k: int):
    """Mede tempo da busca exata (baseline) e retorna também o ground truth."""
    start = time.perf_counter()
    exact_indices, exact_scores = batch_exact_cosine_search(queries, embeddings, top_k=top_k)
    elapsed = time.perf_counter() - start
    avg_query_time = elapsed / len(queries)
    return exact_indices, exact_scores, avg_query_time


def run_single_config(embeddings, queries, exact_indices, top_k, M, ef_construction, ef_search):
    """Constrói um índice HNSW com uma configuração de parâmetros e mede tudo."""
    dim = embeddings.shape[1]
    params = HNSWParams(M=M, ef_construction=ef_construction, ef_search=ef_search)
    index = HNSWSearch(dim=dim, params=params)

    build_time = index.build(embeddings)
    memory_bytes = index.index_memory_bytes()

    # tempo médio de consulta (batch, depois divide pelo número de queries)
    start = time.perf_counter()
    hnsw_labels, hnsw_dists = index.batch_query(queries, top_k=top_k)
    elapsed = time.perf_counter() - start
    avg_query_time = elapsed / len(queries)

    recall = recall_at_k(hnsw_labels, exact_indices)

    return {
        "method": "hnsw",
        "M": M,
        "ef_construction": ef_construction,
        "ef_search": ef_search,
        "n_vectors": embeddings.shape[0],
        "dim": dim,
        "build_time_seconds": round(build_time, 6),
        "memory_bytes": memory_bytes,
        "avg_query_time_seconds": round(avg_query_time, 8),
        "recall_at_k": round(recall, 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", default="artifacts/sentence_embeddings.npy")
    parser.add_argument("--queries", default="artifacts/fake_queries.npy",
                         help=".npy com vetores de consulta (mesma dimensão dos embeddings)")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--output_csv", default="results/benchmark_results.csv")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    print(f"Carregando embeddings de {args.embeddings} ...")
    embeddings = np.load(args.embeddings).astype(np.float32)
    print(f"  -> {embeddings.shape[0]} vetores, dim={embeddings.shape[1]}")

    print(f"Carregando queries de {args.queries} ...")
    queries = np.load(args.queries).astype(np.float32)
    print(f"  -> {queries.shape[0]} queries")

    # 1) Baseline: busca exata
    print("\n=== Busca exata (baseline / gabarito) ===")
    exact_indices, exact_scores, exact_avg_time = measure_exact_search(embeddings, queries, args.top_k)
    print(f"Tempo médio de consulta (exata): {exact_avg_time:.6f} s")

    rows = [{
        "method": "exact",
        "M": "",
        "ef_construction": "",
        "ef_search": "",
        "n_vectors": embeddings.shape[0],
        "dim": embeddings.shape[1],
        "build_time_seconds": 0.0,
        "memory_bytes": embeddings.nbytes,
        "avg_query_time_seconds": round(exact_avg_time, 8),
        "recall_at_k": 1.0,  # é o próprio gabarito
    }]

    # 2) Grid de configurações HNSW
    print("\n=== Grid search HNSW ===")
    total_configs = len(M_VALUES) * len(EF_CONSTRUCTION_VALUES) * len(EF_SEARCH_VALUES)
    count = 0
    for M in M_VALUES:
        for ef_construction in EF_CONSTRUCTION_VALUES:
            for ef_search in EF_SEARCH_VALUES:
                count += 1
                print(f"[{count}/{total_configs}] M={M}, ef_construction={ef_construction}, ef_search={ef_search} ...")
                result = run_single_config(
                    embeddings, queries, exact_indices, args.top_k,
                    M, ef_construction, ef_search,
                )
                rows.append(result)
                print(f"    build={result['build_time_seconds']:.4f}s  "
                      f"query={result['avg_query_time_seconds']:.6f}s  "
                      f"recall@{args.top_k}={result['recall_at_k']:.3f}  "
                      f"mem={result['memory_bytes']/1024:.1f}KB")

    # 3) Salva CSV
    fieldnames = list(rows[0].keys())
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResultados salvos em {args.output_csv}")

    # 4) Gráficos para os slides
    generate_plots(rows, exact_avg_time, args.top_k)
    print("Gráficos salvos em results/*.png")


def generate_plots(rows, exact_avg_time, top_k):
    hnsw_rows = [r for r in rows if r["method"] == "hnsw"]

    # --- Gráfico 1: Recall@k vs ef_search (uma linha por M, ef_construction fixo no maior valor) ---
    plt.figure(figsize=(7, 5))
    ef_const_fixed = max(EF_CONSTRUCTION_VALUES)
    for M in M_VALUES:
        subset = [r for r in hnsw_rows if r["M"] == M and r["ef_construction"] == ef_const_fixed]
        subset.sort(key=lambda r: r["ef_search"])
        xs = [r["ef_search"] for r in subset]
        ys = [r["recall_at_k"] for r in subset]
        plt.plot(xs, ys, marker="o", label=f"M={M}")
    plt.xlabel("ef_search")
    plt.ylabel(f"Recall@{top_k}")
    plt.title(f"Recall@{top_k} vs ef_search (ef_construction={ef_const_fixed})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/recall_vs_ef_search.png", dpi=150)
    plt.close()

    # --- Gráfico 2: tempo de consulta HNSW vs exato ---
    plt.figure(figsize=(7, 5))
    best_recall_row = max(hnsw_rows, key=lambda r: r["recall_at_k"])
    labels = ["Busca exata", f"HNSW\n(M={best_recall_row['M']}, ef_search={best_recall_row['ef_search']})"]
    times = [exact_avg_time, best_recall_row["avg_query_time_seconds"]]
    plt.bar(labels, times, color=["#888888", "#4c72b0"])
    plt.ylabel("Tempo médio de consulta (s)")
    plt.title("Tempo de consulta: exata vs HNSW (melhor recall)")
    for i, t in enumerate(times):
        plt.text(i, t, f"{t:.6f}s", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig("results/query_time_hnsw_vs_exact.png", dpi=150)
    plt.close()

    # --- Gráfico 3: tempo de construção vs M ---
    plt.figure(figsize=(7, 5))
    for ef_construction in EF_CONSTRUCTION_VALUES:
        subset = [r for r in hnsw_rows if r["ef_construction"] == ef_construction]
        agg = {}
        for r in subset:
            agg.setdefault(r["M"], []).append(r["build_time_seconds"])
        xs = sorted(agg.keys())
        ys = [np.mean(agg[m]) for m in xs]
        plt.plot(xs, ys, marker="o", label=f"ef_construction={ef_construction}")
    plt.xlabel("M")
    plt.ylabel("Tempo de construção (s)")
    plt.title("Tempo de construção do índice vs M")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/build_time_vs_M.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
