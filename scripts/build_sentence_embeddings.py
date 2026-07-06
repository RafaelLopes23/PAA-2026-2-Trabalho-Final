"""
Gera sentence embeddings para as sinopses processadas.

Implementacao integrada a partir da branch da Dyesi, com pequenos ajustes:
- output padrao corrigido para `artifacts/sentence_embeddings.npy`;
- fallback para CSV caso o ambiente esteja sem suporte a Parquet;
- arquivo de metricas salvo em JSON para os slides.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.embeddings.sentence_embeddings import SentenceEmbeddingPipeline  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_movies_dataframe(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    try:
        return pd.read_parquet(path)
    except Exception:
        csv_fallback = path.with_suffix(".csv")
        if csv_fallback.exists():
            return pd.read_csv(csv_fallback)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/movies.parquet", help="Base processada do Welder")
    parser.add_argument("--output", default="artifacts/sentence_embeddings.npy", help="Arquivo .npy de saida")
    parser.add_argument(
        "--stats-output",
        default="artifacts/sentence_embeddings_stats.json",
        help="Arquivo JSON com metricas para os slides",
    )
    parser.add_argument("--model-name", default="all-MiniLM-L6-v2", help="Modelo do sentence-transformers")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    input_path = PROJECT_ROOT / args.input
    output_path = PROJECT_ROOT / args.output
    stats_path = PROJECT_ROOT / args.stats_output

    print("--- MODULO 3: SENTENCE EMBEDDINGS (DYESI) ---")
    if not input_path.exists():
        raise FileNotFoundError(
            f"O arquivo {input_path} nao foi encontrado. Rode o preprocessamento antes."
        )

    print(f"Carregando dados de {input_path} ...")
    df = load_movies_dataframe(input_path)
    synopses = df["synopsis"].fillna("").tolist()
    print(f"  -> {len(synopses)} sinopses carregadas.")

    print(f"Inicializando o modelo '{args.model_name}' (licenca Apache 2.0)...")
    pipeline = SentenceEmbeddingPipeline(model_name=args.model_name)

    print(f"Gerando embeddings em lotes (batch_size={args.batch_size})...")
    start_time = time.perf_counter()
    embeddings = pipeline.model.encode(
        synopses,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    elapsed = time.perf_counter() - start_time

    n_vectors, dim = embeddings.shape
    print(f"  -> Embeddings calculados em {elapsed:.4f} segundos.")
    print(f"  -> Dimensao da matriz resultante: {n_vectors} vetores x {dim} dimensoes")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)
    print(f"Matriz salva com sucesso em: {output_path}")

    stats = {
        "model": args.model_name,
        "license": "Apache 2.0",
        "total_vectors": int(n_vectors),
        "dimension": int(dim),
        "generation_time_seconds": round(elapsed, 4),
        "memory_ram_mb": round(embeddings.nbytes / (1024 * 1024), 2),
    }
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Estatisticas salvas em {stats_path}")


if __name__ == "__main__":
    main()
