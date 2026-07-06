"""
Treina o Word2Vec Average e gera a matriz de embeddings das sinopses.

Este script foi integrado ao preprocessamento do Welder. Se a coluna
`synopsis_tokens` ja existir em `movies.parquet`, ela e reaproveitada
diretamente para treino; caso contrario, o script faz uma tokenizacao basica.
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

from src.embeddings.word2vec_average import (  # noqa: E402
    Word2VecAverageEmbedder,
    Word2VecTrainer,
    clean_text,
    normalize_tokens,
)
from src.search.cosine_search import exact_cosine_search  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_file_size_mb(filepath: Path) -> float:
    if filepath.exists():
        return filepath.stat().st_size / (1024 * 1024)
    return 0.0


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


def build_corpus(df: pd.DataFrame) -> list[list[str]]:
    if "synopsis_tokens" in df.columns:
        return [normalize_tokens(text) for text in df["synopsis_tokens"].fillna("")]
    return [clean_text(text) for text in df["synopsis"].fillna("")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/processed/movies.parquet")
    parser.add_argument("--vocab_size", type=int, default=5000)
    parser.add_argument("--embed_dim", type=int, default=100)
    parser.add_argument("--window_size", type=int, default=3)
    parser.add_argument("--n_negatives", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--init_lr", type=float, default=0.025)
    parser.add_argument("--max_docs", type=int, default=None)
    parser.add_argument("--output", default="artifacts/word2vec_embeddings.npy")
    parser.add_argument("--model-output", default="artifacts/word2vec_embeddings.model.npz")
    parser.add_argument("--stats-output", default="artifacts/word2vec_metrics.json")
    args = parser.parse_args()

    dataset_path = PROJECT_ROOT / args.dataset
    output_path = PROJECT_ROOT / args.output
    model_path = PROJECT_ROOT / args.model_output
    stats_path = PROJECT_ROOT / args.stats_output

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Base processada ausente em {dataset_path}. Rode primeiro: python -m src.data.preprocess"
        )

    print(f"Carregando base processada: {dataset_path}")
    df = load_movies_dataframe(dataset_path)
    print(f"  -> {len(df)} filmes carregados")

    print("Preparando corpus para treino...")
    tokenize_start = time.perf_counter()
    corpus = build_corpus(df)
    tokenize_seconds = time.perf_counter() - tokenize_start
    print(f"  -> tokenizacao/pronto em {tokenize_seconds:.2f}s")

    trainer = Word2VecTrainer(
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        window_size=args.window_size,
        n_negatives=args.n_negatives,
        init_lr=args.init_lr,
        epochs=args.epochs,
        seed=42,
    )

    print("Construindo vocabulario...")
    trainer.build_vocab(corpus)
    print("Treinando Word2Vec...")
    train_seconds = trainer.train(corpus, max_docs=args.max_docs)
    print(f"  -> treino concluido em {train_seconds:.2f}s")

    if trainer.W_in is None:
        raise RuntimeError("Treino finalizado sem matriz W_in.")

    embedder = Word2VecAverageEmbedder(trainer.W_in, trainer.word_to_idx)
    texts = (
        df["synopsis_tokens"].fillna("").tolist()
        if "synopsis_tokens" in df.columns
        else df["synopsis"].fillna("").tolist()
    )

    print("Gerando embeddings medios das sinopses...")
    embeddings, vectorize_seconds = embedder.embed_batch(texts)
    print(f"  -> matriz gerada com shape {embeddings.shape} em {vectorize_seconds:.2f}s")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)
    embedder.save_model(str(model_path))

    emb_size_mb = get_file_size_mb(output_path)
    model_size_mb = get_file_size_mb(model_path)

    test_queries = [
        "space travel spaceship sci fi alien battle",
        "romantic comedy drama love wedding relationship",
        "detective police murder crime investigation killer",
    ]
    query_times_ms: list[float] = []
    for query in test_queries:
        query_start = time.perf_counter()
        query_vector = embedder.embed_query(query)
        _indices, _scores = exact_cosine_search(query_vector, embeddings, top_k=5)
        query_times_ms.append((time.perf_counter() - query_start) * 1000.0)

    stats = {
        "dataset": str(dataset_path),
        "n_movies": int(len(df)),
        "vocab_size": int(len(trainer.word_to_idx)),
        "embedding_dim": args.embed_dim,
        "tokenize_seconds": round(tokenize_seconds, 4),
        "train_seconds": round(train_seconds, 4),
        "vectorize_seconds": round(vectorize_seconds, 4),
        "embeddings_shape": list(embeddings.shape),
        "embeddings_size_mb": round(emb_size_mb, 4),
        "model_size_mb": round(model_size_mb, 4),
        "avg_query_time_ms": round(float(np.mean(query_times_ms)), 4),
    }
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Embeddings salvos em: {output_path}")
    print(f"Modelo salvo em: {model_path}")
    print(f"Metricas salvas em: {stats_path}")


if __name__ == "__main__":
    main()
