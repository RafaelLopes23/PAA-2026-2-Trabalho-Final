"""
preprocess.py

Lê plot_summaries.txt e movie.metadata.tsv, combina, limpa, tokeniza
e exporta data/processed/movies.parquet com estatísticas para slides.

Uso:
    python -m src.data.preprocess
    python -m src.data.preprocess --also-csv
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "MovieSummaries"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "movies.parquet"
DEFAULT_STATS = PROJECT_ROOT / "data" / "processed" / "preprocess_stats.json"

MIN_SYNOPSIS_CHARS = 50

WIKI_TEMPLATE_RE = re.compile(r"\{\{[^}]*\}\}")
WIKI_LINK_RE = re.compile(r"\[\[([^|\]]+)(?:\|([^\]]+))?\]\]")
WHITESPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[a-z0-9']+")


def clean_synopsis(text: str) -> str:
    text = WIKI_TEMPLATE_RE.sub(" ", text)
    text = WIKI_LINK_RE.sub(lambda m: (m.group(2) or m.group(1)).strip(), text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def tokenize(text: str) -> str:
    tokens = TOKEN_RE.findall(text.lower())
    return " ".join(tokens)


def load_plot_summaries(path: Path) -> pd.DataFrame:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            wiki_id_str, synopsis = line.split("\t", 1)
            rows.append({"movie_id": wiki_id_str.strip(), "synopsis_raw": synopsis})
    return pd.DataFrame(rows)


def load_movie_metadata(path: Path) -> pd.DataFrame:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 9:
                continue
            rows.append(
                {
                    "movie_id": parts[0].strip(),
                    "title": parts[2].strip(),
                    "release_date": parts[3].strip(),
                    "genres": ast.literal_eval(parts[8]) if parts[8] else {},
                }
            )
    return pd.DataFrame(rows)


def preprocess(
    raw_dir: Path,
    output_path: Path,
    stats_path: Path,
    also_csv: bool = False,
) -> pd.DataFrame:
    start = time.perf_counter()

    plots_path = raw_dir / "plot_summaries.txt"
    meta_path = raw_dir / "movie.metadata.tsv"
    if not plots_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Arquivos brutos ausentes em {raw_dir}. Rode: python -m src.data.download_dataset"
        )

    raw_sizes = {
        "plot_summaries_bytes": plots_path.stat().st_size,
        "movie_metadata_bytes": meta_path.stat().st_size,
    }

    plots_df = load_plot_summaries(plots_path)
    meta_df = load_movie_metadata(meta_path)

    plot_count = len(plots_df)
    metadata_count = len(meta_df)

    merged = plots_df.merge(meta_df[["movie_id", "title"]], on="movie_id", how="left")

    removed_invalid_id = 0
    valid_mask = merged["movie_id"].str.fullmatch(r"\d+")
    removed_invalid_id = int((~valid_mask).sum())
    merged = merged[valid_mask].copy()
    merged["movie_id"] = merged["movie_id"].astype("int64")

    removed_duplicates = int(merged.duplicated(subset=["movie_id"]).sum())
    merged = merged.drop_duplicates(subset=["movie_id"], keep="first")

    removed_no_synopsis = int(
        (merged["synopsis_raw"].isna() | (merged["synopsis_raw"].str.strip() == "")).sum()
    )
    merged = merged[merged["synopsis_raw"].notna() & (merged["synopsis_raw"].str.strip() != "")]

    merged["synopsis"] = merged["synopsis_raw"].map(clean_synopsis)
    removed_empty_after_clean = int((merged["synopsis"].str.strip() == "").sum())
    merged = merged[merged["synopsis"].str.strip() != ""]

    removed_too_short = int((merged["synopsis"].str.len() < MIN_SYNOPSIS_CHARS).sum())
    merged = merged[merged["synopsis"].str.len() >= MIN_SYNOPSIS_CHARS]

    removed_no_title = int((merged["title"].isna() | (merged["title"].str.strip() == "")).sum())
    merged = merged[merged["title"].notna() & (merged["title"].str.strip() != "")]

    merged["synopsis_tokens"] = merged["synopsis"].map(tokenize)
    merged["synopsis_char_len"] = merged["synopsis"].str.len()
    merged["synopsis_token_len"] = merged["synopsis_tokens"].str.split().str.len()

    result = merged[["movie_id", "title", "synopsis", "synopsis_tokens", "synopsis_char_len", "synopsis_token_len"]]
    result = result.reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)

    if also_csv:
        csv_path = output_path.with_suffix(".csv")
        result.to_csv(csv_path, index=False)
        print(f"[ok] {csv_path.relative_to(PROJECT_ROOT)}")

    elapsed = time.perf_counter() - start

    stats = {
        "raw": {
            "plot_summaries_count": plot_count,
            "movie_metadata_count": metadata_count,
            "plot_summaries_bytes": raw_sizes["plot_summaries_bytes"],
            "movie_metadata_bytes": raw_sizes["movie_metadata_bytes"],
            "total_size_bytes": raw_sizes["plot_summaries_bytes"] + raw_sizes["movie_metadata_bytes"],
        },
        "processed": {
            "movies_count": len(result),
            "parquet_size_bytes": output_path.stat().st_size,
            "avg_synopsis_chars": round(float(result["synopsis_char_len"].mean()), 1),
            "avg_synopsis_tokens": round(float(result["synopsis_token_len"].mean()), 1),
            "removed_invalid_id": removed_invalid_id,
            "removed_duplicates": removed_duplicates,
            "removed_no_synopsis": removed_no_synopsis + removed_empty_after_clean,
            "removed_too_short": removed_too_short,
            "removed_no_title": removed_no_title,
        },
        "timing": {
            "preprocess_seconds": round(elapsed, 3),
        },
    }

    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[ok] {output_path.relative_to(PROJECT_ROOT)} ({len(result):,} filmes)")
    print(f"[ok] {stats_path.relative_to(PROJECT_ROOT)}")
    print(f"[tempo] {elapsed:.2f}s")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Pré-processa o CMU Movie Summary Corpus")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--also-csv", action="store_true")
    args = parser.parse_args()

    preprocess(args.raw_dir, args.output, args.stats, also_csv=args.also_csv)


if __name__ == "__main__":
    main()
