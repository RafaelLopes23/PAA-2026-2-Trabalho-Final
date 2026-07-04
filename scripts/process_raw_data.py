"""
process_raw_data.py

Lê os arquivos brutos da pasta data/MovieSummaries/ (extraídos do tarball),
combina os títulos das metadados com as sinopses de plot_summaries.txt
e gera o arquivo data/processed/movies.parquet.
"""

import os
import pandas as pd

def main():
    raw_metadata_path = "data/MovieSummaries/movie.metadata.tsv"
    raw_plots_path = "data/MovieSummaries/plot_summaries.txt"
    output_parquet_path = "data/processed/movies.parquet"

    if not os.path.exists(raw_metadata_path) or not os.path.exists(raw_plots_path):
        print("Erro: Arquivos brutos não encontrados em data/MovieSummaries/")
        return

    print("Carregando metadados dos filmes (ID e Título) ...")
    # Coluna 0 é o Wikipedia movie ID, Coluna 2 é o Movie name (título)
    metadata = pd.read_csv(
        raw_metadata_path,
        sep="\t",
        header=None,
        usecols=[0, 2],
        names=["movie_id", "title"],
        dtype={"movie_id": int, "title": str}
    )

    print("Carregando sinopses dos filmes ...")
    plots = []
    with open(raw_plots_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                plots.append((int(parts[0]), parts[1]))

    df_plots = pd.DataFrame(plots, columns=["movie_id", "synopsis"])

    print("Cruzando dados por movie_id ...")
    df_merged = pd.merge(df_plots, metadata, on="movie_id", how="inner")

    # Garante a ordem correta das colunas
    df_merged = df_merged[["movie_id", "title", "synopsis"]]

    os.makedirs(os.path.dirname(output_parquet_path), exist_ok=True)
    df_merged.to_parquet(output_parquet_path, index=False)
    print(f"Sucesso! {len(df_merged)} filmes processados e salvos em {output_parquet_path}")

if __name__ == "__main__":
    main()
