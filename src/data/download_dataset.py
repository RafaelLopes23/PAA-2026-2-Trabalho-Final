"""
download_dataset.py

Baixa (ou copia) o CMU Movie Summary Corpus para data/raw/MovieSummaries/
e gera a documentação em docs/DATASET.md.

Uso:
    python -m src.data.download_dataset
    python -m src.data.download_dataset --source /caminho/para/MovieSummaries
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path.home() / "Downloads" / "MovieSummaries"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "MovieSummaries"
DOCS_PATH = PROJECT_ROOT / "docs" / "DATASET.md"

REQUIRED_FILES = (
    "plot_summaries.txt",
    "movie.metadata.tsv",
    "README.txt",
)

OPTIONAL_FILES = (
    "character.metadata.tsv",
    "name.clusters.txt",
    "tvtropes.clusters.txt",
)

DOWNLOAD_URLS = {
    "plot_summaries.txt": "https://www.cs.cmu.edu/~ark/movie/plot_summaries.txt.gz",
    "movie.metadata.tsv": "https://www.cs.cmu.edu/~ark/movie/movie.metadata.tsv.gz",
    "character.metadata.tsv": "https://www.cs.cmu.edu/~ark/movie/character.metadata.tsv.gz",
}

EXPECTED_MIN_LINES = {
    "plot_summaries.txt": 42_000,
    "movie.metadata.tsv": 81_000,
}


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"[ok] copiado: {dst.relative_to(PROJECT_ROOT)}")


def download_and_decompress(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    gz_path = dst.with_suffix(dst.suffix + ".gz")
    print(f"[download] {url}")
    urllib.request.urlretrieve(url, gz_path)
    with gzip.open(gz_path, "rb") as src, dst.open("wb") as out:
        shutil.copyfileobj(src, out)
    gz_path.unlink(missing_ok=True)
    print(f"[ok] baixado: {dst.relative_to(PROJECT_ROOT)}")


def ensure_dataset(source_dir: Path | None, include_optional: bool) -> dict:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    files_to_copy = list(REQUIRED_FILES)
    if include_optional:
        files_to_copy.extend(OPTIONAL_FILES)

    if source_dir and source_dir.exists():
        for name in files_to_copy:
            src = source_dir / name
            if not src.exists():
                if name in REQUIRED_FILES:
                    raise FileNotFoundError(f"Arquivo obrigatório ausente em {source_dir}: {name}")
                print(f"[skip] opcional ausente: {name}")
                continue
            copy_file(src, RAW_DIR / name)
    else:
        missing = [name for name in REQUIRED_FILES if not (RAW_DIR / name).exists()]
        if missing:
            print("[info] arquivos ausentes em data/raw/; tentando download oficial...")
            for name in missing:
                url = DOWNLOAD_URLS.get(name)
                if url is None:
                    raise FileNotFoundError(
                        f"Arquivo obrigatório ausente e sem URL de download: {name}"
                    )
                download_and_decompress(url, RAW_DIR / name)

    stats: dict = {"files": {}}
    for name in files_to_copy:
        path = RAW_DIR / name
        if not path.exists():
            continue
        line_count = count_lines(path) if path.suffix in {".txt", ".tsv"} else None
        stats["files"][name] = {
            "size_bytes": path.stat().st_size,
            "line_count": line_count,
        }

    for name, min_lines in EXPECTED_MIN_LINES.items():
        info = stats["files"].get(name)
        if info and info["line_count"] is not None and info["line_count"] < min_lines:
            raise ValueError(
                f"{name}: esperado >= {min_lines} linhas, encontrado {info['line_count']}"
            )

    return stats


def write_dataset_doc(stats: dict) -> None:
    DOCS_PATH.parent.mkdir(parents=True, exist_ok=True)

    file_lines = []
    for name, info in stats["files"].items():
        size_mb = info["size_bytes"] / (1024 * 1024)
        lines = info["line_count"]
        line_info = f", {lines:,} linhas" if lines is not None else ""
        file_lines.append(f"| `{name}` | {size_mb:.1f} MB{line_info} |")

    content = f"""# CMU Movie Summary Corpus

## Fonte

- **Dataset:** [CMU Movie Summary Corpus](https://www.cs.cmu.edu/~ark/movie/)
- **Licença:** Creative Commons Attribution-ShareAlike
- **Referência:** David Bamman, Brendan O'Connor and Noah Smith, *Learning Latent Personas of Film Characters*, ACL 2013.

## Como obter os dados

```bash
python -m src.data.download_dataset
```

Por padrão, o script copia os arquivos de `~/Downloads/MovieSummaries`
para `data/raw/MovieSummaries/`. Se os arquivos não existirem localmente,
tenta baixar as versões `.gz` oficiais do site da CMU.

## Arquivos no projeto

| Arquivo | Tamanho |
|---------|---------|
{chr(10).join(file_lines)}

## Formato

- **Encoding:** UTF-8
- **Delimitador:** tab (`\\t`)
- **Cabeçalho:** nenhum (colunas definidas no README oficial)

### plot_summaries.txt

Duas colunas por linha:

```
<Wikipedia_movie_ID>\\t<plot_summary_text>
```

### movie.metadata.tsv

Nove colunas:

1. Wikipedia movie ID
2. Freebase movie ID
3. Movie name
4. Movie release date
5. Movie box office revenue
6. Movie runtime
7. Movie languages (dict Python)
8. Movie countries (dict Python)
9. Movie genres (dict Python)

### character.metadata.tsv (opcional)

Treze colunas de metadados de personagens. Não é usado no pré-processamento
inicial, mas fica disponível para trabalhos futuros.

## Próximo passo

```bash
python -m src.data.preprocess
```

Gera `data/processed/movies.parquet` e `data/processed/preprocess_stats.json`.
"""
    DOCS_PATH.write_text(content, encoding="utf-8")
    print(f"[ok] documentação: {DOCS_PATH.relative_to(PROJECT_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepara o CMU Movie Summary Corpus")
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Diretório de origem com os arquivos extraídos",
    )
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Copiar também character.metadata.tsv e arquivos de clusters",
    )
    args = parser.parse_args()

    stats = ensure_dataset(args.source, args.include_optional)
    write_dataset_doc(stats)

    total_bytes = sum(info["size_bytes"] for info in stats["files"].values())
    print(f"\n[resumo] {len(stats['files'])} arquivos, {total_bytes / (1024 * 1024):.1f} MB total")


if __name__ == "__main__":
    main()
