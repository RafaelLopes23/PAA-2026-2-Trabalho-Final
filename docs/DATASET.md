# CMU Movie Summary Corpus

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
| `plot_summaries.txt` | 72.4 MB, 42,306 linhas |
| `movie.metadata.tsv` | 15.5 MB, 81,741 linhas |
| `README.txt` | 0.0 MB, 79 linhas |
| `character.metadata.tsv` | 39.6 MB, 450,669 linhas |
| `name.clusters.txt` | 0.1 MB, 2,666 linhas |
| `tvtropes.clusters.txt` | 0.1 MB, 501 linhas |

## Formato

- **Encoding:** UTF-8
- **Delimitador:** tab (`\t`)
- **Cabeçalho:** nenhum (colunas definidas no README oficial)

### plot_summaries.txt

Duas colunas por linha:

```
<Wikipedia_movie_ID>\t<plot_summary_text>
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
