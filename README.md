# PAA 2026.2 — Sistema Q&A sobre Filmes

Trabalho final da disciplina de Projeto e Análise de Algoritmos (PAA). O objetivo é desenvolver um sistema de perguntas e respostas sobre filmes, utilizando o [CMU Movie Summary Corpus](https://www.cs.cmu.edu/~ark/movie/), com busca semântica nas sinopses e formatação de respostas via LLM local.

## Visão geral da arquitetura

```
Pergunta (linguagem natural)
        │
        ▼
  Embedding da pergunta          ← Dyesi (Sentence) / Pedro (Word2Vec, opcional)
        │
        ▼
  Busca por similaridade         ← Welder (cosseno exato) / Henrique (HNSW)
        │
        ▼
  Top-k sinopses relevantes
        │
        ▼
  LLM local formata resposta     ← Rafael (SmolLM, TinyLlama, Phi-3, Mistral, etc.)
```

O repositório está organizado em etapas encadeadas: **dados brutos → pré-processamento → embeddings → busca → Q&A**.

---

## Status do projeto

| Etapa | Responsável | Status | Entregável |
|-------|-------------|--------|------------|
| 1. Download e documentação do corpus | Welder | **Concluído** | `src/data/download_dataset.py`, `docs/DATASET.md` |
| 2. Pré-processamento e limpeza | Welder | **Concluído** | `src/data/preprocess.py`, `data/processed/movies.parquet` |
| 3. Busca exata por cosseno | Welder | **Concluído** | `src/search/cosine_search.py` |
| 4. Sentence Embeddings | Dyesi | **Pendente** | `artifacts/sentence_embeddings.npy` |
| 5. Word2Vec (opcional) | Pedro | **Pendente** | `artifacts/word2vec_embeddings.npy` |
| 6. Índice HNSW + benchmarks | Henrique | **Concluído** | `src/search/hnsw_search.py`, `scripts/run_benchmarks.py` |
| 7. Sistema Q&A com LLM | Rafael | **Pendente** | API/CLI de perguntas e respostas |

### Onde estamos agora

A **base de dados real** já foi processada (42.201 filmes com sinopse e título). A **busca exata** e o **HNSW** já foram integrados e testados com 42.201 vetores. O próximo passo crítico é a **Dyesi gerar os embeddings reais** das sinopses, alinhados linha a linha com o Parquet. Depois disso, o Rafael pode montar o fluxo de Q&A com o LLM local.

---

## O que já foi feito

### Welder — Dados e busca exata

- **Download do CMU Movie Summary Corpus** para `data/raw/MovieSummaries/` (via cópia local ou download oficial `.gz`).
- **Pré-processamento** com join de sinopses + metadados, limpeza de markup Wikipedia, tokenização básica e filtros de qualidade.
- **`movies.parquet`** com 42.201 filmes (`movie_id`, `title`, `synopsis`, `synopsis_tokens`, métricas de tamanho).
- **`preprocess_stats.json`** com números para slides (tamanhos, contagens, médias, tempo).
- **`cosine_search.py`** com busca exata genérica (Word2Vec ou Sentence Embeddings), complexidade **O(N × d)** por consulta.

**Métricas do pré-processamento** (`data/processed/preprocess_stats.json`):

| Métrica | Valor |
|---------|-------|
| Sinopses brutas | 42.306 |
| Metadados de filmes | 81.741 |
| Filmes processados | 42.201 |
| Tamanho bruto (plots + metadata) | ~88 MB |
| Média de caracteres na sinopse | 1.779,9 |
| Média de tokens na sinopse | 311,9 |
| Tempo de pré-processamento | ~6,8 s |
| Removidos (sem título) | 99 |
| Removidos (sem sinopse) | 6 |

### Henrique — HNSW e avaliação experimental

- Implementação de **`HNSWSearch`** (build, query, save/load) em `src/search/hnsw_search.py`.
- Scripts de **construção do índice** e **benchmarks** com grid search de parâmetros.
- Comparação HNSW vs busca exata com **recall@k**, tempo de query e gráficos em `results/`.
- Benchmarks já executados com **N = 42.201** e **d = 384** (embeddings de teste; substituir pelos reais da Dyesi).

Resultado observado com dados reais de contagem (embeddings aleatórios de validação):

- Busca exata: ~0,48 ms/query
- HNSW: ~0,04–0,36 ms/query (dependendo de `ef_search`)
- Recall@5 do HNSW varia de ~0,05 a ~0,53 conforme parâmetros

> Com embeddings reais e queries reais, os valores de recall e tempo devem ser reavaliados. Detalhes adicionais em [`README_henrique.md`](README_henrique.md).

---

## Estrutura do repositório

```
PAA-2026-2-Trabalho-Final/
├── README.md                      # Este arquivo
├── README_henrique.md             # Detalhes da parte HNSW/benchmarks
├── requirements.txt
├── docs/
│   └── DATASET.md                 # Documentação do CMU corpus
├── src/
│   ├── data/
│   │   ├── download_dataset.py    # Copia/baixa corpus → data/raw/
│   │   └── preprocess.py          # Gera movies.parquet + stats
│   └── search/
│       ├── cosine_search.py       # Busca exata O(N·d) — Welder
│       └── hnsw_search.py         # Índice HNSW — Henrique
├── scripts/
│   ├── generate_fake_data.py      # Dados sintéticos (só para testes)
│   ├── build_hnsw_index.py        # Constrói artifacts/hnsw_index.bin
│   └── run_benchmarks.py          # Grid search + gráficos
├── data/
│   ├── raw/MovieSummaries/        # Corpus bruto (gitignored, ~127 MB)
│   └── processed/
│       ├── movies.parquet         # Base processada (42.201 filmes)
│       ├── movies.csv             # Versão CSV opcional
│       └── preprocess_stats.json  # Métricas para slides
├── artifacts/                     # Embeddings e índice (gitignored)
└── results/                       # Benchmarks e gráficos (gitignored)
```

---

## Como executar

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Preparar dados (Welder)

```bash
# Copia corpus para data/raw/ e gera docs/DATASET.md
python3 -m src.data.download_dataset --include-optional

# Gera movies.parquet e preprocess_stats.json
python3 -m src.data.preprocess
```

> `data/raw/` não é versionado no git. Quem clonar o repositório precisa rodar o passo acima (ou apontar `--source` para o diretório local do corpus).

### 3. Gerar embeddings (Dyesi — pendente)

A Dyesi deve produzir `artifacts/sentence_embeddings.npy` com shape `(42201, d)`, onde a **linha i** corresponde à **linha i** de `movies.parquet`.

Opcionalmente, queries de teste em `artifacts/fake_queries.npy` com shape `(n_queries, d)`.

### 4. Construir índice e rodar benchmarks (Henrique)

```bash
python3 scripts/build_hnsw_index.py
python3 scripts/run_benchmarks.py --top_k 5
```

Saídas em `results/benchmark_results.csv` e `results/*.png`.

### 5. Dados sintéticos (apenas para testes isolados)

```bash
python3 scripts/generate_fake_data.py --n 5000 --dim 384
```

Use somente se ainda não houver `movies.parquet` real ou embeddings da Dyesi.

---

## Contrato de dados

### `movies.parquet`

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `movie_id` | int64 | Wikipedia movie ID (CMU) |
| `title` | str | Título do filme |
| `synopsis` | str | Sinopse limpa (texto para embeddings) |
| `synopsis_tokens` | str | Tokens separados por espaço (Word2Vec) |
| `synopsis_char_len` | int | Comprimento em caracteres |
| `synopsis_token_len` | int | Comprimento em tokens |

### Alinhamento embeddings ↔ Parquet

```
movies.parquet[i]  ↔  sentence_embeddings.npy[i]  ↔  word2vec_embeddings.npy[i]
```

A ordem das linhas é preservada pelo `preprocess.py` e **não deve ser alterada** ao gerar embeddings.

### Interface de busca (`cosine_search.py`)

```python
indices, scores = exact_cosine_search(query_vector, embeddings, top_k=5)
indices, scores = batch_exact_cosine_search(query_vectors, embeddings, top_k=5)
```

Genérica para qualquer dimensão `d` (Word2Vec, Sentence Embeddings, etc.).

---

## Análise de complexidade

| Etapa | Complexidade | Medição empírica (N≈42k) |
|-------|-------------|--------------------------|
| Pré-processamento | O(P · L) | ~6,8 s |
| Busca exata (1 query) | O(N · d) | ~0,48 ms (d=384) |
| Busca exata (50 queries) | O(Q · N · d) | ~24 ms |
| Build HNSW | O(N log N) | ~2–24 s (varia com M) |
| Query HNSW | O(log N) aprox. | ~0,04–0,36 ms |

O ganho do HNSW sobre busca exata depende de N, d e dos parâmetros (`M`, `ef_search`). Para N pequeno, a busca exata vetorizada com NumPy pode ser competitiva ou mais rápida — isso é esperado e vale documentar nos slides.

---

## Próximos passos

1. **Dyesi** — gerar `artifacts/sentence_embeddings.npy` real a partir das sinopses.
2. **Pedro** *(opcional)* — gerar `artifacts/word2vec_embeddings.npy` usando `synopsis_tokens`.
3. **Henrique** — re-rodar benchmarks com embeddings e queries reais.
4. **Rafael** — integrar retrieval + LLM local no sistema Q&A final.

---

## Referências

- [CMU Movie Summary Corpus](https://www.cs.cmu.edu/~ark/movie/)
- Bamman, O'Connor & Smith, *Learning Latent Personas of Film Characters*, ACL 2013
- Documentação do dataset: [`docs/DATASET.md`](docs/DATASET.md)
