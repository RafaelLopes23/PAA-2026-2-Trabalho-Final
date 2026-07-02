# Parte do Henrique — HNSW e avaliação experimental

## Como rodar (ordem)

```bash
pip install hnswlib numpy pandas matplotlib pyarrow --break-system-packages

# 1) Enquanto Welder/Dyesi não entregam os arquivos reais, gere dados falsos:
python scripts/generate_fake_data.py --n 5000 --dim 384

# 2) Construir o índice HNSW (parâmetros default: M=16, ef_construction=200, ef_search=100)
python scripts/build_hnsw_index.py

# 3) Rodar todos os benchmarks (grid search + comparação com busca exata)
python scripts/run_benchmarks.py
```

## Quando os arquivos reais chegarem

Troque os dados falsos pelos reais, sem mudar nada de código:

- `data/processed/movies.parquet` -> vem do Welder
- `artifacts/sentence_embeddings.npy` -> vem da Dyesi
- `artifacts/word2vec_embeddings.npy` -> vem do Pedro (opcional, "se sobrar tempo")
- `src/search/cosine_search.py` -> SUBSTITUA pelo arquivo final do Welder
  (a interface `exact_cosine_search` / `batch_exact_cosine_search` deve
  ser mantida igual, ou ajuste as chamadas em run_benchmarks.py)

Para gerar queries reais de teste (em vez de fake_queries.npy), vetorize
um conjunto de perguntas de exemplo com o mesmo modelo de sentence
embeddings usado pela Dyesi, e salve como .npy no mesmo formato.

## O que cada arquivo faz

- `src/search/hnsw_search.py` — classe HNSWSearch (build/query/save/load) + recall_at_k()
- `src/search/cosine_search.py` — STUB da busca exata (placeholder até o Welder entregar)
- `scripts/build_hnsw_index.py` — constrói e salva artifacts/hnsw_index.bin
- `scripts/run_benchmarks.py` — grid search de M/ef_construction/ef_search,
  compara com busca exata, gera results/benchmark_results.csv e os 3 gráficos
- `scripts/generate_fake_data.py` — gera dados sintéticos para testes (não é entregável oficial)

## Resultado já observado com dados de teste (3000 filmes, dim=384)

Interessante para os slides: com N=3000, a busca exata (vetorizada com
numpy) foi na verdade MAIS RÁPIDA que o HNSW em alguns casos
(~0.00016s vs ~0.0002-0.0004s). Isso é esperado e vale explicar na
apresentação: o ganho do HNSW só aparece de forma clara quando N cresce
bastante (dezenas/centenas de milhares de vetores), pois a busca exata
é O(N*d) mas com operações vetorizadas em numpy/BLAS muito otimizadas
para N pequeno. Recomendo rodar os benchmarks também com um N maior
(gere fake data com --n 50000 ou --n 100000) para mostrar o cruzamento
das curvas — isso é exatamente o tipo de análise que o item 6 dos
critérios de avaliação (complexidade) está pedindo.
