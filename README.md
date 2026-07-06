# PAA 2026.2 - Sistema Q&A sobre Filmes

Trabalho final da disciplina de Projeto e Analise de Algoritmos. O sistema recebe uma pergunta em linguagem natural, recupera sinopses relevantes do **CMU Movie Summary Corpus** e devolve uma resposta formatada com um **LLM local**.

## Visao geral

Fluxo final da aplicacao:

```text
Pergunta do usuario
        |
        v
Embedding da pergunta
  |- Word2Vec Average
  |- Sentence Embeddings
        |
        v
Recuperacao
  |- Busca exata por cosseno
  |- HNSW aproximado
        |
        v
Top-k sinopses
        |
        v
Formatacao final com LLM local
  |- Ollama + tinyllama
  |- ou backend local via transformers
```

## O que foi integrado

Esta branch junta as partes do grupo na pratica:

- **Welder**: `download_dataset.py`, `preprocess.py`, `cosine_search.py` e contrato de dados em `movies.parquet`.
- **Pedro**: Word2Vec Average adaptado para treinar diretamente sobre a base processada do Welder.
- **Dyesi**: sentence embeddings com `all-MiniLM-L6-v2` e geracao da matriz `sentence_embeddings.npy`.
- **Henrique**: HNSW, construcao do indice e scripts de benchmark.
- **Rafael**: pipeline unificado, API FastAPI, carregamento unico dos recursos e integracao com LLM local.

## Adaptacoes feitas para integrar tudo

### Sobre a parte do Welder

- Mantive `movies.parquet` como base canonica do projeto.
- Reaproveitei `exact_cosine_search()` como backend comum para `word2vec` e `sentence`.
- O pipeline valida se o numero de linhas dos embeddings bate com o numero de filmes da base.

### Sobre a parte do Pedro

- O Word2Vec foi adaptado para consumir `data/processed/movies.parquet` em vez de um processamento bruto paralelo.
- O treino usa `synopsis_tokens` do Welder quando disponivel.
- Alem da matriz `word2vec_embeddings.npy`, o projeto salva `word2vec_embeddings.model.npz` para embedar queries depois.
- O script final de geracao ficou em `scripts/build_word2vec_embeddings.py`.

### Sobre a parte da Dyesi

- O modulo de sentence embeddings foi integrado em `src/embeddings/sentence_embeddings.py`.
- O script final ficou em `scripts/build_sentence_embeddings.py`.
- Corrigi o fluxo para o output padrao ser `artifacts/sentence_embeddings.npy`.
- O pipeline passou a reconhecer automaticamente `SentenceEmbeddingPipeline` como encoder de consulta.

### Sobre a parte do Henrique

- O indice HNSW continua vindo de `src/search/hnsw_search.py`.
- O build final do indice usa os sentence embeddings reais da Dyesi.
- O metodo `hnsw` na API usa o mesmo encoder de consulta do metodo `sentence`.

### Sobre parte do Rafael

- A camada comum em `src/pipeline.py`.
- A API em `src/api/main.py` e `src/api/schemas.py`.
- A integracao de LLM em `src/llm/tinyllama.py`.
- A resposta final tenta usar um LLM local real; se a saida vier ruim ou fora do formato esperado, cai para uma resposta segura baseada nas sinopses recuperadas.

## Estrutura principal

```text
src/
├── api/
│   ├── main.py
│   └── schemas.py
├── data/
│   ├── download_dataset.py
│   └── preprocess.py
├── embeddings/
│   ├── sentence_embeddings.py
│   └── word2vec_average.py
├── llm/
│   └── tinyllama.py
├── pipeline.py
└── search/
    ├── cosine_search.py
    └── hnsw_search.py
```

## Ambiente recomendado

O projeto foi testado nesta integracao com:

- Python em `venv`
- `sentence-transformers`
- `torch` CPU-only
- `FastAPI`
- `hnswlib`

Para a maquina (`Ryzen 7 5700X` + `RX 6600 8 GB`), a melhor opcao de execucao local ficou:

1. usar `sentence-transformers` para os embeddings;
2. usar `TinyLlama` via `Ollama`;
3. manter os caches dentro do proprio projeto para poder rodar de novo sem depender de internet.

## Passo a passo completo

### 1. Criar ambiente virtual

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Instalar PyTorch CPU-only

Isso evita puxar o pacote CUDA gigante do PyPI.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 3. Instalar o resto das dependencias

```bash
pip install -r requirements.txt
```

### 4. Preparar o dataset

Se a base ainda nao estiver pronta:

```bash
python -m src.data.download_dataset --include-optional
python -m src.data.preprocess --also-csv
```

### 5. Gerar Word2Vec Average

Execucao completa:

```bash
python scripts/build_word2vec_embeddings.py
```

Arquivos gerados:

- `artifacts/word2vec_embeddings.npy`
- `artifacts/word2vec_embeddings.model.npz`
- `artifacts/word2vec_metrics.json`

Metricas obtidas nesta integracao:

- `42201` filmes
- treino em `280.166 s`
- geracao da matriz em `2.3753 s`
- matriz final `42201 x 100`

### 6. Gerar sentence embeddings

```bash
python scripts/build_sentence_embeddings.py
```

Arquivos gerados:

- `artifacts/sentence_embeddings.npy`
- `artifacts/sentence_embeddings_stats.json`

Metricas registradas nesta integracao:

- modelo `all-MiniLM-L6-v2`
- `42201` vetores
- dimensao `384`
- geracao em `993.0693 s`

### 7. Construir o indice HNSW

```bash
python scripts/build_hnsw_index.py --embeddings artifacts/sentence_embeddings.npy
```

Metadados desta integracao:

- `42201` elementos
- `dim = 384`
- `M = 16`
- `ef_construction = 200`
- `ef_search = 100`
- build em `2.7475 s`

### 8. Configurar o cache dos modelos

```bash
export HF_HOME=$(pwd)/.cache/huggingface
export PAA_HF_LOCAL_ONLY=1
```

### 9. Instalar o Ollama localmente no projeto

Sem `sudo`, usando os mesmos binarios oficiais que funcionaram nesta integracao:

```bash
bash scripts/install_local_ollama.sh
```

### 10. Subir o servidor local do Ollama

Em um terminal:

```bash
bash scripts/start_local_ollama.sh
```

### 11. Baixar o modelo TinyLlama

Em outro terminal:

```bash
bash scripts/pull_tinyllama.sh
```

Para confirmar:

```bash
HOME=$(pwd) OLLAMA_MODELS=$(pwd)/.ollama/models OLLAMA_HOST=127.0.0.1:11434 ./.local/ollama/bin/ollama list
```

### 12. Escolher o backend do LLM

#### Opcao A: Ollama

Foi a opcao validada de ponta a ponta nesta branch:

```bash
export PAA_LLM_BACKEND=ollama
export PAA_LLM_MODEL=tinyllama
export PAA_LLM_ENDPOINT=http://127.0.0.1:11434/api/generate
export PAA_LLM_ENABLED=1
```

#### Opcao B: transformers local

Tambem funciona, mas ficou mais pesada na pratica:

```bash
export PAA_LLM_BACKEND=transformers
export PAA_LLM_TRANSFORMERS_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0
export PAA_LLM_ENABLED=1
```

Observacoes:

- na primeira execucao, o projeto baixa os pesos do modelo;
- nessa configuracao, o download do TinyLlama ficou em torno de `2.2 GB`;
- depois do primeiro download, os arquivos ficam no cache.

### 13. Subir a API

```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

## Endpoints

- `GET /health`
- `GET /methods`
- `POST /query`

### Exemplo de health

```bash
curl http://127.0.0.1:8000/health
```

### Exemplo de methods

```bash
curl http://127.0.0.1:8000/methods
```

### Exemplo de query

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Former dream architect Dom Cobb infiltrates the subconscious of targets while dreaming.",
    "method": "hnsw",
    "top_k": 3
  }'
```

## Metodos disponiveis

- `word2vec`: busca exata por cosseno sobre `word2vec_embeddings.npy`
- `sentence`: busca exata por cosseno sobre `sentence_embeddings.npy`
- `hnsw`: busca aproximada HNSW sobre os sentence embeddings
- `cosine`: alias para a melhor busca exata disponivel no momento

## Como a resposta final funciona

O pipeline faz:

1. embed da pergunta;
2. recuperacao dos filmes mais proximos;
3. envio do contexto recuperado para o LLM local;
4. validacao do formato da resposta do LLM;
5. fallback seguro se o modelo pequeno responder mal.

Esse fallback foi mantido de proposito para a resposta final nao quebrar ou inventar informacoes caso o TinyLlama produza texto ruim.

## Resultado da validacao desta integracao

Nesta branch, eu validei:

- `word2vec`, `sentence`, `hnsw` e `cosine` aparecem em `/methods`;
- `GET /health` respondeu `200`;
- `GET /methods` respondeu `200`;
- `POST /query` respondeu `200`;
- o `TinyLlama` via `Ollama` carregou e executou;
- o modelo `tinyllama:latest` ficou disponivel localmente no `ollama list`;
- quando a saida do TinyLlama veio com placeholders ou formato ruim, o sistema caiu para a resposta segura automaticamente;
- com `HF_HOME` local e `PAA_HF_LOCAL_ONLY=1`, os sentence embeddings funcionaram sem depender de novas requisicoes externas.

## Benchmarks e complexidade

- preprocessamento: custo dominado por leitura e tokenizacao
- busca exata por cosseno: aproximadamente `O(Nd)` por consulta
- HNSW: construcao aproximada `O(N log N)` e busca aproximada `O(log N)` na pratica
- Word2Vec Average: custo de treino depende do corpus, janela e numero de negativas

Os scripts do Henrique continuam disponiveis:

```bash
python scripts/run_benchmarks.py --top_k 5
```

## Observacoes de empacotamento

O PDF da disciplina limita o envio direto a `100 MB`. Por isso:

- `data/raw/`, `artifacts/`, `results/` e `.cache/` nao devem entrar no pacote final sem planejamento;
- o ideal e enviar o codigo e os slides, e compartilhar os artefatos pesados por link quando necessario.

## Branch atual

Esta integracao foi montada na branch:

```bash
RafaelLopes23-integracao
``'
