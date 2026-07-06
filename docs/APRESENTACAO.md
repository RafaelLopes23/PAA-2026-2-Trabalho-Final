# Apoio Para Apresentacao - Projeto Final PAA

Este arquivo organiza as respostas para os pontos pedidos no PDF da disciplina e serve como base para montar os slides. A ideia aqui e deixar o conteudo mais completo do que o necessario para o slide, para voce poder resumir depois sem perder informacao importante.

## 1. Integrantes do grupo

- Welder Oliveira
- Pedro
- Dyesi
- Henrique Franca
- Rafael Lopes

## 2. Linguagens, bibliotecas e licencas

### Linguagem principal

- `Python`

### Bibliotecas principais do projeto

- `FastAPI`:
  usada para expor a API HTTP do sistema.
  Licenca: `MIT`.
- `Uvicorn`:
  servidor ASGI usado para executar a aplicacao FastAPI.
  Licenca: `BSD-3-Clause`.
- `NumPy`:
  usado para armazenar vetores, calcular normas e operar com matrizes de embeddings.
  Licenca: familia `BSD/MIT` em distribuicao moderna; pode ser resumido no slide como `BSD-like`.
- `pandas`:
  usado para leitura, limpeza e manipulacao tabular da base de filmes.
  Licenca: `BSD-3-Clause`.
- `pyarrow`:
  usado para leitura e escrita de `parquet`.
  Licenca: `Apache-2.0`.
- `sentence-transformers`:
  usado para gerar sentence embeddings com o modelo `all-MiniLM-L6-v2`.
  Licenca da biblioteca: `Apache-2.0`.
- `PyTorch`:
  backend numerico usado pelo `sentence-transformers` e opcionalmente pela execucao local do TinyLlama via `transformers`.
  Licenca: `BSD-3-Clause`.
- `hnswlib`:
  usado para o indice de busca aproximada HNSW.
  Licenca: `Apache-2.0`.
- `matplotlib`:
  usado para graficos nos benchmarks.
  Licenca: propria do Matplotlib.

### Modelo de sentence embeddings

- Modelo escolhido: `all-MiniLM-L6-v2`
- Dimensao: `384`
- Licenca: `Apache 2.0`
- Motivo da escolha:
  e um modelo pequeno, muito conhecido, rapido para gerar embeddings e suficientemente forte para uma demonstracao em CPU ou GPU modesta.

### Modelo de LLM local

- Modelo usado: `tinyllama`
- Runtime usado: `Ollama`
- Papel no sistema:
  nao faz a busca semantica; apenas recebe os resultados recuperados e monta uma resposta final em linguagem natural.

## 3. Programas e componentes utilizados

O sistema nao usa banco de dados relacional tradicional. Em vez disso, ele trabalha com artefatos locais em arquivo.

### Componentes principais

- `Python + venv`:
  ambiente de execucao do projeto.
- `FastAPI`:
  camada da API.
- `Uvicorn`:
  servidor HTTP da API.
- `Ollama`:
  processo separado que serve o modelo `tinyllama`.
- `Parquet`:
  armazenamento da base processada de filmes.
- `NumPy .npy`:
  armazenamento das matrizes de embeddings.
- `Indice binario HNSW`:
  armazenamento persistente do indice aproximado.

### Por que essa arquitetura faz sentido

- simplifica a reproducao do projeto;
- evita a complexidade de instalar e manter um banco externo;
- permite carregar os dados uma unica vez ao iniciar o servidor;
- deixa a medicao de tempo mais transparente, porque cada etapa fica bem separada.

## 4. Como o sistema funciona

O fluxo completo do sistema pode ser explicado em cinco etapas.

### Etapa 1: entrada da pergunta

O usuario envia uma pergunta em linguagem natural para o endpoint `POST /query`, informando:

- `question`
- `method`
- `top_k`

Exemplo:

```json
{
  "question": "Former dream architect Dom Cobb infiltrates the subconscious of targets while dreaming.",
  "method": "hnsw",
  "top_k": 5
}
```

### Etapa 2: vetorizar a pergunta

Dependendo do metodo escolhido:

- `word2vec`:
  a pergunta e tokenizada e transformada na media dos vetores das palavras conhecidas.
- `sentence`:
  a frase inteira e enviada ao modelo `all-MiniLM-L6-v2`, que produz um embedding denso de 384 dimensoes.
- `hnsw`:
  usa o mesmo embedding de consulta do metodo `sentence`, porque o indice HNSW foi construido sobre os sentence embeddings.

### Etapa 3: recuperar sinopses relevantes

- `word2vec` e `sentence` usam busca exata por similaridade de cosseno.
- `hnsw` usa busca aproximada com um indice HNSW ja construido.

O resultado dessa fase e uma lista `top-k` de filmes mais proximos, cada um com:

- titulo
- identificador
- score de similaridade
- trecho da sinopse

### Etapa 4: gerar a resposta final

As sinopses recuperadas sao enviadas ao modulo do LLM local.

O `TinyLlama` recebe:

- a pergunta original
- o metodo de recuperacao
- os filmes mais proximos
- um prompt pedindo uma resposta curta e baseada apenas no contexto recuperado

### Etapa 5: fallback seguro

Modelos pequenos podem responder mal, repetir o prompt ou inventar formato. Por isso, o projeto implementa um fallback:

- se o LLM responder corretamente, a resposta dele e usada;
- se o LLM repetir o contexto, devolver placeholders ou sair do formato esperado, o sistema retorna uma resposta estruturada baseada no primeiro resultado recuperado.

Esse ponto e importante para a apresentacao porque mostra preocupacao com robustez e nao apenas com funcionamento ideal.

## 5. Dados de treinamento e base utilizada

### Base escolhida

- `CMU Movie Summary Corpus`

### O que existe na base

A base fornece informacoes como:

- identificador do filme
- titulo
- metadados
- sinopse textual

### Como a base e obtida

O projeto usa:

- `src/data/download_dataset.py` para baixar os arquivos
- `src/data/preprocess.py` para combinar e limpar os dados

### Como a base final ficou

Na integracao atual:

- quantidade de filmes processados: `42201`
- arquivo canonico final: `data/processed/movies.parquet`
- tambem pode ser gerado: `data/processed/movies.csv`

### Observacao importante

As sinopses da base estao majoritariamente em ingles. Isso afeta diretamente a qualidade da busca semantica quando a pergunta do usuario esta em portugues muito livre. Na pratica, consultas em ingles ou mais proximas do texto da sinopse tendem a funcionar melhor.

## 6. Pre-processamento e analise de complexidade

### O que foi feito no pre-processamento

O modulo do Welder ficou responsavel por:

- ler os arquivos originais do corpus;
- combinar metadados com sinopses;
- manter os campos mais importantes, como `movie_id`, `title` e `synopsis`;
- remover registros invalidos;
- remover duplicados;
- descartar filmes sem sinopse;
- aplicar limpeza textual;
- gerar tokenizacao basica para reutilizacao posterior;
- salvar a base processada em formato final.

### Por que esse passo e importante

Sem esse tratamento:

- haveria registros inconsistentes;
- o Word2Vec treinaria em textos sujos;
- os embeddings poderiam ficar desalinhados com a base;
- a busca retornaria resultados piores.

### Complexidade do pre-processamento

Se `N` for o numero de filmes e `L` o tamanho medio do texto:

- leitura dos registros: aproximadamente `O(N)`
- limpeza e tokenizacao: aproximadamente `O(NL)`
- remocao de invalidos e duplicados: em geral linear ou quase linear com apoio de estruturas de hash

Na pratica, o custo dominante e percorrer todos os textos e processar as sinopses.

## 7. Treinamento e geracao das representacoes vetoriais

Aqui o projeto compara duas formas de representacao semantica.

### 7.1 Word2Vec Average

#### Como funciona

- treina-se um modelo Word2Vec sobre os tokens das sinopses;
- cada palavra recebe um vetor;
- a sinopse inteira passa a ser representada pela media dos vetores das palavras conhecidas.

#### Vantagens

- simples de implementar;
- rapido para consultar;
- barato em memoria se comparado a modelos maiores;
- bom como baseline classico.

#### Limitacoes

- perde ordem e estrutura da frase;
- a media de palavras nao entende bem contexto completo;
- sofre mais com ambiguidades e consultas descritivas complexas.

#### Medicoes desta integracao

- filmes: `42201`
- treino: `280.166 s`
- geracao dos vetores: `2.3753 s`
- dimensao dos vetores: `100`

#### Complexidade

Se `T` for o numero total de tokens:

- treinamento do Word2Vec depende de hiperparametros como janela, numero de epocas e negativas;
- como aproximacao didatica, pode-se dizer que o custo cresce com o numero de tokens processados e o numero de epocas;
- geracao do vetor medio por documento e aproximadamente linear no tamanho do documento.

### 7.2 Sentence Embeddings

#### Como funciona

- usa-se um modelo pre-treinado de `sentence-transformers`;
- cada sinopse inteira e codificada diretamente como um unico vetor denso;
- a frase completa passa a ser representada de forma mais contextual.

#### Vantagens

- melhor captura de significado global da frase;
- normalmente produz resultados semanticos superiores ao Word2Vec Average;
- mais adequado para consultas descritivas.

#### Limitacoes

- custo de geracao maior;
- depende de modelo externo mais pesado;
- demanda mais memoria e mais tempo para gerar todos os embeddings.

#### Medicoes desta integracao

- modelo: `all-MiniLM-L6-v2`
- vetores gerados: `42201`
- dimensao: `384`
- tempo de geracao: `993.0693 s`
- memoria aproximada da matriz: `61.82 MB`

#### Complexidade

Se `N` for o numero de sinopses e `L` o tamanho medio de cada uma:

- o custo total de geracao cresce aproximadamente com `N` multiplicado pelo custo de inferencia do encoder por texto;
- esse custo e bem maior do que a media de Word2Vec, mas em troca a qualidade tende a ser melhor.

## 8. Como o servidor forma a resposta e complexidade

### Estrutura do servidor

O Rafael integrou os modulos em:

- `src/pipeline.py`
- `src/api/main.py`
- `src/api/schemas.py`
- `src/llm/tinyllama.py`

### O que acontece em uma consulta

1. o servidor recebe a pergunta;
2. seleciona o metodo pedido;
3. gera o embedding da pergunta;
4. consulta a base vetorial ou o indice HNSW;
5. coleta os `top-k` resultados;
6. monta o prompt do LLM;
7. gera a resposta final;
8. retorna resultados, tempos e avisos.

### Como a inicializacao foi pensada

Modelos, embeddings e indices sao carregados uma unica vez no startup da API. Isso evita:

- recarregar arquivos pesados a cada request;
- repetir custo de inicializacao;
- distorcer a medicao do tempo de consulta.

### Complexidade por consulta

Se `d` for a dimensao dos vetores e `N` o numero de filmes:

- busca exata por cosseno:
  aproximadamente `O(Nd)` por consulta.
- busca HNSW:
  custo sublinear na pratica, frequentemente descrito como proximo de `O(log N)` para busca, embora dependa da configuracao do indice e da distribuicao dos dados.

O custo total do endpoint tambem inclui:

- geracao do embedding da pergunta;
- tempo do LLM local.

## 9. Recursos computacionais disponiveis e o que e possivel fazer

### Maquina usada para a validacao local

- CPU: `Ryzen 7 5700X`
- GPU: `RX 6600 8 GB`

### O que essa maquina permite

- preprocessar a base localmente;
- treinar o Word2Vec;
- gerar sentence embeddings para o corpus;
- construir o indice HNSW;
- rodar a API localmente;
- executar um LLM pequeno como TinyLlama via Ollama.

### O que essa maquina nao e ideal para fazer

- treinar modelos grandes do zero;
- usar LLMs muito maiores com baixa latencia;
- repetir experimentos muito pesados varias vezes sob prazo curto.

### Conclusao pratica

Para o escopo da disciplina, os recursos foram suficientes para uma demonstracao funcional e para comparar estrategias de busca semantica, mesmo que nao sejam recursos de laboratorio de treinamento pesado.

## 10. Tempo de execucao de cada fase

### Medicoes obtidas nesta integracao

- treino do Word2Vec: `280.166 s`
- geracao da matriz Word2Vec: `2.3753 s`
- geracao dos sentence embeddings: `993.0693 s`
- construcao do indice HNSW: `2.7475 s`

### Onde essas metricas estao documentadas

- `data/processed/preprocess_stats.json`:
  contem as estatisticas da base bruta e processada, incluindo tamanho dos arquivos, numero de filmes, media das sinopses e tempo de pre-processamento.
- `artifacts/word2vec_metrics.json`:
  contem as metricas do Word2Vec Average, incluindo tempo de treino, tempo de vetorizacao, dimensao, tamanho dos artefatos e tempo medio de consulta.
- `artifacts/sentence_embeddings_stats.json`:
  contem as metricas dos sentence embeddings, incluindo modelo, licenca, dimensao, numero de vetores, tempo de geracao e memoria aproximada.
- `artifacts/hnsw_index.bin.meta.json`:
  contem as metricas e parametros do indice HNSW, incluindo tempo de construcao, dimensao, numero de elementos, `M`, `ef_construction` e `ef_search`.

### Tempo de consulta

Nos testes desta integracao:

- busca exata com sentence embeddings ficou na faixa de dezenas de milissegundos para a etapa de busca;
- HNSW ficou na faixa de menos de `1 ms` para a etapa de recuperacao, sem contar o tempo de gerar o embedding da pergunta;
- o LLM local foi a parte mais lenta da resposta final quando ativado.

### Interpretacao

O projeto tem duas fases bem diferentes:

- fases offline:
  preprocessamento, treinamento e geracao dos artefatos;
- fase online:
  consulta do usuario.

Essa separacao e boa para a disciplina porque permite gastar mais tempo offline para reduzir a latencia em tempo de uso.

## 11. Como o tempo de treinamento afeta a qualidade

### No Word2Vec

Mais tempo de treinamento geralmente significa:

- mais iteracoes sobre o corpus;
- vetores de palavras potencialmente melhores;
- maior custo computacional.

Se o treinamento for curto demais:

- os vetores podem ficar pouco informativos;
- a media das palavras perde qualidade;
- a busca semantica tende a piorar.

### Nos sentence embeddings

No nosso caso, nao houve treino do zero. O modelo ja veio pre-treinado. Portanto, o custo maior nao esta em treinar o modelo, mas em gerar embeddings para todas as sinopses.

Mesmo sem treinamento local demorado, a qualidade tende a ser melhor do que Word2Vec Average porque o modelo ja incorpora informacao semantica de treinamento anterior.

### Mensagem principal para o slide

Tempo de treinamento maior pode melhorar qualidade, mas com retorno decrescente. Em muitos casos, um modelo pre-treinado pequeno oferece melhor relacao entre custo e qualidade do que treinar algo do zero no contexto da disciplina.

## 12. Como o tempo de inferencia afeta a qualidade

### Na recuperacao

- busca exata:
  avalia todos os vetores, tende a ser mais fiel ao ranking real, mas custa mais tempo.
- HNSW:
  e bem mais rapido, mas pode perder um pouco de recall em troca da velocidade.

### No LLM

- modelos pequenos respondem mais rapido;
- em compensacao, podem gerar respostas menos estaveis ou menos precisas;
- por isso foi necessario colocar validacao de formato e fallback.

### Conclusao

Existe um compromisso entre:

- velocidade;
- qualidade semantica da recuperacao;
- qualidade textual da resposta final.

No projeto, esse compromisso apareceu claramente:

- `sentence` e `hnsw` recuperam melhor do que `word2vec` em consultas descritivas;
- `hnsw` reduz muito a latencia de busca;
- `tinyllama` e suficiente para formatar, mas nao para substituir uma camada forte de recuperacao.

## 13. Demais aspectos relevantes do projeto

### Integracao entre partes do grupo

Foi necessario unificar formatos e contratos entre os modulos:

- base processada unica em `movies.parquet`;
- validacao de alinhamento entre numero de filmes e numero de embeddings;
- adaptacao do Word2Vec para usar a base final;
- reaproveitamento da busca exata do Welder;
- uso do mesmo encoder de consulta para `sentence` e `hnsw`;
- centralizacao do fluxo no `pipeline`.

### Robustez do sistema

O projeto nao apenas executa o caso ideal. Ele tambem:

- trata metodos indisponiveis;
- trata falta de artefatos;
- trata resposta ruim do LLM;
- expone tempos e avisos na resposta.

### Limitacoes observadas

- a base esta em ingles;
- consultas em portugues muito livres podem recuperar pior;
- Word2Vec Average teve desempenho semantico inferior em varias consultas descritivas;
- TinyLlama, por ser pequeno, nem sempre segue o formato desejado.

### Conclusao geral

O sistema final atende ao objetivo da disciplina:

- recebe pergunta em linguagem natural;
- usa busca semantica em uma base real de filmes;
- compara varios metodos de representacao e busca;
- gera uma resposta final com LLM local;
- permite discutir desempenho, complexidade, qualidade e trade-offs.

## Sugestoes para a demonstracao

Esta parte nao precisa ir inteira para os slides, mas ajuda a planejar a apresentacao pratica.

### Ordem sugerida

1. mostrar `GET /health`;
2. mostrar `GET /methods`;
3. fazer uma consulta com `word2vec`;
4. repetir com `sentence`;
5. repetir com `hnsw`;
6. destacar tempos de busca e diferencas de resultado;
7. mostrar a resposta final formatada.

### Consulta que funcionou bem

```text
Former dream architect Dom Cobb infiltrates the subconscious of targets while dreaming.
```

Com essa formulação, `sentence` e `hnsw` retornaram `Inception` no topo durante a validacao.

### Observacao importante para a demo

Como a base esta em ingles e o encoder tambem funciona melhor em ingles, a demonstracao fica mais consistente com consultas em ingles ou muito proximas da sinopse.
