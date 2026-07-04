"""
build_word2vec_embeddings.py

Script entregável para treinar o Word2Vec, gerar a matriz de embeddings,
salvar o resultado, medir tempos, tamanhos e validar a qualidade dos resultados.

Uso:
    python scripts/build_word2vec_embeddings.py \
        --vocab_size 5000 \
        --embed_dim 100 \
        --epochs 1 \
        --window_size 3
"""

import argparse
import os
import sys
import time
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.embeddings.word2vec_average import clean_text, Word2VecTrainer, Word2VecAverageEmbedder
from src.search.cosine_search import exact_cosine_search

def get_file_size_mb(filepath: str) -> float:
    """Retorna o tamanho do arquivo em Megabytes."""
    if os.path.exists(filepath):
        return os.path.getsize(filepath) / (1024 * 1024)
    return 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=5000,
                        help="Tamanho do vocabulário")
    parser.add_argument("--embed_dim", type=int, default=100,
                        help="Dimensão dos embeddings de palavra")
    parser.add_argument("--window_size", type=int, default=3,
                        help="Tamanho da janela de contexto")
    parser.add_argument("--n_negatives", type=int, default=5,
                        help="Número de amostras negativas por par")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Número de épocas de treinamento")
    parser.add_argument("--init_lr", type=float, default=0.025,
                        help="Taxa de aprendizado inicial")
    parser.add_argument("--max_docs", type=int, default=None,
                        help="Limita o número de documentos no treino para testes rápidos")
    parser.add_argument("--output", default="artifacts/word2vec_embeddings.npy",
                        help="Caminho de saída para os embeddings")
    args = parser.parse_args()

    parquet_path = "data/processed/movies.parquet"
    if not os.path.exists(parquet_path):
        print(f"Erro: {parquet_path} não encontrado. Execute scripts/process_raw_data.py primeiro.")
        return

    print(f"Carregando {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    n_movies = len(df)
    print(f"  -> {n_movies} filmes carregados.")

    # 1. Limpeza e tokenização para treinamento do Word2Vec
    print("Tokenizando sinopses para treinamento...")
    start_tokenize = time.perf_counter()
    corpus = [clean_text(syn) for syn in df["synopsis"]]
    tokenize_time = time.perf_counter() - start_tokenize
    print(f"  -> Concluído em {tokenize_time:.2f}s")

    # 2. Treinamento do Word2Vec
    print("\n=== Passo 1: Treinando Word2Vec ===")
    trainer = Word2VecTrainer(
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        window_size=args.window_size,
        n_negatives=args.n_negatives,
        init_lr=args.init_lr,
        epochs=args.epochs,
        seed=42
    )
    
    trainer.build_vocab(corpus)
    
    # Medindo tempo de treinamento
    train_time = trainer.train(corpus, max_docs=args.max_docs)
    print(f"Tempo de treinamento: {train_time:.4f} segundos")

    # 3. Criação do Embedder e geração de vetores médios
    print("\n=== Passo 2: Gerando Vetores Médios das Sinopses ===")
    embedder = Word2VecAverageEmbedder(trainer.W_in, trainer.word_to_idx)
    
    # Medindo tempo para gerar os vetores
    embeddings, gen_time = embedder.embed_batch(df["synopsis"].tolist())
    print(f"Tempo para gerar os vetores: {gen_time:.4f} segundos")
    print(f"Matriz de embeddings gerada com formato: {embeddings.shape}")

    # 4. Salvando a matriz de vetores
    print("\n=== Passo 3: Salvando Arquivos ===")
    # Salva a matriz de embeddings médios de todos os filmes (N x d)
    np.save(args.output, embeddings)
    print(f"Matriz de embeddings dos filmes salva em {args.output}")

    # Salva o modelo Word2Vec compactado (pesos W_in e vocabulário)
    model_path = args.output.replace(".npy", ".model.npz")
    embedder.save_model(model_path)
    
    # Medindo tamanho dos arquivos gerados
    emb_size_mb = get_file_size_mb(args.output)
    model_size_mb = get_file_size_mb(model_path)
    
    print(f"Tamanho do arquivo de embeddings dos filmes ({args.output}): {emb_size_mb:.4f} MB")
    print(f"Tamanho do arquivo do modelo Word2Vec ({model_path}): {model_size_mb:.4f} MB")
    print(f"Tamanho total ocupado em disco: {emb_size_mb + model_size_mb:.4f} MB")

    # 5. Vetorização de queries e busca exata (Cosine Similarity)
    print("\n=== Passo 4: Rodando Consultas de Teste (Qualidade e Tempo) ===")
    
    test_queries = [
        "space travel spaceship sci-fi alien battle",
        "romantic comedy drama love wedding relationship",
        "detective police murder crime investigation killer",
        "martial arts action fight kung fu revenge",
        "scary ghost horror haunted house nightmare"
    ]
    
    query_times = []
    
    for i, q in enumerate(test_queries):
        # Medindo tempo para vetorizar a pergunta do usuário
        start_q_vec = time.perf_counter()
        q_vector = embedder.embed_text(q)
        q_vec_time = time.perf_counter() - start_q_vec
        
        # Medindo tempo da busca
        start_search = time.perf_counter()
        indices, scores = exact_cosine_search(q_vector, embeddings, top_k=5)
        search_time = time.perf_counter() - start_search
        
        total_query_time = q_vec_time + search_time
        query_times.append(total_query_time)
        
        print(f"\nConsulta {i+1}: '{q}'")
        print(f"  [Tempo de vetorização + busca: {total_query_time*1000:.4f} ms]")
        print("  Top 5 Resultados:")
        for rank, (idx, score) in enumerate(zip(indices, scores)):
            row = df.iloc[idx]
            # Mostra apenas os primeiros 100 caracteres da sinopse para verificação
            short_syn = row['synopsis'][:100].replace('\n', ' ') + "..."
            print(f"    {rank+1}. ID: {row['movie_id']} | Título: {row['title']} | Score: {score:.4f}")
            print(f"       Sinopse: {short_syn}")

    avg_query_time = np.mean(query_times)
    print(f"\nTempo médio de consulta (vetorização + busca cosseno exata): {avg_query_time*1000:.4f} ms")

    # Resumo para os slides / relatório
    print("\n" + "="*40)
    print("RESUMO DE MÉTRICAS (Word2Vec Average)")
    print("="*40)
    print(f"Tempo de Treinamento:       {train_time:.4f} s")
    print(f"Tempo de Geração de Vetores:{gen_time:.4f} s")
    print(f"Tamanho Matriz (.npy):      {emb_size_mb:.4f} MB")
    print(f"Tamanho Modelo (.model.npz):{model_size_mb:.4f} MB")
    print(f"Tempo Médio de Consulta:     {avg_query_time*1000:.4f} ms")
    print("="*40)

if __name__ == "__main__":
    main()
