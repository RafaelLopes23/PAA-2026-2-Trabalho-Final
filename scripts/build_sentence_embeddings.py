import os
import time
import json
import argparse
import sys

# Garante que o Python encontre a pasta 'src' na raiz do projeto
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from src.embeddings.sentence_embeddings import SentenceEmbeddingPipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/movies.parquet", 
                        help="Caminho do arquivo parquet do Welder")
    parser.add_argument("--output", default="artifacts/word2vec_embeddings.npy", # Mudar se o henrique usar outro nome, mas no padrão do grupo é sentence_embeddings.npy
                        help="Caminho de saída da matriz de embeddings")
    args = parser.parse_args()
    
   
    output_npy = "artifacts/sentence_embeddings.npy" if args.output == "artifacts/word2vec_embeddings.npy" else args.output

    print("--- MÓDULO 3: SENTENCE EMBEDDINGS (DYESI) ---")
    
    if not os.path.exists(args.input):
        print(f"❌ Erro: O arquivo {args.input} não foi encontrado. rodar o pré-processamento antes.")
        return

    # 1. Carregar os dados puros limpos
    print(f"Carregando dados de {args.input} ...")
    df = pd.read_parquet(args.input)
    synopses = df["synopsis"].tolist()
    print(f"  -> {len(synopses)} sinopses carregadas.")

    # 2. Inicializar o pipeline
    print("Inicializando o modelo 'all-MiniLM-L6-v2' (Licença Apache 2.0)...")
    pipeline = SentenceEmbeddingPipeline()

    # 3. Processar em lote para performance (Complexidade O(N * L^2 * d))
    print("Gerando embeddings em lotes (batch_size=128)...")
    start_time = time.perf_counter()
    
    # Usando o encode direto do modelo interno para processar a lista inteira de uma vez de forma otimizada
    embeddings = pipeline.model.encode(
        synopses, 
        batch_size=128, 
        show_progress_bar=True, 
        convert_to_numpy=True
    )
    
    elapsed = time.perf_counter() - start_time
    n, dim = embeddings.shape
    print(f"  -> Embeddings calculados em {elapsed:.4f} segundos.")
    print(f"  -> Dimensão da matriz resultante: {n} vetores x {dim} dimensões")

    # 4. Salvar os resultados na pasta de artefatos
    os.makedirs("artifacts", exist_ok=True)
    np.save(output_npy, embeddings.astype(np.float32))
    print(f" Matriz salva com sucesso em: {output_npy}")

    # Salvar arquivo de estatísticas para os slides de PAA
    stats = {
        "model": "all-MiniLM-L6-v2",
        "license": "Apache 2.0",
        "total_vectors": n,
        "dimension": dim,
        "generation_time_seconds": round(elapsed, 4),
        "memory_ram_mb": round(embeddings.nbytes / (1024 * 1024), 2)
    }
    stats_path = "artifacts/sentence_embeddings_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)
    print(f" Estatísticas salvas em {stats_path}.")

if __name__ == "__main__":
    main()