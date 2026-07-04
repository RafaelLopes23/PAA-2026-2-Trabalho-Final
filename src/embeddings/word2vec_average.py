"""
word2vec_average.py

Implementação do Word2Vec (Skip-gram com Negative Sampling) treinado do zero em NumPy,
além de utilitários para média de vetores (Word2Vec Average) para representar as sinopses.

Desenvolvido do zero devido a incompatibilidades de compilação do gensim no Python 3.14+
e limite de cota de disco para bibliotecas pesadas como PyTorch.
"""

import os
import re
import time
import numpy as np

# Conjunto padrão de stopwords em inglês para filtrar ruído nas sinopses
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "at", "an", "is", "was", "were", "be", "been", "this", "that",
    "these", "those", "it", "its", "he", "she", "they", "him", "her", "them",
    "his", "their", "as", "from", "into", "about", "who", "which", "what",
    "why", "how", "you", "i", "we", "us", "are", "have", "has", "had", "do",
    "does", "did", "but", "so", "no", "not"
}

def clean_text(text: str) -> list[str]:
    """Limpa e tokeniza o texto, removendo pontuação e convertendo para minúsculas."""
    if not isinstance(text, str):
        return []
    # Remove pontuação e mantém apenas letras e números
    text = re.sub(r"[^\w\s]", " ", text.lower())
    # Separa por espaços em branco
    tokens = text.split()
    return tokens

class Word2VecTrainer:
    def __init__(
        self,
        vocab_size: int = 5000,
        embed_dim: int = 100,
        window_size: int = 3,
        n_negatives: int = 5,
        init_lr: float = 0.025,
        min_lr: float = 0.0001,
        epochs: int = 1,
        subsample_t: float = 1e-3,
        seed: int = 42
    ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.n_negatives = n_negatives
        self.init_lr = init_lr
        self.min_lr = min_lr
        self.epochs = epochs
        self.subsample_t = subsample_t
        self.seed = seed

        self.word_to_idx = {}
        self.idx_to_word = []
        self.W_in = None
        self.W_out = None

    def build_vocab(self, corpus: list[list[str]]):
        """Constrói o vocabulário baseado nas palavras mais frequentes, excluindo stopwords."""
        word_counts = {}
        total_words = 0
        for doc in corpus:
            for word in doc:
                if word not in STOPWORDS:
                    word_counts[word] = word_counts.get(word, 0) + 1
                    total_words += 1

        # Ordena por frequência decrescente
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Seleciona as top V palavras
        vocab_words = sorted_words[:self.vocab_size]
        
        self.word_to_idx = {word: idx for idx, (word, _) in enumerate(vocab_words)}
        self.idx_to_word = [word for word, _ in vocab_words]
        
        # Guarda frequências reais dos termos do vocabulário para amostragem negativa e subsampling
        vocab_counts = np.array([word_counts[w] for w in self.idx_to_word], dtype=np.float32)
        
        # 1. Tabela de subsampling
        # P(keep) = (sqrt(t / f) + t / f)
        # f = count / total_words
        freqs = vocab_counts / total_words
        self.keep_probs = np.minimum(
            1.0,
            (np.sqrt(self.subsample_t / freqs) + self.subsample_t / freqs)
        )

        # 2. Tabela de negative sampling (unigram^0.75)
        pow_counts = vocab_counts ** 0.75
        self.neg_probs = pow_counts / np.sum(pow_counts)

        print(f"Vocabulário construído com {len(self.word_to_idx)} palavras.")
        print(f"Total de palavras analisadas no corpus: {total_words}")

    def _subsample_sentence(self, sentence: list[str]) -> list[int]:
        """Aplica subsampling para descartar palavras muito frequentes e retorna os IDs correspondentes."""
        rng = np.random.default_rng(self.seed)
        idx_sentence = []
        for word in sentence:
            if word in self.word_to_idx:
                idx = self.word_to_idx[word]
                prob = self.keep_probs[idx]
                if rng.random() <= prob:
                    idx_sentence.append(idx)
        return idx_sentence

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        # Sigmoide com clip para evitar overflow/underflow numérico
        x = np.clip(x, -15, 15)
        return 1.0 / (1.0 + np.exp(-x))

    def train(self, corpus: list[list[str]], max_docs: int = None):
        """Treina os embeddings Skip-gram usando NumPy vetorizado por palavra-alvo."""
        start_time = time.perf_counter()
        
        print("Preparando o corpus...")
        # Filtra o corpus para a quantidade máxima solicitada (opcional para agilizar testes)
        if max_docs is not None:
            corpus = corpus[:max_docs]

        # Converte corpus para listas de índices filtrados por subsampling
        processed_corpus = []
        for doc in corpus:
            filtered = self._subsample_sentence(doc)
            if len(filtered) > 1:
                processed_corpus.append(filtered)
        
        total_docs = len(processed_corpus)
        print(f"Corpus pronto. {total_docs} documentos válidos pós-subsampling.")

        # Inicializa matrizes de pesos de entrada e saída com distribuição uniforme
        rng = np.random.default_rng(self.seed)
        self.W_in = rng.uniform(
            -0.5 / self.embed_dim,
            0.5 / self.embed_dim,
            (len(self.word_to_idx), self.embed_dim)
        ).astype(np.float32)
        
        # Inicializa W_out randomicamente para quebrar a simetria (evita colapso dos vetores)
        self.W_out = rng.uniform(
            -0.5 / self.embed_dim,
            0.5 / self.embed_dim,
            (len(self.word_to_idx), self.embed_dim)
        ).astype(np.float32)

        # Pre-gera pool de amostras negativas para agilizar treinamento (evita chamar random em loops internos)
        # Geramos 5 milhões de IDs baseados na distribuição de probabilidades
        print("Gerando pool de amostras negativas...")
        neg_pool_size = 5000000
        neg_pool = rng.choice(
            len(self.word_to_idx),
            size=neg_pool_size,
            p=self.neg_probs
        ).astype(np.int32)
        neg_idx = 0

        # Loop de treinamento
        print(f"Iniciando treinamento Word2Vec por {self.epochs} época(s)...")
        total_steps = self.epochs * total_docs
        step_count = 0
        
        for epoch in range(self.epochs):
            epoch_start = time.perf_counter()
            loss = 0.0
            
            # Embaralha os documentos a cada época
            rng.shuffle(processed_corpus)
            
            for doc in processed_corpus:
                step_count += 1
                
                # Decaimento linear da taxa de aprendizado
                progress = step_count / total_steps
                lr = self.init_lr * (1.0 - progress) + self.min_lr * progress
                
                n_words = len(doc)
                for i in range(n_words):
                    w_t = doc[i]  # Target word index
                    v_t = self.W_in[w_t].copy()  # Copia o vetor para evitar modificações colaterais durante o loop
                    
                    # Context window
                    window = rng.integers(1, self.window_size + 1)
                    start = max(0, i - window)
                    end = min(n_words, i + window + 1)
                    
                    # Índices de contexto
                    contexts = [doc[j] for j in range(start, end) if j != i]
                    if not contexts:
                        continue
                    
                    n_contexts = len(contexts)
                    
                    # Seleciona amostras negativas do pool pre-gerado
                    n_neg = n_contexts * self.n_negatives
                    # Reinicia o pool se necessário
                    if neg_idx + n_neg >= neg_pool_size:
                        neg_pool = rng.choice(
                            len(self.word_to_idx),
                            size=neg_pool_size,
                            p=self.neg_probs
                        ).astype(np.int32)
                        neg_idx = 0
                    
                    negatives = neg_pool[neg_idx : neg_idx + n_neg].tolist()
                    neg_idx += n_neg
                    
                    # Junta todos os candidatos em uma única operação vetorizada
                    # C + N
                    candidates = contexts + negatives
                    
                    # Rótulos: 1 para contextos reais, 0 para negativos
                    labels = np.zeros(len(candidates), dtype=np.float32)
                    labels[:n_contexts] = 1.0
                    
                    # Calcula scores e probabilidades via vetorização NumPy
                    # W_out[candidates] tem shape (len(candidates), d)
                    w_out_cand = self.W_out[candidates]
                    scores = np.dot(w_out_cand, v_t)
                    probs = self._sigmoid(scores)
                    
                    # Erro: Y - P
                    errors = labels - probs
                    
                    # Atualiza gradientes usando add.at para tratar índices repetidos corretamente
                    # grad_in é a soma ponderada dos vetores de saída
                    grad_in = np.dot(errors, w_out_cand)
                    
                    # W_out[cand] += lr * error * v_t
                    np.add.at(
                        self.W_out,
                        candidates,
                        (lr * errors[:, None] * v_t).astype(np.float32)
                    )
                    
                    # W_in[w_t] += lr * grad_in
                    self.W_in[w_t] += (lr * grad_in).astype(np.float32)
                    
            epoch_elapsed = time.perf_counter() - epoch_start
            print(f"  -> Época {epoch + 1}/{self.epochs} concluída em {epoch_elapsed:.2f}s")
            
        elapsed = time.perf_counter() - start_time
        print(f"Treinamento concluído com sucesso em {elapsed:.2f} segundos.")
        return elapsed

class Word2VecAverageEmbedder:
    def __init__(self, W_in: np.ndarray, word_to_idx: dict[str, int]):
        self.W_in = W_in.astype(np.float32)
        self.word_to_idx = word_to_idx
        self.dim = W_in.shape[1] if W_in is not None else 0

    def embed_text(self, text: str) -> np.ndarray:
        """
        Gera a representação vetorial de um texto calculando a média
        dos vetores de suas palavras presentes no vocabulário.
        """
        tokens = clean_text(text)
        # Filtra stopwords
        tokens = [t for t in tokens if t not in STOPWORDS]
        
        # Mapeia para índices no vocabulário
        idxs = [self.word_to_idx[t] for t in tokens if t in self.word_to_idx]
        
        if not idxs:
            # Caso não haja palavras conhecidas, retorna vetor de zeros
            return np.zeros(self.dim, dtype=np.float32)
            
        # Pega os vetores e tira a média
        vectors = self.W_in[idxs]
        return np.mean(vectors, axis=0)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Vetoriza uma lista de sinopses em lote."""
        start = time.perf_counter()
        embeddings = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            embeddings[i] = self.embed_text(text)
        elapsed = time.perf_counter() - start
        return embeddings, elapsed

    def save_model(self, filepath: str):
        """Salva a matriz de pesos W_in e o vocabulário em um arquivo compactado .npz."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # Ordena o vocabulário pelo índice para salvar
        vocab_words = [word for word, idx in sorted(self.word_to_idx.items(), key=lambda x: x[1])]
        np.savez_compressed(
            filepath,
            W_in=self.W_in,
            vocab=np.array(vocab_words, dtype=object)
        )
        print(f"Modelo Word2Vec salvo em {filepath}")

    @classmethod
    def load_model(cls, filepath: str) -> "Word2VecAverageEmbedder":
        """Carrega a matriz de pesos e o vocabulário de um arquivo .npz."""
        data = np.load(filepath, allow_pickle=True)
        W_in = data["W_in"]
        vocab_words = data["vocab"]
        word_to_idx = {word: idx for idx, word in enumerate(vocab_words)}
        return cls(W_in, word_to_idx)
