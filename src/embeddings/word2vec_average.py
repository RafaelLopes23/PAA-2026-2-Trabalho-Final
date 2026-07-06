"""
Word2Vec Average treinado em NumPy, sem dependencias pesadas.

Este modulo reaproveita a ideia da branch do Pedro, mas foi adaptado para a
estrutura atual do projeto: os textos vindos do Welder ja chegam limpos em
`movies.parquet`, com a coluna `synopsis_tokens` pronta para treino.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass

import numpy as np

STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "him",
    "his",
    "how",
    "i",
    "in",
    "into",
    "is",
    "it",
    "its",
    "no",
    "not",
    "of",
    "on",
    "or",
    "she",
    "so",
    "that",
    "the",
    "their",
    "them",
    "these",
    "they",
    "this",
    "those",
    "to",
    "us",
    "was",
    "we",
    "were",
    "what",
    "which",
    "who",
    "why",
    "with",
    "you",
}

TOKEN_RE = re.compile(r"[a-z0-9']+")
TOKEN_LINE_RE = re.compile(r"[a-z0-9' ]+")


def clean_text(text: str) -> list[str]:
    """Limpa e tokeniza um texto livre."""
    if not isinstance(text, str):
        return []
    return TOKEN_RE.findall(text.lower())


def normalize_tokens(text_or_tokens: str) -> list[str]:
    """
    Converte uma string de tokens do Welder em lista.

    Se o texto ainda nao estiver tokenizado, cai no `clean_text`.
    """
    if not isinstance(text_or_tokens, str):
        return []
    stripped = text_or_tokens.strip()
    if " " in stripped and TOKEN_LINE_RE.fullmatch(stripped):
        tokens = text_or_tokens.split()
    else:
        tokens = clean_text(text_or_tokens)
    return [token for token in tokens if token and token not in STOPWORDS]


@dataclass
class Word2VecMetrics:
    train_seconds: float
    vectorize_seconds: float
    vocab_size: int
    embedding_dim: int


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
        seed: int = 42,
    ) -> None:
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.n_negatives = n_negatives
        self.init_lr = init_lr
        self.min_lr = min_lr
        self.epochs = epochs
        self.subsample_t = subsample_t
        self.seed = seed

        self.word_to_idx: dict[str, int] = {}
        self.idx_to_word: list[str] = []
        self.keep_probs: np.ndarray | None = None
        self.neg_probs: np.ndarray | None = None
        self.W_in: np.ndarray | None = None
        self.W_out: np.ndarray | None = None

    def build_vocab(self, corpus: list[list[str]]) -> None:
        """Constroi o vocabulario das palavras mais frequentes."""
        word_counts: dict[str, int] = {}
        total_words = 0
        for doc in corpus:
            for word in doc:
                word_counts[word] = word_counts.get(word, 0) + 1
                total_words += 1

        if total_words == 0:
            raise ValueError("Corpus vazio apos tokenizacao; nao foi possivel treinar o Word2Vec.")

        sorted_words = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)
        vocab_words = sorted_words[: self.vocab_size]

        self.word_to_idx = {word: idx for idx, (word, _) in enumerate(vocab_words)}
        self.idx_to_word = [word for word, _ in vocab_words]

        vocab_counts = np.array([word_counts[word] for word in self.idx_to_word], dtype=np.float32)
        freqs = vocab_counts / float(total_words)
        self.keep_probs = np.minimum(
            1.0,
            np.sqrt(self.subsample_t / freqs) + (self.subsample_t / freqs),
        )

        pow_counts = vocab_counts ** 0.75
        self.neg_probs = pow_counts / np.sum(pow_counts)

    def _subsample_sentence(self, sentence: list[str], rng: np.random.Generator) -> list[int]:
        if self.keep_probs is None:
            raise RuntimeError("build_vocab() deve ser chamado antes do treino.")
        idx_sentence: list[int] = []
        for word in sentence:
            idx = self.word_to_idx.get(word)
            if idx is None:
                continue
            if rng.random() <= self.keep_probs[idx]:
                idx_sentence.append(idx)
        return idx_sentence

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -15, 15)
        return 1.0 / (1.0 + np.exp(-x))

    def train(self, corpus: list[list[str]], max_docs: int | None = None) -> float:
        """
        Treina embeddings Skip-gram com negative sampling.

        A implementacao continua simples e totalmente em NumPy para manter o
        projeto leve e reproduzivel na maquina do grupo.
        """
        if not self.word_to_idx or self.neg_probs is None:
            raise RuntimeError("build_vocab() deve ser executado antes do treino.")

        start_time = time.perf_counter()
        rng = np.random.default_rng(self.seed)

        if max_docs is not None:
            corpus = corpus[:max_docs]

        processed_corpus: list[list[int]] = []
        for doc in corpus:
            filtered = self._subsample_sentence(doc, rng)
            if len(filtered) > 1:
                processed_corpus.append(filtered)

        if not processed_corpus:
            raise ValueError("Corpus vazio apos subsampling; ajuste os parametros de treino.")

        vocab_len = len(self.word_to_idx)
        self.W_in = rng.uniform(
            -0.5 / self.embed_dim,
            0.5 / self.embed_dim,
            (vocab_len, self.embed_dim),
        ).astype(np.float32)
        self.W_out = rng.uniform(
            -0.5 / self.embed_dim,
            0.5 / self.embed_dim,
            (vocab_len, self.embed_dim),
        ).astype(np.float32)

        neg_pool_size = 5_000_000
        neg_pool = rng.choice(vocab_len, size=neg_pool_size, p=self.neg_probs).astype(np.int32)
        neg_idx = 0
        total_steps = max(1, self.epochs * len(processed_corpus))
        step_count = 0

        for _epoch in range(self.epochs):
            rng.shuffle(processed_corpus)
            for doc in processed_corpus:
                step_count += 1
                progress = step_count / total_steps
                lr = self.init_lr * (1.0 - progress) + self.min_lr * progress

                for i, target in enumerate(doc):
                    if self.W_in is None or self.W_out is None:
                        raise RuntimeError("Matrizes de pesos ausentes durante o treino.")

                    vector_target = self.W_in[target].copy()
                    window = int(rng.integers(1, self.window_size + 1))
                    start = max(0, i - window)
                    end = min(len(doc), i + window + 1)
                    contexts = [doc[j] for j in range(start, end) if j != i]
                    if not contexts:
                        continue

                    n_contexts = len(contexts)
                    n_neg = n_contexts * self.n_negatives
                    if neg_idx + n_neg >= neg_pool_size:
                        neg_pool = rng.choice(vocab_len, size=neg_pool_size, p=self.neg_probs).astype(np.int32)
                        neg_idx = 0

                    negatives = neg_pool[neg_idx : neg_idx + n_neg].tolist()
                    neg_idx += n_neg

                    candidates = contexts + negatives
                    labels = np.zeros(len(candidates), dtype=np.float32)
                    labels[:n_contexts] = 1.0

                    candidate_vectors = self.W_out[candidates]
                    scores = candidate_vectors @ vector_target
                    probs = self._sigmoid(scores)
                    errors = labels - probs

                    grad_in = errors @ candidate_vectors
                    np.add.at(
                        self.W_out,
                        candidates,
                        (lr * errors[:, None] * vector_target).astype(np.float32),
                    )
                    self.W_in[target] += (lr * grad_in).astype(np.float32)

        return time.perf_counter() - start_time


class Word2VecAverageEmbedder:
    def __init__(self, W_in: np.ndarray, word_to_idx: dict[str, int]) -> None:
        self.W_in = W_in.astype(np.float32)
        self.word_to_idx = word_to_idx
        self.dim = int(W_in.shape[1]) if W_in is not None else 0

    def embed_text(self, text: str) -> np.ndarray:
        tokens = normalize_tokens(text)
        indices = [self.word_to_idx[token] for token in tokens if token in self.word_to_idx]
        if not indices:
            return np.zeros(self.dim, dtype=np.float32)
        return np.mean(self.W_in[indices], axis=0).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_text(text)

    def embed_batch(self, texts: list[str]) -> tuple[np.ndarray, float]:
        start = time.perf_counter()
        embeddings = np.zeros((len(texts), self.dim), dtype=np.float32)
        for idx, text in enumerate(texts):
            embeddings[idx] = self.embed_text(text)
        return embeddings, time.perf_counter() - start

    def save_model(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        vocab_words = [word for word, _idx in sorted(self.word_to_idx.items(), key=lambda item: item[1])]
        np.savez_compressed(
            filepath,
            W_in=self.W_in,
            vocab=np.array(vocab_words, dtype=object),
        )

    @classmethod
    def load_model(cls, filepath: str) -> "Word2VecAverageEmbedder":
        data = np.load(filepath, allow_pickle=True)
        W_in = data["W_in"]
        vocab_words = data["vocab"]
        word_to_idx = {word: idx for idx, word in enumerate(vocab_words)}
        return cls(W_in=W_in, word_to_idx=word_to_idx)
