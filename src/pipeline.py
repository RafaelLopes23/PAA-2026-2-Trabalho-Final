"""
Camada de integracao do sistema de perguntas e respostas.

Esta e a ponte entre:
- embeddings das sinopses;
- metodos de busca (cosseno exato e HNSW);
- LLM local para formatacao da resposta.

O objetivo aqui e deixar o projeto utilizavel ja com Word2Vec + HNSW/Welder e
com a API pronta para receber outros backends de embeddings sem retrabalho.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.embeddings.word2vec_average import Word2VecAverageEmbedder
from src.llm.tinyllama import GenerationResult, TinyLlamaClient, TinyLlamaConfig
from src.search.cosine_search import exact_cosine_search

try:
    from src.search.hnsw_search import HNSWSearch
    HNSW_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001
    HNSWSearch = None  # type: ignore[assignment]
    HNSW_IMPORT_ERROR = exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PipelineError(RuntimeError):
    """Erro de alto nivel do pipeline."""


class MethodUnavailableError(PipelineError):
    """Metodo solicitado nao esta pronto para uso."""


@dataclass
class MethodInfo:
    name: str
    description: str
    available: bool
    reason: str = ""
    resolved_method: str | None = None


class QueryEmbedderAdapter:
    def __init__(self, obj: Any, name: str) -> None:
        self.obj = obj
        self.name = name

    def embed_query(self, text: str) -> np.ndarray:
        if hasattr(self.obj, "embed_query"):
            vector = self.obj.embed_query(text)
        elif hasattr(self.obj, "get_embedding"):
            vector = self.obj.get_embedding(text)
        elif hasattr(self.obj, "embed_text"):
            vector = self.obj.embed_text(text)
        elif hasattr(self.obj, "encode"):
            vector = self.obj.encode([text])
        else:
            raise PipelineError(f"Embedder '{self.name}' nao expoe embed_query(), embed_text() nem encode().")

        array = np.asarray(vector, dtype=np.float32)
        if array.ndim == 2:
            array = array[0]
        return array.reshape(-1).astype(np.float32)


class ExactRetriever:
    def __init__(
        self,
        movies_df: pd.DataFrame,
        embeddings: np.ndarray,
        embedder: QueryEmbedderAdapter,
        method_name: str,
        description: str,
    ) -> None:
        self.movies_df = movies_df
        self.embeddings = embeddings
        self.embedder = embedder
        self.method_name = method_name
        self.description = description

    def search(self, question: str, top_k: int) -> tuple[list[dict[str, object]], dict[str, float]]:
        embed_start = time.perf_counter()
        query_vector = self.embedder.embed_query(question)
        query_embedding_ms = (time.perf_counter() - embed_start) * 1000.0

        if float(np.linalg.norm(query_vector)) <= 1e-12:
            raise PipelineError(
                f"A consulta nao gerou um vetor valido usando o metodo '{self.method_name}'."
            )

        search_start = time.perf_counter()
        indices, scores = exact_cosine_search(query_vector, self.embeddings, top_k=top_k)
        retrieval_ms = (time.perf_counter() - search_start) * 1000.0
        return self._build_hits(indices, scores), {
            "query_embedding": round(query_embedding_ms, 3),
            "retrieval": round(retrieval_ms, 3),
        }

    def _build_hits(self, indices: np.ndarray, scores: np.ndarray) -> list[dict[str, object]]:
        hits: list[dict[str, object]] = []
        for rank, (idx, score) in enumerate(zip(indices.tolist(), scores.tolist()), start=1):
            row = self.movies_df.iloc[idx]
            synopsis = str(row["synopsis"])
            hits.append(
                {
                    "rank": rank,
                    "index": int(idx),
                    "movie_id": int(row["movie_id"]),
                    "title": str(row["title"]),
                    "score": float(score),
                    "synopsis": synopsis,
                    "synopsis_preview": synopsis[:320] + ("..." if len(synopsis) > 320 else ""),
                }
            )
        return hits


class HNSWRetriever:
    def __init__(
        self,
        movies_df: pd.DataFrame,
        index: HNSWSearch,
        embedder: QueryEmbedderAdapter,
        method_name: str,
        description: str,
    ) -> None:
        self.movies_df = movies_df
        self.index = index
        self.embedder = embedder
        self.method_name = method_name
        self.description = description

    def search(self, question: str, top_k: int) -> tuple[list[dict[str, object]], dict[str, float]]:
        top_k = min(top_k, len(self.movies_df))
        embed_start = time.perf_counter()
        query_vector = self.embedder.embed_query(question)
        query_embedding_ms = (time.perf_counter() - embed_start) * 1000.0

        if float(np.linalg.norm(query_vector)) <= 1e-12:
            raise PipelineError("A consulta nao gerou embedding valido para o indice HNSW.")

        search_start = time.perf_counter()
        indices, distances = self.index.query(query_vector, top_k=top_k)
        retrieval_ms = (time.perf_counter() - search_start) * 1000.0
        scores = 1.0 - np.asarray(distances, dtype=np.float32)
        return self._build_hits(indices, scores), {
            "query_embedding": round(query_embedding_ms, 3),
            "retrieval": round(retrieval_ms, 3),
        }

    def _build_hits(self, indices: np.ndarray, scores: np.ndarray) -> list[dict[str, object]]:
        hits: list[dict[str, object]] = []
        for rank, (idx, score) in enumerate(zip(indices.tolist(), scores.tolist()), start=1):
            row = self.movies_df.iloc[idx]
            synopsis = str(row["synopsis"])
            hits.append(
                {
                    "rank": rank,
                    "index": int(idx),
                    "movie_id": int(row["movie_id"]),
                    "title": str(row["title"]),
                    "score": float(score),
                    "synopsis": synopsis,
                    "synopsis_preview": synopsis[:320] + ("..." if len(synopsis) > 320 else ""),
                }
            )
        return hits


class QueryPipeline:
    def __init__(
        self,
        movies_df: pd.DataFrame | None,
        llm_client: TinyLlamaClient,
        dataset_error: str | None = None,
    ) -> None:
        self.movies_df = movies_df
        self.llm_client = llm_client
        self.dataset_error = dataset_error
        self.retrievers: dict[str, ExactRetriever | HNSWRetriever] = {}
        self.method_infos: dict[str, MethodInfo] = {}

    @property
    def movies_count(self) -> int:
        return 0 if self.movies_df is None else int(len(self.movies_df))

    def register_method(
        self,
        name: str,
        description: str,
        retriever: ExactRetriever | HNSWRetriever | None = None,
        reason: str = "",
        resolved_method: str | None = None,
    ) -> None:
        available = retriever is not None
        if retriever is not None:
            self.retrievers[name] = retriever
        self.method_infos[name] = MethodInfo(
            name=name,
            description=description,
            available=available,
            reason=reason,
            resolved_method=resolved_method,
        )

    def available_methods(self) -> list[str]:
        return [name for name, info in self.method_infos.items() if info.available]

    def methods_payload(self) -> dict[str, dict[str, object]]:
        payload: dict[str, dict[str, object]] = {}
        for name, info in self.method_infos.items():
            payload[name] = {
                "description": info.description,
                "available": info.available,
                "reason": info.reason,
                "resolved_method": info.resolved_method,
            }
        return payload

    def query(self, question: str, method: str, top_k: int) -> dict[str, object]:
        requested_method = method.lower().strip()
        info = self.method_infos.get(requested_method)
        if info is None:
            raise MethodUnavailableError(
                f"Metodo '{method}' desconhecido. Disponiveis: {', '.join(sorted(self.method_infos))}."
            )
        if not info.available or requested_method not in self.retrievers:
            reason = info.reason or "Metodo ainda nao disponivel no ambiente atual."
            raise MethodUnavailableError(f"Metodo '{method}' indisponivel: {reason}")

        total_start = time.perf_counter()
        retriever = self.retrievers[requested_method]
        results, timings = retriever.search(question=question, top_k=top_k)

        llm_start = time.perf_counter()
        generation: GenerationResult = self.llm_client.generate_answer(
            question=question,
            results=results,
            retrieval_method=requested_method,
        )
        llm_ms = (time.perf_counter() - llm_start) * 1000.0
        total_ms = (time.perf_counter() - total_start) * 1000.0

        public_results = []
        for item in results:
            public_results.append(
                {
                    "rank": item["rank"],
                    "movie_id": item["movie_id"],
                    "title": item["title"],
                    "score": round(float(item["score"]), 4),
                    "synopsis_preview": item["synopsis_preview"],
                }
            )

        return {
            "question": question,
            "method": requested_method,
            "resolved_method": info.resolved_method or requested_method,
            "top_k": top_k,
            "answer": generation.answer,
            "llm_backend": generation.backend,
            "results": public_results,
            "warnings": generation.warnings,
            "timings_ms": {
                "query_embedding": timings["query_embedding"],
                "retrieval": timings["retrieval"],
                "llm": round(llm_ms, 3),
                "total": round(total_ms, 3),
            },
            "available_methods": self.available_methods(),
        }


def _read_movies_dataframe(primary_path: Path, csv_fallback_path: Path) -> pd.DataFrame:
    attempts: list[str] = []
    for path in (primary_path, csv_fallback_path):
        if not path.exists():
            attempts.append(f"{path}: arquivo ausente")
            continue
        try:
            if path.suffix == ".csv":
                df = pd.read_csv(path)
            else:
                df = pd.read_parquet(path)
            required_columns = {"movie_id", "title", "synopsis"}
            missing = required_columns.difference(df.columns)
            if missing:
                raise ValueError(f"colunas obrigatorias ausentes: {sorted(missing)}")
            return df.reset_index(drop=True)
        except Exception as exc:  # noqa: BLE001
            attempts.append(f"{path}: {exc}")
    raise PipelineError("Nao foi possivel carregar a base de filmes. Tentativas: " + " | ".join(attempts))


def _load_numpy(path: Path) -> np.ndarray:
    return np.load(path, mmap_mode="r").astype(np.float32)


def _load_word2vec_embedder(model_path: Path) -> QueryEmbedderAdapter:
    embedder = Word2VecAverageEmbedder.load_model(str(model_path))
    return QueryEmbedderAdapter(embedder, name="word2vec")


def _build_sentence_embedder() -> QueryEmbedderAdapter | None:
    factory_spec = os.getenv("PAA_SENTENCE_ENCODER_FACTORY", "").strip()
    model_name = os.getenv("PAA_SENTENCE_MODEL_NAME", "").strip()

    if factory_spec:
        obj = _load_from_factory_spec(factory_spec, model_name=model_name)
        if obj is not None:
            return QueryEmbedderAdapter(obj, name="sentence")

    if importlib.util.find_spec("src.embeddings.sentence_embeddings") is not None:
        module = importlib.import_module("src.embeddings.sentence_embeddings")
        for attr_name in (
            "load_sentence_embedder",
            "load_model",
            "get_embedder",
            "SentenceEmbedder",
            "SentenceEmbeddingPipeline",
            "SentenceEmbeddingModel",
            "SentenceEmbeddingsModel",
        ):
            if hasattr(module, attr_name):
                obj = _call_factory(getattr(module, attr_name), model_name=model_name)
                if obj is not None:
                    return QueryEmbedderAdapter(obj, name="sentence")

    if model_name:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError:
            return None
        model = SentenceTransformer(model_name)
        return QueryEmbedderAdapter(model, name="sentence")

    return None


def _load_from_factory_spec(factory_spec: str, model_name: str) -> Any | None:
    if ":" not in factory_spec:
        raise PipelineError(
            "PAA_SENTENCE_ENCODER_FACTORY deve seguir o formato modulo:atributo."
        )
    module_name, attr_name = factory_spec.split(":", 1)
    module = importlib.import_module(module_name)
    if not hasattr(module, attr_name):
        raise PipelineError(f"Atributo '{attr_name}' nao encontrado em '{module_name}'.")
    return _call_factory(getattr(module, attr_name), model_name=model_name)


def _call_factory(factory: Any, model_name: str) -> Any | None:
    if not callable(factory):
        return factory
    for kwargs in (
        {"model_name": model_name} if model_name else None,
        {"model_name_or_path": model_name} if model_name else None,
        {},
    ):
        if kwargs is None:
            continue
        try:
            return factory(**kwargs)
        except TypeError:
            continue
    try:
        return factory()
    except TypeError:
        return None


def build_default_pipeline() -> QueryPipeline:
    movies_path = Path(os.getenv("PAA_MOVIES_PATH", PROJECT_ROOT / "data/processed/movies.parquet"))
    movies_csv_path = Path(os.getenv("PAA_MOVIES_CSV_PATH", PROJECT_ROOT / "data/processed/movies.csv"))
    word2vec_embeddings_path = Path(
        os.getenv("PAA_WORD2VEC_EMBEDDINGS", PROJECT_ROOT / "artifacts/word2vec_embeddings.npy")
    )
    word2vec_model_path = Path(
        os.getenv("PAA_WORD2VEC_MODEL", PROJECT_ROOT / "artifacts/word2vec_embeddings.model.npz")
    )
    sentence_embeddings_path = Path(
        os.getenv("PAA_SENTENCE_EMBEDDINGS", PROJECT_ROOT / "artifacts/sentence_embeddings.npy")
    )
    hnsw_index_path = Path(os.getenv("PAA_HNSW_INDEX", PROJECT_ROOT / "artifacts/hnsw_index.bin"))

    llm_client = TinyLlamaClient(TinyLlamaConfig.from_env())

    movies_df: pd.DataFrame | None
    dataset_error: str | None = None
    try:
        movies_df = _read_movies_dataframe(movies_path, movies_csv_path)
    except PipelineError as exc:
        movies_df = None
        dataset_error = str(exc)

    pipeline = QueryPipeline(movies_df=movies_df, llm_client=llm_client, dataset_error=dataset_error)

    word2vec_description = "Busca exata por similaridade de cosseno sobre embeddings Word2Vec Average."
    sentence_description = "Busca exata por similaridade de cosseno sobre sentence embeddings."
    hnsw_description = "Busca aproximada HNSW sobre sentence embeddings."
    cosine_description = (
        "Alias para a melhor busca exata disponivel: prioriza sentence embeddings e cai para Word2Vec."
    )

    if movies_df is None:
        reason = dataset_error or "Base de filmes indisponivel."
        pipeline.register_method("word2vec", word2vec_description, reason=reason)
        pipeline.register_method("sentence", sentence_description, reason=reason)
        pipeline.register_method("hnsw", hnsw_description, reason=reason)
        pipeline.register_method("cosine", cosine_description, reason=reason)
        return pipeline

    sentence_embedder_error = ""
    try:
        sentence_embedder = _build_sentence_embedder()
    except Exception as exc:  # noqa: BLE001
        sentence_embedder = None
        sentence_embedder_error = str(exc)

    word2vec_retriever: ExactRetriever | None = None
    if not word2vec_embeddings_path.exists():
        pipeline.register_method("word2vec", word2vec_description, reason="Arquivo de embeddings Word2Vec ausente.")
    elif not word2vec_model_path.exists():
        pipeline.register_method("word2vec", word2vec_description, reason="Modelo Word2Vec (.model.npz) ausente.")
    else:
        word2vec_embeddings = _load_numpy(word2vec_embeddings_path)
        if word2vec_embeddings.shape[0] != len(movies_df):
            pipeline.register_method(
                "word2vec",
                word2vec_description,
                reason=(
                    f"Embeddings Word2Vec com {word2vec_embeddings.shape[0]} linhas, "
                    f"mas a base tem {len(movies_df)} filmes."
                ),
            )
        else:
            word2vec_retriever = ExactRetriever(
                movies_df=movies_df,
                embeddings=word2vec_embeddings,
                embedder=_load_word2vec_embedder(word2vec_model_path),
                method_name="word2vec",
                description=word2vec_description,
            )
            pipeline.register_method("word2vec", word2vec_description, retriever=word2vec_retriever)

    sentence_retriever: ExactRetriever | None = None
    if not sentence_embeddings_path.exists():
        pipeline.register_method("sentence", sentence_description, reason="Arquivo sentence_embeddings.npy ausente.")
    elif sentence_embedder is None:
        pipeline.register_method(
            "sentence",
            sentence_description,
            reason=(
                sentence_embedder_error
                or "Encoder de consulta para sentence embeddings ainda nao esta disponivel neste ambiente."
            ),
        )
    else:
        sentence_embeddings = _load_numpy(sentence_embeddings_path)
        if sentence_embeddings.shape[0] != len(movies_df):
            pipeline.register_method(
                "sentence",
                sentence_description,
                reason=(
                    f"Sentence embeddings com {sentence_embeddings.shape[0]} linhas, "
                    f"mas a base tem {len(movies_df)} filmes."
                ),
            )
        else:
            sentence_retriever = ExactRetriever(
                movies_df=movies_df,
                embeddings=sentence_embeddings,
                embedder=sentence_embedder,
                method_name="sentence",
                description=sentence_description,
            )
            pipeline.register_method("sentence", sentence_description, retriever=sentence_retriever)

    if HNSW_IMPORT_ERROR is not None:
        pipeline.register_method(
            "hnsw",
            hnsw_description,
            reason=f"Dependencia do HNSW indisponivel: {HNSW_IMPORT_ERROR}",
        )
    elif not hnsw_index_path.exists():
        pipeline.register_method("hnsw", hnsw_description, reason="Indice HNSW ausente.")
    elif sentence_embedder is None:
        pipeline.register_method(
            "hnsw",
            hnsw_description,
            reason="HNSW depende do mesmo encoder de query dos sentence embeddings.",
        )
    else:
        try:
            hnsw_index = HNSWSearch.load(str(hnsw_index_path))
            if hnsw_index.n_elements != len(movies_df):
                pipeline.register_method(
                    "hnsw",
                    hnsw_description,
                    reason=(
                        f"Indice HNSW com {hnsw_index.n_elements} itens, "
                        f"mas a base tem {len(movies_df)} filmes."
                    ),
                )
            else:
                pipeline.register_method(
                    "hnsw",
                    hnsw_description,
                    retriever=HNSWRetriever(
                        movies_df=movies_df,
                        index=hnsw_index,
                        embedder=sentence_embedder,
                        method_name="hnsw",
                        description=hnsw_description,
                    ),
                )
        except Exception as exc:  # noqa: BLE001
            pipeline.register_method("hnsw", hnsw_description, reason=f"Falha ao carregar indice HNSW: {exc}")

    if sentence_retriever is not None:
        pipeline.register_method(
            "cosine",
            cosine_description,
            retriever=sentence_retriever,
            resolved_method="sentence",
        )
    elif word2vec_retriever is not None:
        pipeline.register_method(
            "cosine",
            cosine_description,
            retriever=word2vec_retriever,
            resolved_method="word2vec",
        )
    else:
        pipeline.register_method(
            "cosine",
            cosine_description,
            reason="Nenhuma busca exata disponivel no momento.",
        )

    return pipeline
