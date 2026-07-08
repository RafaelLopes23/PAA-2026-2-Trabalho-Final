"""Lightweight client for a local TinyLlama model."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass
class TinyLlamaConfig:
    enabled: bool = True
    backend: str = "auto"
    ollama_model: str = "tinyllama"
    transformers_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    endpoint: str = "http://localhost:11434/api/generate"
    timeout_seconds: float = 60.0
    temperature: float = 0.1
    max_results_in_prompt: int = 5
    max_synopsis_chars: int = 700
    max_new_tokens: int = 220
    cache_dir: str = ".cache/huggingface"
    local_files_only: bool = False

    @classmethod
    def from_env(cls) -> "TinyLlamaConfig":
        enabled = os.getenv("PAA_LLM_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
        return cls(
            enabled=enabled,
            backend=os.getenv("PAA_LLM_BACKEND", "auto"),
            ollama_model=os.getenv("PAA_LLM_MODEL", os.getenv("PAA_LLM_OLLAMA_MODEL", "tinyllama")),
            transformers_model=os.getenv(
                "PAA_LLM_TRANSFORMERS_MODEL",
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            ),
            endpoint=os.getenv("PAA_LLM_ENDPOINT", "http://localhost:11434/api/generate"),
            timeout_seconds=float(os.getenv("PAA_LLM_TIMEOUT_SECONDS", "60")),
            temperature=float(os.getenv("PAA_LLM_TEMPERATURE", "0.1")),
            max_results_in_prompt=int(os.getenv("PAA_LLM_MAX_RESULTS", "5")),
            max_synopsis_chars=int(os.getenv("PAA_LLM_MAX_SYNOPSIS_CHARS", "700")),
            max_new_tokens=int(os.getenv("PAA_LLM_MAX_NEW_TOKENS", "220")),
            cache_dir=(
                os.getenv("PAA_LLM_CACHE_DIR")
                or os.getenv("PAA_HF_CACHE_DIR")
                or os.getenv("HF_HOME")
                or ".cache/huggingface"
            ),
            local_files_only=os.getenv("PAA_HF_LOCAL_ONLY", "0").strip().lower() in {"1", "true", "yes"},
        )


@dataclass
class GenerationResult:
    answer: str
    backend: str
    warnings: list[str] = field(default_factory=list)


class TinyLlamaClient:
    def __init__(self, config: TinyLlamaConfig | None = None) -> None:
        self.config = config or TinyLlamaConfig.from_env()
        self._tokenizer = None
        self._model = None

    def translate_to_english(self, text: str) -> str:
        if not self.config.enabled:
            return text
        try:
            prompt = (
                "Translate the following movie query from Portuguese to English.\n"
                "Portuguese: astronautas no espaço\n"
                "English: astronauts in space\n\n"
                "Portuguese: filme de ação com carros\n"
                "English: action movie with cars\n\n"
                f"Portuguese: {text}\n"
                "English:"
            )
            payload = json.dumps(
                {
                    "model": self.config.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "raw": True,
                    "options": {
                        "temperature": 0.0,
                        "num_predict": 40,
                        "stop": ["\n"],
                    },
                }
            ).encode("utf-8")
            request = urllib.request.Request(
                self.config.endpoint,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=3.0) as response:
                body = response.read().decode("utf-8")
            data = json.loads(body)
            translation = data.get("response", "").strip()
            # Clean quotes if any
            if translation.startswith('"') and translation.endswith('"'):
                translation = translation[1:-1].strip()
            if translation.startswith("'") and translation.endswith("'"):
                translation = translation[1:-1].strip()
            if translation:
                return translation
        except Exception as exc:
            print(f"Translation failed: {exc}", flush=True)
        return text

    def generate_answer(
        self,
        question: str,
        results: Sequence[dict[str, object]],
        retrieval_method: str,
    ) -> GenerationResult:
        if not results:
            return GenerationResult(
                answer=self._format_answer(
                    title="Unable to determine",
                    justification="No sufficiently relevant synopses were retrieved.",
                    alternatives="None",
                ),
                backend="no-results",
            )

        prompt = self._build_prompt(question, results, retrieval_method)
        if not self.config.enabled:
            return GenerationResult(
                answer=self._fallback_answer(results, unavailable_reason="Local LLM disabled."),
                backend="fallback",
                warnings=["Local LLM disabled; response was formatted from retrieved results."],
            )

        warnings: list[str] = []

        if self.config.backend in {"auto", "ollama"}:
            try:
                answer = self._normalize_answer(self._call_ollama(prompt), results)
                return GenerationResult(answer=answer, backend=f"ollama:{self.config.ollama_model}")
            except (urllib.error.URLError, TimeoutError) as exc:
                warnings.append(f"Ollama unavailable: {exc}")
                if self.config.backend == "ollama":
                    return GenerationResult(
                        answer=self._fallback_answer(results, unavailable_reason=str(exc)),
                        backend="fallback",
                        warnings=warnings,
                    )
            except ValueError as exc:
                warnings.append(f"Ollama returned invalid output: {exc}")
                if self.config.backend == "ollama":
                    return GenerationResult(
                        answer=self._fallback_answer(results, unavailable_reason=str(exc)),
                        backend="fallback",
                        warnings=warnings,
                    )

        if self.config.backend in {"auto", "transformers"}:
            try:
                answer = self._normalize_answer(self._call_transformers(prompt), results)
                return GenerationResult(
                    answer=answer,
                    backend=f"transformers:{self.config.transformers_model}",
                    warnings=warnings,
                )
            except ValueError as exc:
                warnings.append(f"Transformers returned invalid output: {exc}")
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"Transformers unavailable: {exc}")

        reason = "; ".join(warnings) if warnings else "no local backend available"
        return GenerationResult(
            answer=self._fallback_answer(results, unavailable_reason=reason),
            backend="fallback",
            warnings=warnings or ["No local LLM backend available."],
        )

    def _call_ollama(self, prompt: str) -> str:
        system_prompt = (
            "You are a movie search critic. The retrieval engine already ranked the candidate movies. "
            "Answer in English using only those candidates. Never invent a movie title."
        )
        # Adapt endpoint from /api/generate to /api/chat to let Ollama format messages correctly
        chat_endpoint = self.config.endpoint.replace("/api/generate", "/api/chat")
        payload = json.dumps(
            {
                "model": self.config.ollama_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_new_tokens,
                },
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            chat_endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
            body = response.read().decode("utf-8")
        data = json.loads(body)
        answer = data.get("message", {}).get("content", "").strip()
        if not answer:
            raise ValueError("TinyLlama returned an empty response.")
        return answer

    def _call_transformers(self, prompt: str) -> str:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self._tokenizer is None or self._model is None:
            cache_dir = Path(self.config.cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.transformers_model,
                cache_dir=str(cache_dir),
                local_files_only=self.config.local_files_only,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.transformers_model,
                cache_dir=str(cache_dir),
                torch_dtype=torch.float32,
                local_files_only=self.config.local_files_only,
            )
            self._model.eval()
            if self._tokenizer.pad_token is None and self._tokenizer.eos_token is not None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

        system_prompt = (
            "You are a movie search critic. "
            "The retrieval engine already ranked the candidate movies. "
            "Answer in English using only those candidates. "
            "Never invent a movie title. "
            "Do not copy the question or repeat the result list. "
            "Start directly with the final answer. "
            "If the evidence is weak, explain that while still commenting on the retrieved ranking."
        )

        if hasattr(self._tokenizer, "apply_chat_template") and self._tokenizer.chat_template:
            rendered_prompt = self._tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            rendered_prompt = (
                f"### System:\n{system_prompt}\n\n"
                f"### User:\n{prompt}\n\n"
                "### Assistant:\n"
            )

        inputs = self._tokenizer(
            rendered_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                temperature=max(self.config.temperature, 1e-5),
                repetition_penalty=1.1,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[1] :]
        answer = self._tokenizer.decode(generated, skip_special_tokens=True).strip()
        answer = answer.removeprefix("Assistant:").strip()
        answer = answer.removeprefix("### Assistant:").strip()
        if not answer:
            raise ValueError("The transformers backend returned an empty response.")
        return answer

    @classmethod
    def _normalize_answer(cls, answer: str, results: Sequence[dict[str, object]]) -> str:
        cleaned = answer.strip()
        title_map = cls._retrieved_title_map(results)
        valid_titles = set(title_map)

        if not cleaned:
            raise ValueError("The LLM did not follow the expected format.")
        repeated_context_tokens = ("User question", "Similarity", "Synopsis:", "Title:", "Retrieved Movies:")
        if any(token in cleaned for token in repeated_context_tokens):
            print(f"--- LLM VALIDATION FAILURE (repeated context) ---\nRaw answer:\n{answer}\n------------------------------", flush=True)
            raise ValueError("The LLM repeated the context instead of producing the final answer.")

        markers = ("Likely movie:", "Best match:", "Movie:")
        for marker in markers:
            marker_index = cleaned.lower().find(marker.lower())
            if marker_index >= 0:
                normalized = cleaned[marker_index:].strip()
                
                # Check for actual unresolved placeholder strings
                placeholders = (
                    "movie title",
                    "short explanation", "alternative titles",
                    "best matching", "short explanation", "other matching", "other movie",
                    "write the title", "write a"
                )
                if any(p in normalized.lower() for p in placeholders):
                    print(f"--- LLM VALIDATION FAILURE (placeholders) ---\nRaw answer:\n{answer}\n------------------------------", flush=True)
                    raise ValueError("The LLM returned placeholders instead of a real title.")

                parsed_title = cls._extract_field(normalized, ("Likely movie", "Best match", "Movie"))
                title_key = cls._title_key(parsed_title)
                if title_key == cls._title_key("Unable to determine") and results:
                    raise ValueError("The LLM declined to choose among retrieved results.")
                if title_key not in valid_titles:
                    raise ValueError(f"The LLM selected a title outside the retrieved results: {parsed_title}")

                reason = cls._extract_field(normalized, ("Reason", "Justification"))
                alternatives = cls._extract_field(normalized, ("Alternatives", "Alternative"))
                return cls._format_answer(
                    title=title_map[title_key],
                    justification=cls._clean_text(reason) or cls._fallback_reason(results[0]),
                    alternatives=cls._normalize_alternatives(alternatives, title_map, selected_title_key=title_key),
                )

        raise ValueError("The LLM did not follow the expected format.")

    def _build_prompt(
        self,
        question: str,
        results: Sequence[dict[str, object]],
        retrieval_method: str,
    ) -> str:
        lines = [
            "Instruction: comment on the ranked movie results produced by the search method.",
            "The search method is the source of truth for candidate movies.",
            "Choose the likely movie only from the retrieved titles below. Prefer Rank 1 unless another retrieved synopsis is clearly a better match.",
            "Never mention a title that is not in the retrieved list.",
            "If the evidence is weak, explain the uncertainty in the Reason, but still use a retrieved title.",
            "",
            "Response format (replace the text inside brackets with your answer):",
            "Likely movie: [exact retrieved movie title]",
            "Reason: [short English comment about how the retrieved synopsis relates to the question]",
            "Alternatives: [other exact retrieved movie titles, or None]",
            "",
            f"Search method: {retrieval_method}",
            f"Question: {question}",
            "",
            "Retrieved Movies, already ranked by the search method:",
        ]
        for idx, item in enumerate(results[: self.config.max_results_in_prompt], start=1):
            synopsis = str(item["synopsis"])[: self.config.max_synopsis_chars]
            score = float(item.get("score", 0.0))
            lines.extend(
                [
                    f"- Rank {idx} | Title: {item['title']} | Similarity: {score:.4f}",
                    f"  Synopsis: {synopsis}",
                ]
            )
        return "\n".join(lines)

    @staticmethod
    def _fallback_answer(
        results: Sequence[dict[str, object]],
        unavailable_reason: str,
    ) -> str:
        titles = [str(item["title"]) for item in results[:3]]
        first_title = titles[0] if titles else "Unknown"
        alternatives = ", ".join(titles[1:]) if len(titles) > 1 else "None"
        return TinyLlamaClient._format_answer(
            title=first_title,
            justification=TinyLlamaClient._fallback_reason(results[0]) if results else unavailable_reason,
            alternatives=alternatives,
        )

    @staticmethod
    def _fallback_reason(item: dict[str, object]) -> str:
        score = item.get("score")
        if isinstance(score, (int, float)):
            return (
                f"The search method ranked this movie first with similarity {float(score):.4f}. "
                "This analysis is based only on the retrieved synopsis and ranking."
            )
        return "The search method ranked this movie first among the retrieved candidates."

    @staticmethod
    def _retrieved_title_map(results: Sequence[dict[str, object]]) -> dict[str, str]:
        return {
            TinyLlamaClient._title_key(str(item["title"])): str(item["title"])
            for item in results
            if str(item.get("title", "")).strip()
        }

    @staticmethod
    def _title_key(value: str) -> str:
        cleaned = TinyLlamaClient._clean_text(value)
        return " ".join(cleaned.lower().split())

    @staticmethod
    def _extract_field(text: str, field_names: Sequence[str]) -> str:
        lines = text.splitlines()
        lower_names = tuple(name.lower() for name in field_names)
        for index, line in enumerate(lines):
            stripped = line.strip()
            lowered = stripped.lower()
            for name in lower_names:
                prefix = f"{name}:"
                if lowered.startswith(prefix):
                    value = stripped[len(prefix):].strip()
                    continuation: list[str] = []
                    for next_line in lines[index + 1:]:
                        next_stripped = next_line.strip()
                        next_lowered = next_stripped.lower()
                        field_markers = (
                            "likely movie",
                            "best match",
                            "movie",
                            "reason",
                            "justification",
                            "alternatives",
                            "alternative",
                        )
                        if any(next_lowered.startswith(f"{candidate}:") for candidate in field_markers):
                            break
                        if next_stripped:
                            continuation.append(next_stripped)
                    return TinyLlamaClient._clean_text(" ".join([value, *continuation]).strip())
        return ""

    @staticmethod
    def _normalize_alternatives(
        alternatives: str,
        title_map: dict[str, str],
        selected_title_key: str,
    ) -> str:
        if not alternatives or alternatives.strip().lower() == "none":
            return "None"
        selected: list[str] = []
        for chunk in alternatives.replace(";", ",").split(","):
            key = TinyLlamaClient._title_key(chunk)
            if key in title_map and key != selected_title_key and title_map[key] not in selected:
                selected.append(title_map[key])
        return ", ".join(selected) if selected else "None"

    @staticmethod
    def _clean_text(value: str) -> str:
        return value.strip().strip("\"'`").replace("<", "").replace(">", "").replace("[", "").replace("]", "")

    @staticmethod
    def _format_answer(title: str, justification: str, alternatives: str) -> str:
        return (
            f"Likely movie: {title}\n"
            f"Reason: {justification}\n"
            f"Alternatives: {alternatives}"
        )
