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
                answer = self._normalize_answer(self._call_ollama(prompt))
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
                answer = self._normalize_answer(self._call_transformers(prompt))
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
        system_prompt = "You are a movie assistant. Answer in English using only the provided synopses and the requested format."
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
            "You are a movie assistant. "
            "Answer in English using only the retrieved synopses. "
            "Do not copy the question or repeat the result list. "
            "Start directly with the final answer. "
            "If there is not enough evidence, say so explicitly."
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

    @staticmethod
    def _normalize_answer(answer: str) -> str:
        cleaned = answer.strip()
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
                
                # Clean any accidental brackets or angle brackets wrapping title/text
                normalized = normalized.replace("<", "").replace(">", "").replace("[", "").replace("]", "")
                return normalized

        if any(token in cleaned for token in ("User question", "Similarity", "Synopsis:", "Title:", "Retrieved Movies:")):
            print(f"--- LLM VALIDATION FAILURE (repeated context) ---\nRaw answer:\n{answer}\n------------------------------", flush=True)
            raise ValueError("The LLM repeated the context instead of producing the final answer.")

        # Loose parsing fallback: if the LLM produced a non-empty paragraph, accept it as the justification
        if cleaned:
            justification = cleaned.replace("<", "").replace(">", "").replace("[", "").replace("]", "")
            return TinyLlamaClient._format_answer(
                title="Unable to determine",
                justification=justification,
                alternatives="None",
            )

        raise ValueError("The LLM did not follow the expected format.")

    def _build_prompt(
        self,
        question: str,
        results: Sequence[dict[str, object]],
        retrieval_method: str,
    ) -> str:
        lines = [
            "Instruction: Based on the movie synopses below, identify which movie best matches the user's Question.",
            "If no movie matches the Question, write 'Unable to determine' as the likely movie.",
            "",
            "Response format (replace the text inside brackets with your answer):",
            "Likely movie: [Best matching movie title]",
            "Reason: [Short explanation in English based on the synopses]",
            "Alternatives: [Other matching movie titles from the list, or None]",
            "",
            f"Question: {question}",
            "",
            "Retrieved Movies:",
        ]
        for idx, item in enumerate(results[: self.config.max_results_in_prompt], start=1):
            synopsis = str(item["synopsis"])[: self.config.max_synopsis_chars]
            lines.extend(
                [
                    f"- {item['title']}: {synopsis}",
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
            justification=(
                "This was the closest retrieved synopsis for the query. "
                "The answer is based only on the semantic search results."
            ),
            alternatives=alternatives,
        )

    @staticmethod
    def _format_answer(title: str, justification: str, alternatives: str) -> str:
        return (
            f"Likely movie: {title}\n"
            f"Reason: {justification}\n"
            f"Alternatives: {alternatives}"
        )
