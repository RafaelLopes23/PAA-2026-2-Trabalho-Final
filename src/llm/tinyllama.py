"""
Cliente leve para um TinyLlama local.

Por padrao, usa a API HTTP do Ollama em `http://localhost:11434/api/generate`,
o que evita acoplar o servidor FastAPI a um runtime especifico de GPU/CPU.
Se o backend local nao estiver disponivel, o modulo cai para um resumo
deterministico baseado apenas nos resultados recuperados.
"""

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

    def generate_answer(
        self,
        question: str,
        results: Sequence[dict[str, object]],
        retrieval_method: str,
    ) -> GenerationResult:
        if not results:
            return GenerationResult(
                answer=self._format_answer(
                    title="nao foi possivel determinar",
                    justification="Nao encontrei sinopses relevantes o suficiente para responder com seguranca.",
                    alternatives="nenhuma",
                ),
                backend="no-results",
            )

        prompt = self._build_prompt(question, results, retrieval_method)
        if not self.config.enabled:
            return GenerationResult(
                answer=self._fallback_answer(results, unavailable_reason="LLM local desabilitado."),
                backend="fallback",
                warnings=["LLM local desabilitado; resposta formatada sem TinyLlama."],
            )

        warnings: list[str] = []

        if self.config.backend in {"auto", "ollama"}:
            try:
                answer = self._normalize_answer(self._call_ollama(prompt))
                return GenerationResult(answer=answer, backend=f"ollama:{self.config.ollama_model}")
            except (urllib.error.URLError, TimeoutError) as exc:
                warnings.append(f"Ollama indisponivel: {exc}")
                if self.config.backend == "ollama":
                    return GenerationResult(
                        answer=self._fallback_answer(results, unavailable_reason=str(exc)),
                        backend="fallback",
                        warnings=warnings,
                    )
            except ValueError as exc:
                warnings.append(f"Ollama gerou saida invalida: {exc}")
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
                warnings.append(f"Transformers gerou saida invalida: {exc}")
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"Transformers indisponivel: {exc}")

        reason = "; ".join(warnings) if warnings else "nenhum backend local disponivel"
        return GenerationResult(
            answer=self._fallback_answer(results, unavailable_reason=reason),
            backend="fallback",
            warnings=warnings or ["Nenhum backend de LLM local disponivel."],
        )

    def _call_ollama(self, prompt: str) -> str:
        payload = json.dumps(
            {
                "model": self.config.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                },
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.config.endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
            body = response.read().decode("utf-8")
        data = json.loads(body)
        answer = data.get("response", "").strip()
        if not answer:
            raise ValueError("O backend do TinyLlama respondeu sem texto.")
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
            "Voce e um assistente de filmes. "
            "Responda em portugues do Brasil usando apenas as sinopses recuperadas. "
            "Nao copie a pergunta nem repita a lista de resultados. "
            "Comece diretamente pela resposta final. "
            "Se nao houver evidencia suficiente, diga isso explicitamente."
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
            raise ValueError("O backend transformers respondeu sem texto.")
        return answer

    @staticmethod
    def _normalize_answer(answer: str) -> str:
        cleaned = answer.strip()
        markers = ("Filme provável:", "Filme provavel:")
        for marker in markers:
            if marker in cleaned:
                normalized = cleaned[cleaned.index(marker) :].strip()
                if "<" in normalized or ">" in normalized:
                    raise ValueError("O LLM respondeu com placeholders em vez de um titulo real.")
                return normalized

        if any(token in cleaned for token in ("Pergunta do usuario", "Similaridade", "Sinopse:", "Titulo:")):
            raise ValueError("O LLM repetiu o contexto em vez de produzir a resposta final.")

        raise ValueError("O LLM nao seguiu o formato esperado.")

    def _build_prompt(
        self,
        question: str,
        results: Sequence[dict[str, object]],
        retrieval_method: str,
    ) -> str:
        lines = [
            f"Metodo de recuperacao: {retrieval_method}.",
            f"Pergunta do usuario: {question}",
            "",
            "Use apenas estes resultados recuperados:",
        ]
        for idx, item in enumerate(results[: self.config.max_results_in_prompt], start=1):
            synopsis = str(item["synopsis"])[: self.config.max_synopsis_chars]
            lines.extend(
                [
                    f"{idx}. {item['title']} (similaridade {float(item['score']):.4f})",
                    f"   {synopsis}",
                    "",
                ]
            )
        lines.extend(
            [
                "Responda exatamente neste formato:",
                "Filme provável: <titulo ou nao foi possivel determinar>",
                "Justificativa: <uma explicacao curta baseada nas sinopses>",
                "Alternativas: <titulos alternativos ou nenhuma>",
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _fallback_answer(
        results: Sequence[dict[str, object]],
        unavailable_reason: str,
    ) -> str:
        titles = [str(item["title"]) for item in results[:3]]
        first_title = titles[0] if titles else "desconhecido"
        alternatives = ", ".join(titles[1:]) if len(titles) > 1 else "nenhuma alternativa adicional"
        return TinyLlamaClient._format_answer(
            title=first_title,
            justification=(
                "O backend local nao produziu uma resposta confiavel. "
                "Com base apenas nas sinopses recuperadas, este foi o item mais similar."
            ),
            alternatives=alternatives,
        )

    @staticmethod
    def _format_answer(title: str, justification: str, alternatives: str) -> str:
        return (
            f"Filme provável: {title}\n"
            f"Justificativa: {justification}\n"
            f"Alternativas: {alternatives}"
        )
