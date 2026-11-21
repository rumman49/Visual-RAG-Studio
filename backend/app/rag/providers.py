"""Explicit Ollama HTTP adapters. No provider or model fallback is performed."""

from __future__ import annotations

import importlib.util
import math
import re
from functools import lru_cache
from typing import Any

import httpx


class ProviderError(RuntimeError):
    pass


class SentenceTransformersProvider:
    """Local multilingual sentence embeddings, loaded lazily on first use."""

    def __init__(self, local_files_only: bool = False):
        self.local_files_only = local_files_only

    def embed(self, model: str | None, texts: list[str]) -> tuple[list[list[float]], int]:
        if not model:
            raise ProviderError("No Sentence Transformers model is configured.")
        if not texts or any(not text.strip() for text in texts):
            raise ProviderError("Embedding input must contain non-empty text.")
        if importlib.util.find_spec("sentence_transformers") is None:
            raise ProviderError(
                "sentence-transformers is not installed. Run `uv sync --group dev` in backend."
            )
        try:
            from sentence_transformers import SentenceTransformer

            encoder = _load_sentence_model(model, self.local_files_only, SentenceTransformer)
            encoded = encoder.encode(
                texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
            )
            vectors = [[float(value) for value in row] for row in encoded]
        except Exception as exc:
            raise ProviderError(
                f"Could not load or run Sentence Transformers model '{model}': {exc}"
            ) from exc
        if len(vectors) != len(texts) or not vectors or not vectors[0]:
            raise ProviderError("Sentence Transformers returned malformed embeddings.")
        dimension = len(vectors[0])
        if any(
            len(row) != dimension or not all(math.isfinite(value) for value in row)
            for row in vectors
        ):
            raise ProviderError(
                "Sentence Transformers returned inconsistent or non-finite embeddings."
            )
        return vectors, dimension

    @staticmethod
    def health(model: str, local_files_only: bool) -> dict:
        installed = importlib.util.find_spec("sentence_transformers") is not None
        return {
            "provider": "sentence_transformers",
            "configured": bool(model),
            "online": installed,
            "embedding_model": model,
            "embedding_model_available": installed and bool(model),
            "generation_model": None,
            "generation_model_available": None,
            "error": None
            if installed
            else "sentence-transformers is not installed in this Python environment.",
            "weights_load_on_first_use": bool(model) and not local_files_only,
        }


class FastLocalEmbeddingProvider:
    """Zero-dependency local feature embedding provider."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.model = "fast_local"

    def embed(self, model: str | None, texts: list[str]) -> tuple[list[list[float]], int]:
        if not texts or any(not text.strip() for text in texts):
            raise ProviderError("Embedding input must contain non-empty text.")
        try:
            import ragspark.embedder as r_embedder

            vectors = r_embedder.embed(texts, provider="fast_local", dimension=self.dimension)
            return [[float(v) for v in row] for row in vectors], self.dimension
        except Exception as exc:
            raise ProviderError(f"Fast local embedding failed: {exc}") from exc

    @staticmethod
    def health(model: str = "fast_local") -> dict:
        return {
            "provider": "fast_local",
            "configured": True,
            "online": True,
            "embedding_model": model or "fast_local",
            "embedding_model_available": True,
            "generation_model": None,
            "generation_model_available": None,
            "error": None,
            "weights_load_on_first_use": False,
        }


class ExtractiveGeneratorProvider:
    """Zero-dependency local extractive grounded generator."""

    def __init__(self):
        pass

    def generate(
        self,
        model: str | None,
        system: str,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> str:
        try:
            import json
            chunk_id = None
            evidence_text = prompt
            question_text = prompt
            try:
                data = json.loads(prompt)
                question_text = data.get("question", prompt)
                evidence = data.get("evidence", [])
                if evidence:
                    chunk_id = evidence[0].get("chunk_id")
                    evidence_text = "\n\n".join(e.get("text", "") for e in evidence)
            except Exception:
                pass

            import re
            stop_words = {
                "the", "and", "for", "are", "was", "were", "can", "has", "had", "not", "but",
                "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
                "does", "doing", "would", "could", "should", "shall", "will", "about",
                "there", "their", "these", "those", "have", "having", "been", "from",
                "with", "into", "that", "this", "some", "many", "most", "other",
                "kya", "kaun", "kaunsa", "kaise", "kyun", "kahan", "kab", "kisne", "kisi",
                "hai", "hain", "tha", "thi", "par", "aur", "mein", "liye", "karo", "gaya", "gayi", "hoga",
            }
            q_words = [
                w.lower()
                for w in re.findall(r"[\w]+", question_text)
                if len(w) >= 3 and w.lower() not in stop_words
            ]
            if q_words:
                ev_lower = evidence_text.lower()
                matches = sum(1 for w in q_words if w in ev_lower)
                if matches == 0:
                    return "I could not find enough retrieved evidence to answer that question."

            import ragspark.llm as r_llm

            res = r_llm.generate(
                prompt=question_text,
                provider="extractive",
                context=evidence_text,
                fallback_to_extractive=True,
            )
            text = (res.text or res.answer or str(res)).strip()
            if chunk_id and f"[{chunk_id}]" not in text:
                text = f"{text} [{chunk_id}]"
            return text
        except Exception as exc:
            raise ProviderError(f"Extractive generation failed: {exc}") from exc


@lru_cache(maxsize=2)
def _load_sentence_model(model: str, local_files_only: bool, model_class):
    return model_class(model, local_files_only=local_files_only, device="cpu")


def _model_name_match(configured: str | None, installed: set[str]) -> bool:
    if not configured:
        return False
    normalized = configured if ":" in configured else f"{configured}:latest"
    return normalized in installed


class OllamaProvider:
    def __init__(
        self, base_url: str, timeout_seconds: float = 90, client: httpx.Client | None = None
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.client = client

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        try:
            if self.client:
                response = self.client.request(
                    method, f"{self.base_url}{path}", timeout=self.timeout_seconds, **kwargs
                )
            else:
                with httpx.Client(timeout=self.timeout_seconds) as client:
                    response = client.request(method, f"{self.base_url}{path}", **kwargs)
        except httpx.ConnectError as exc:
            raise ProviderError(
                "Ollama is unavailable. Start Ollama and check the configured local service URL."
            ) from exc
        except httpx.TimeoutException as exc:
            raise ProviderError(
                "Ollama request timed out. Check the service and model load."
            ) from exc
        except httpx.RequestError as exc:
            raise ProviderError("Ollama request failed before a response was received.") from exc
        return response

    @staticmethod
    def _check_response(response: httpx.Response, operation: str) -> dict[str, Any]:
        if response.is_error:
            detail = response.text[:300].replace("\n", " ")
            raise ProviderError(
                f"Ollama {operation} failed with HTTP {response.status_code}: {detail}"
            )
        try:
            return response.json()
        except ValueError as exc:
            raise ProviderError(f"Ollama {operation} returned invalid JSON.") from exc

    def tags(self) -> set[str]:
        payload = self._check_response(self._request("GET", "/api/tags"), "readiness check")
        return {
            item["name"]
            for item in payload.get("models", [])
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        }

    def embed(self, model: str | None, texts: list[str]) -> tuple[list[list[float]], int]:
        if not model:
            raise ProviderError("No embedding model is configured. Set VRS_OLLAMA_EMBEDDING_MODEL.")
        if not texts or any(not text.strip() for text in texts):
            raise ProviderError("Embedding input must contain non-empty text.")
        response = self._request(
            "POST", "/api/embed", json={"model": model, "input": texts, "truncate": False}
        )
        if response.status_code in {404, 405} and "model" not in response.text.lower():
            rows = []
            for text in texts:
                legacy = self._check_response(
                    self._request("POST", "/api/embeddings", json={"model": model, "prompt": text}),
                    "embedding",
                )
                rows.append(legacy.get("embedding"))
        else:
            payload = self._check_response(response, "embedding")
            rows = payload.get("embeddings")
        if not isinstance(rows, list) or len(rows) != len(texts):
            raise ProviderError(
                "Ollama returned a different number of embeddings than input texts."
            )
        normalized: list[list[float]] = []
        dimension: int | None = None
        for row in rows:
            if not isinstance(row, list) or not row:
                raise ProviderError("Ollama returned an empty or malformed embedding.")
            vector = [float(value) for value in row]
            if not all(math.isfinite(value) for value in vector):
                raise ProviderError("Ollama returned non-finite embedding values.")
            dimension = dimension or len(vector)
            if len(vector) != dimension:
                raise ProviderError("Ollama returned embeddings with inconsistent dimensions.")
            normalized.append(vector)
        assert dimension is not None
        return normalized, dimension

    def generate(
        self,
        model: str | None,
        system: str,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        if not model:
            raise ProviderError(
                "No generation model is configured. Set VRS_OLLAMA_GENERATION_MODEL."
            )
        payload = self._check_response(
            self._request(
                "POST",
                "/api/chat",
                json={
                    "model": model,
                    "stream": False,
                    "think": False,
                    "options": {
                        **({"num_predict": max_tokens} if max_tokens is not None else {}),
                        **({"temperature": temperature} if temperature is not None else {}),
                    },
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                },
            ),
            "generation",
        )
        msg = payload.get("message", {}) if isinstance(payload.get("message"), dict) else {}
        content = (msg.get("content") or "").strip()
        response_text = (payload.get("response") or "").strip()
        thinking = (msg.get("thinking") or "").strip()
        result = content or response_text or thinking
        result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()
        if not result:
            raise ProviderError("Ollama returned an empty generation response.")
        return result

    def health(self, embedding_model: str | None, generation_model: str | None) -> dict:
        configured = bool(embedding_model and generation_model)
        try:
            installed = self.tags()
        except ProviderError as exc:
            return {
                "provider": "ollama",
                "configured": configured,
                "online": False,
                "embedding_model": embedding_model,
                "generation_model": generation_model,
                "embedding_model_available": False if embedding_model else None,
                "generation_model_available": False if generation_model else None,
                "error": str(exc),
            }
        embed_available = _model_name_match(embedding_model, installed) if embedding_model else None
        generation_available = (
            _model_name_match(generation_model, installed) if generation_model else None
        )
        return {
            "provider": "ollama",
            "configured": configured,
            "online": True,
            "embedding_model": embedding_model,
            "generation_model": generation_model,
            "embedding_model_available": embed_available,
            "generation_model_available": generation_available,
            "error": None,
        }


class HuggingFaceProvider:
    """Hugging Face Inference Providers chat adapter using the OpenAI-compatible API."""

    endpoint = "https://router.huggingface.co/v1/chat/completions"
    _model_pattern = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?::[A-Za-z0-9_.-]+)?$")

    def __init__(
        self, token: str | None, timeout_seconds: float = 90, client: httpx.Client | None = None
    ):
        self.token = token
        self.timeout_seconds = timeout_seconds
        self.client = client

    @classmethod
    def resolve_model(cls, model: str | None, policy: str) -> str:
        if not model or not cls._model_pattern.fullmatch(model):
            raise ProviderError("Set a valid Hugging Face model ID such as org/model.")
        if policy not in {"fastest", "cheapest", "preferred"}:
            raise ProviderError(
                "Hugging Face provider policy must be fastest, cheapest, or preferred."
            )
        return model if ":" in model else f"{model}:{policy}"

    def generate(
        self,
        model: str | None,
        system: str,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.2,
        policy: str = "fastest",
    ) -> str:
        if not self.token:
            raise ProviderError("HF_TOKEN is missing. Configure it in the backend environment.")
        resolved_model = self.resolve_model(model, policy)
        if not 1 <= max_tokens <= 8192 or not 0 <= temperature <= 2:
            raise ProviderError("Generation parameters are outside the supported limits.")
        payload = {
            "model": resolved_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        try:
            if self.client:
                response = self.client.post(
                    self.endpoint,
                    headers={"Authorization": f"Bearer {self.token}"},
                    json=payload,
                    timeout=self.timeout_seconds,
                )
            else:
                with httpx.Client(timeout=self.timeout_seconds) as client:
                    response = client.post(
                        self.endpoint,
                        headers={"Authorization": f"Bearer {self.token}"},
                        json=payload,
                    )
        except httpx.TimeoutException as exc:
            raise ProviderError("Hugging Face request timed out.") from exc
        except httpx.RequestError as exc:
            raise ProviderError(
                "Hugging Face request failed before a response was received."
            ) from exc
        if response.status_code in {401, 403}:
            raise ProviderError(
                "Hugging Face rejected the token. Check HF_TOKEN permissions for Inference Providers."
            )
        if response.status_code == 402:
            raise ProviderError(
                "Hugging Face Inference Providers has no available account credit; review billing."
            )
        if response.status_code == 429:
            detail = response.text.casefold()
            if any(word in detail for word in ("credit", "quota", "billing", "payment")):
                raise ProviderError(
                    "Hugging Face account quota or credits are exhausted; review billing and usage."
                )
            raise ProviderError(
                "Hugging Face rate limit reached; retry after the provider limit resets."
            )
        if response.status_code == 400:
            raise ProviderError("Hugging Face rejected the model or generation parameters.")
        if response.status_code == 404:
            raise ProviderError(
                "The requested model/provider is unavailable through Hugging Face Inference Providers."
            )
        if response.is_error:
            raise ProviderError(
                f"Hugging Face Inference Providers failed with HTTP {response.status_code}."
            )
        try:
            data = response.json()
            choices = data.get("choices")
            message = choices[0].get("message") if isinstance(choices, list) and choices else None
            content = message.get("content") if isinstance(message, dict) else None
        except (ValueError, AttributeError, IndexError, TypeError) as exc:
            raise ProviderError(
                "Hugging Face returned a malformed chat-completion response."
            ) from exc
        if not isinstance(content, str) or not content.strip():
            raise ProviderError("Hugging Face returned an empty chat completion.")
        return content.strip()


class GroqProvider:
    """Groq OpenAI-compatible chat completions; the key is supplied per request lifetime."""

    endpoint = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(
        self, api_key: str | None, timeout_seconds: float = 90, client: httpx.Client | None = None
    ):
        self.api_key = api_key.strip() if api_key and api_key.strip() else None
        self.timeout_seconds = timeout_seconds
        self.client = client

    def generate(
        self,
        model: str | None,
        system: str,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> str:
        if not self.api_key:
            raise ProviderError("Groq API key is missing. Add it in the generation settings.")
        if not model:
            raise ProviderError("Enter a Groq chat model ID.")
        if not 1 <= max_tokens <= 8192 or not 0 <= temperature <= 2:
            raise ProviderError("Generation parameters are outside the supported limits.")
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        # Groq's GPT-OSS models can spend their output budget on a separate
        # reasoning field, leaving message.content empty. We only display the
        # final answer in this product, so explicitly suppress reasoning for
        # that model family. Other Groq models may not accept this parameter.
        if model.startswith("openai/gpt-oss-"):
            payload["include_reasoning"] = False
        try:
            if self.client:
                response = self.client.post(
                    self.endpoint,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                    timeout=self.timeout_seconds,
                )
            else:
                with httpx.Client(timeout=self.timeout_seconds) as client:
                    response = client.post(
                        self.endpoint,
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json=payload,
                    )
        except httpx.TimeoutException as exc:
            raise ProviderError("Groq request timed out.") from exc
        except httpx.RequestError as exc:
            raise ProviderError("Groq request failed before a response was received.") from exc
        if response.status_code in {401, 403}:
            raise ProviderError("Groq rejected the API key. Check the key in generation settings.")
        if response.status_code == 429:
            raise ProviderError(
                "Groq rate limit or account quota reached; review Groq usage and limits."
            )
        if response.status_code == 400:
            raise ProviderError("Groq rejected the model or generation parameters.")
        if response.status_code == 404:
            raise ProviderError("The selected Groq model is unavailable to this account.")
        if response.is_error:
            raise ProviderError(f"Groq chat completion failed with HTTP {response.status_code}.")
        try:
            data = response.json()
            choices = data.get("choices")
            message = choices[0].get("message") if isinstance(choices, list) and choices else None
            content = message.get("content") if isinstance(message, dict) else None
            finish_reason = choices[0].get("finish_reason") if choices else None
            has_reasoning = bool(message.get("reasoning")) if isinstance(message, dict) else False
        except (ValueError, AttributeError, IndexError, TypeError) as exc:
            raise ProviderError("Groq returned a malformed chat-completion response.") from exc
        if not isinstance(content, str) or not content.strip():
            if finish_reason == "length":
                raise ProviderError(
                    "Groq exhausted the output-token limit before producing a final answer. "
                    "Increase the generation token limit."
                )
            if has_reasoning:
                raise ProviderError(
                    "Groq returned reasoning without a final answer. Enable final-answer-only "
                    "output or increase the output-token limit."
                )
            raise ProviderError("Groq returned an empty chat completion.")
        return content.strip()


class OpenAIProvider:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 90,
        client: httpx.Client | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.client = client

    def generate(
        self,
        model: str | None,
        system: str,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> str:
        if not self.api_key:
            raise ProviderError("OpenAI API key is missing. Add your API key in the generation settings.")
        if not model:
            raise ProviderError("Enter an OpenAI model ID (e.g. gpt-4o-mini).")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        try:
            if self.client:
                res = self.client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_seconds,
                )
            else:
                with httpx.Client(timeout=self.timeout_seconds) as client:
                    res = client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
        except httpx.TimeoutException as exc:
            raise ProviderError("OpenAI request timed out.") from exc
        except Exception as exc:
            raise ProviderError(f"OpenAI connection failed: {exc!s}") from exc

        if res.is_error:
            try:
                err_json = res.json().get("error", {})
                err_msg = err_json.get("message") or res.text[:200]
            except Exception:
                err_msg = res.text[:200]
            raise ProviderError(f"OpenAI API error ({res.status_code}): {err_msg}")

        data = res.json()
        choices = data.get("choices", [])
        if not choices:
            raise ProviderError("OpenAI returned an empty choice list.")
        content = choices[0].get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise ProviderError("OpenAI returned an empty response.")
        return content.strip()


class GeminiProvider:
    def __init__(
        self,
        api_key: str | None = None,
        timeout_seconds: float = 90,
        client: httpx.Client | None = None,
    ) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.client = client

    def generate(
        self,
        model: str | None,
        system: str,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> str:
        if not self.api_key:
            raise ProviderError("Gemini API key is missing. Add your API key in the generation settings.")
        if not model:
            raise ProviderError("Enter a Gemini model ID (e.g. gemini-2.0-flash).")
        clean_model = model.removeprefix("models/")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{clean_model}:generateContent?key={self.api_key}"
        payload = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            },
        }
        try:
            if self.client:
                res = self.client.post(url, json=payload, timeout=self.timeout_seconds)
            else:
                with httpx.Client(timeout=self.timeout_seconds) as client:
                    res = client.post(url, json=payload)
        except httpx.TimeoutException as exc:
            raise ProviderError("Gemini request timed out.") from exc
        except Exception as exc:
            raise ProviderError(f"Gemini connection failed: {exc!s}") from exc

        if res.is_error:
            try:
                err_msg = res.json().get("error", {}).get("message") or res.text[:200]
            except Exception:
                err_msg = res.text[:200]
            raise ProviderError(f"Gemini API error ({res.status_code}): {err_msg}")

        data = res.json()
        candidates = data.get("candidates", [])
        if not candidates:
            raise ProviderError("Gemini returned no candidates.")
        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts:
            raise ProviderError("Gemini returned no content parts.")
        text = parts[0].get("text", "")
        if not isinstance(text, str) or not text.strip():
            raise ProviderError("Gemini returned an empty response.")
        return text.strip()


class AnthropicProvider:
    endpoint = "https://api.anthropic.com/v1/messages"

    def __init__(
        self,
        api_key: str | None = None,
        timeout_seconds: float = 90,
        client: httpx.Client | None = None,
    ) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.client = client

    def generate(
        self,
        model: str | None,
        system: str,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> str:
        if not self.api_key:
            raise ProviderError("Anthropic API key is missing. Add your API key in the generation settings.")
        if not model:
            raise ProviderError("Enter an Anthropic model ID (e.g. claude-3-5-sonnet-20241022).")
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        try:
            if self.client:
                res = self.client.post(self.endpoint, headers=headers, json=payload, timeout=self.timeout_seconds)
            else:
                with httpx.Client(timeout=self.timeout_seconds) as client:
                    res = client.post(self.endpoint, headers=headers, json=payload)
        except httpx.TimeoutException as exc:
            raise ProviderError("Anthropic request timed out.") from exc
        except Exception as exc:
            raise ProviderError(f"Anthropic connection failed: {exc!s}") from exc

        if res.is_error:
            try:
                err_msg = res.json().get("error", {}).get("message") or res.text[:200]
            except Exception:
                err_msg = res.text[:200]
            raise ProviderError(f"Anthropic API error ({res.status_code}): {err_msg}")

        data = res.json()
        content_items = data.get("content", [])
        if not content_items:
            raise ProviderError("Anthropic returned an empty content list.")
        text = content_items[0].get("text", "")
        if not isinstance(text, str) or not text.strip():
            raise ProviderError("Anthropic returned an empty response.")
        return text.strip()
