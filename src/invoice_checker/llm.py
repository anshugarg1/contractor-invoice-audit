"""LLM access.

The graph depends only on the InvoiceLLM protocol, so the tests and the rules-only
baseline can plug in a fake without touching the graph.
"""

from __future__ import annotations

import os
from typing import Protocol, TypeVar

from . import prompts
from .models import Claim, Classification, Document, ExtractedDocument, JudgeResult

T = TypeVar("T")

# Providers whose LangChain classes accept max_retries and timeout.
_RETRY_PROVIDERS = {"openai", "anthropic", "groq", "google_genai"}
LLM_MAX_RETRIES = 3
LLM_TIMEOUT_S = 60
LLM_OLLAMA_TIMEOUT_S = 180  # local models are slower and the first call loads the model onto the GPU

class InvoiceLLM(Protocol):
    model_name: str

    def classify(self, text: str) -> Classification: ...

    def extract(self, text: str) -> ExtractedDocument: ...

    def judge(self, claim: Claim, invoice: Document) -> JudgeResult: ...


def _chat_model(model_name: str):
    from langchain.chat_models import init_chat_model

    provider = model_name.split(":", 1)[0]
    kwargs: dict = {}
    if provider in _RETRY_PROVIDERS:
        kwargs.update(max_retries=LLM_MAX_RETRIES, timeout=LLM_TIMEOUT_S)
    elif provider == "ollama":  # ChatOllama has no timeout setting of its own; it goes to its HTTP client
        timeout = float(os.environ.get("LLM_OLLAMA_TIMEOUT_S", LLM_OLLAMA_TIMEOUT_S))
        kwargs["client_kwargs"] = {"timeout": timeout}
        
    if temperature := os.environ.get("LLM_TEMPERATURE", "").strip():
        kwargs["temperature"] = float(temperature)
    return init_chat_model(model_name, **kwargs)

class LangChainLLM:
    """Any chat model LangChain supports, selected with LLM_MODEL='provider:model'."""

    def __init__(self, model: str | None = None, fallbacks: list[str] | None = None) -> None:
        self.model_name = (model or os.environ.get("LLM_MODEL", "")).strip()
        if not self.model_name:
            raise RuntimeError(
                "LLM_MODEL is not set. Copy .env.example to .env and choose a model, "
                "for example LLM_MODEL=openai:gpt-4.1-mini"
            )
        if fallbacks is None:
            fallbacks = [m.strip() for m in os.environ.get("LLM_FALLBACK_MODELS", "").split(",") if m.strip()]
        self.fallback_names = fallbacks
        chats = [_chat_model(name) for name in [self.model_name, *self.fallback_names]]
          
        # Leave the method to LangChain unless overridden (e.g. LLM_STRUCTURED_METHOD=json_schema).
        method = os.environ.get("LLM_STRUCTURED_METHOD", "").strip() or None
        
        def structured(schema):
            primary, *backups = [
                  chat.with_structured_output(schema, method=method) if method else chat.with_structured_output(schema)
                  for chat in chats
            ]
            return primary.with_fallbacks(backups) if backups else primary

        self._classify = structured(Classification)
        self._extract = structured(ExtractedDocument)
        self._judge = structured(JudgeResult)

    def classify(self, text: str) -> Classification:
        return _require(self._classify.invoke(prompts.classify_messages(text)))

    def extract(self, text: str) -> ExtractedDocument:
        return _require(self._extract.invoke(prompts.extract_messages(text)))

    def judge(self, claim: Claim, invoice: Document) -> JudgeResult:
        return _require(self._judge.invoke(prompts.judge_messages(claim, invoice)))


def _require(result: T | None) -> T:
    if result is None:
        raise ValueError("the model returned no structured output")
    return result
