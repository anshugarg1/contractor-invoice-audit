"""LLM gateway behaviour without network calls: fallback to the backup model."""

import pytest
from langchain_core.runnables import RunnableLambda

from invoice_checker import llm
from invoice_checker.models import Classification

ANSWER = Classification(doc_type="invoice", trade="Sanitär")

class FakeChat:
    """Stands in for a LangChain chat model: returns `answer`, or raises it if it is an exception."""

    def __init__(self, answer):
        self.answer = answer
        self.calls = 0

    def with_structured_output(self, schema, **kwargs):
        def respond(messages):
            self.calls += 1
            if isinstance(self.answer, Exception):
                raise self.answer
            return self.answer

        return RunnableLambda(respond)


def gateway(monkeypatch, primary, *backups):
    monkeypatch.delenv("LLM_FALLBACK_MODELS", raising=False)
    monkeypatch.delenv("LLM_STRUCTURED_METHOD", raising=False)
    chats = {"primary:model": primary} | {f"backup{i}:model": chat for i, chat in enumerate(backups, 1)}
    monkeypatch.setattr(llm, "_chat_model", lambda name: chats[name])
    return llm.LangChainLLM("primary:model", fallbacks=[name for name in chats if name != "primary:model"])


def test_primary_answers_and_backup_is_not_called(monkeypatch):
    primary, backup = FakeChat(ANSWER), FakeChat(ANSWER)
    assert gateway(monkeypatch, primary, backup).classify("Rechnung") == ANSWER
    assert (primary.calls, backup.calls) == (1, 0)


def test_backup_answers_when_primary_times_out(monkeypatch):
    primary, backup = FakeChat(TimeoutError()), FakeChat(ANSWER)
    assert gateway(monkeypatch, primary, backup).classify("Rechnung") == ANSWER
    assert (primary.calls, backup.calls) == (1, 1)


def test_error_surfaces_when_both_models_fail(monkeypatch):
    primary, backup = FakeChat(TimeoutError()), FakeChat(ConnectionError())
    with pytest.raises(TimeoutError):
        gateway(monkeypatch, primary, backup).classify("Rechnung")


def test_fallbacks_are_tried_in_order_until_one_answers(monkeypatch):
    primary, second, third = FakeChat(TimeoutError()), FakeChat(ConnectionError()), FakeChat(ANSWER)
    assert gateway(monkeypatch, primary, second, third).classify("Rechnung") == ANSWER
    assert (primary.calls, second.calls, third.calls) == (1, 1, 1)


def test_fallbacks_are_read_from_env_in_order(monkeypatch):
    monkeypatch.setattr(llm, "_chat_model", lambda name: FakeChat(ANSWER))
    monkeypatch.setenv("LLM_FALLBACK_MODELS", "groq:openai/gpt-oss-20b, ,ollama:qwen3")
    assert llm.LangChainLLM("primary:model").fallback_names == ["groq:openai/gpt-oss-20b", "ollama:qwen3"]

