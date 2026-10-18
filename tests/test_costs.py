import pytest
from invoice_checker.costs import cost_usd

def usage(input_tokens, output_tokens):
    return {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": input_tokens + output_tokens}


def test_cost_uses_input_and_output_prices():
    assert cost_usd({"openai/gpt-oss-120b": usage(1_000_000, 1_000_000)}) == pytest.approx(0.75)


def test_costs_of_primary_and_fallback_model_add_up():
    both = {"openai/gpt-oss-120b": usage(1_000_000, 0), "openai/gpt-oss-20b": usage(0, 1_000_000)}
    assert cost_usd(both) == pytest.approx(0.15 + 0.30)


def test_unknown_model_cost_is_unknown_not_zero():
    assert cost_usd({"some/new-model": usage(10, 10)}) is None


def test_no_llm_calls_cost_nothing():
    assert cost_usd({}) == 0.0


def test_local_model_has_a_small_known_cost():
    cost = cost_usd({"qwen3:4b": usage(1_000_000, 1_000_000)})
    assert cost is not None and 0 < cost < 1