
"""Token cost per model, so every eval run reports what it cost.

  Prices are USD per 1 million tokens (input, output), from the provider's pricing page.
  A model missing here gets cost None (unknown), never 0, so a report can't look cheaper than it was."""

from __future__ import annotations

# Groq prices, checked 2026-10-18 on console.groq.com/docs/models.
PRICES_PER_MILLION: dict[str, tuple[float, float]] = {
    "openai/gpt-oss-120b": (0.15, 0.60),
    "openai/gpt-oss-20b": (0.075, 0.30),
}

def cost_usd(usage: dict[str, dict]) -> float | None:
    """Cost of the token usage per model name, as LangChain's UsageMetadataCallbackHandler collects it."""
    total = 0.0
    for model, tokens in usage.items():
        if model not in PRICES_PER_MILLION:
            return None
        price_in, price_out = PRICES_PER_MILLION[model]
        total += (tokens["input_tokens"] * price_in + tokens["output_tokens"] * price_out) / 1_000_000
    return total


