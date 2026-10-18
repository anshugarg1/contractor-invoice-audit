
"""Token cost per model, so every eval run reports what it cost.

  Prices are USD per 1 million tokens (input, output), from the provider's pricing page.
  A model missing here gets cost None (unknown), never 0, so a report can't look cheaper than it was."""

from __future__ import annotations
from time import time

# Groq prices, checked 2026-10-18 on console.groq.com/docs/models.
PRICES_PER_MILLION: dict[str, tuple[float, float]] = {
    "openai/gpt-oss-120b": (0.15, 0.60),
    "openai/gpt-oss-20b": (0.075, 0.30),
    # Local models via Ollama: no API fee. Estimated laptop electricity, not a bill:
    # ~100 W under load, 0.35 per kWh; output speed measured 2026-10-18 on an RTX 5070 Laptop GPU
    # (qwen3:4b 81.5 tokens/s, qwen3:8b 19.9 tokens/s), input speed assumed until measured.
    #time for 1M tokens (hours)  =  1,000,000 ÷ speed (tokens/s) ÷ 3,600
    #energy (kWh)                =  time (hours) × power (kW)
    #cost per 1M tokens          =  energy (kWh) × electricity price (per kWh)
    #As one formula
    #cost per 1M tokens  =  (power in watts × price per kWh) ÷ (3.6 × speed in tokens/s)
    "qwen3:4b": (0.01, 0.13),
    "qwen3:8b": (0.02, 0.18)
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


