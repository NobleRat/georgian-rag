"""Approximate OpenAI prices (USD per 1M tokens, Standard tier, prompts <=272K tokens).

Source: https://developers.openai.com/api/docs/pricing, checked 2026-11-01.
These numbers are used only for estimates shown in the app; the authoritative
figures are on the OpenAI billing/usage dashboard. Update this table if prices change.
"""
from __future__ import annotations

PRICES_CHECKED_ON = "2026-11-01"

# model -> (input, cached_input, output)
CHAT_PRICES: dict[str, tuple[float, float, float]] = {
    "gpt-6-astra": (10.00, 1.00, 50.00),
    "gpt-6.1-sol": (2.00, 0.10, 10.00),
    "gpt-6.1-terra": (1.00, 0.05, 5.00),
    "gpt-6-sol": (2.00, 0.20, 10.00),
    "gpt-6-luna": (0.10, 0.01, 0.50),
    "gpt-5.6-sol": (4.00, 0.40, 20.00),
    "gpt-5.6-terra": (2.00, 0.20, 12.00),
    "gpt-5.6-luna": (0.20, 0.02, 1.20),
    "gpt-5.5": (5.00, 0.50, 30.00),
    "gpt-5.4": (2.50, 0.25, 15.00),
    "gpt-5.4-mini": (0.75, 0.075, 4.50),
    "gpt-5.4-nano": (0.20, 0.02, 1.25),
    "gpt-5.2": (1.75, 0.175, 14.00),
    "gpt-5.1": (1.25, 0.125, 10.00),
    "gpt-5": (1.25, 0.125, 10.00),
    "gpt-5-mini": (0.25, 0.025, 2.00),
    "gpt-4.1": (2.00, 0.50, 8.00),
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "gpt-4o": (2.50, 1.25, 10.00),
    "gpt-4o-mini": (0.15, 0.075, 0.60),
}

EMBEDDING_PRICES: dict[str, float] = {
    "text-embedding-3-large": 0.13,
    "text-embedding-3-small": 0.02,
}


def chat_cost(model: str, input_tokens: int, cached_tokens: int, output_tokens: int) -> float | None:
    price = CHAT_PRICES.get(model)
    if price is None:
        return None
    inp, cached, out = price
    uncached = max(input_tokens - cached_tokens, 0)
    return (uncached * inp + cached_tokens * cached + output_tokens * out) / 1_000_000


def embedding_cost(model: str, tokens: int) -> float | None:
    price = EMBEDDING_PRICES.get(model)
    return None if price is None else tokens * price / 1_000_000
