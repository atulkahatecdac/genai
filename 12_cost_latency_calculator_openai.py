"""
Demo 12: Approximate cost and latency calculation for OpenAI (gpt-4o-mini).

Runs a sample prompt against gpt-4o-mini, measures wall-clock latency, and
estimates the dollar cost of the call from the token usage returned by the
API and gpt-4o-mini's published per-token pricing.

NOTE: The per-token prices below are illustrative and may drift from
OpenAI's current published pricing - check https://openai.com/api/pricing
and update PRICE_PER_1M_INPUT_TOKENS / PRICE_PER_1M_OUTPUT_TOKENS before
relying on this for a real budget.

Setup:
    pip install openai python-dotenv
    Set OPENAI_API_KEY in a .env file (see .env.example).
"""

import os
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "gpt-4o-mini"

# Approximate published rates for gpt-4o-mini (USD per 1M tokens). Verify
# against OpenAI's pricing page before using these numbers for real cost
# projections.
PRICE_PER_1M_INPUT_TOKENS = 0.15
PRICE_PER_1M_OUTPUT_TOKENS = 0.60

PROMPT = "Explain the difference between supervised and unsupervised learning in 4 sentences."


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    input_cost = (input_tokens / 1_000_000) * PRICE_PER_1M_INPUT_TOKENS
    output_cost = (output_tokens / 1_000_000) * PRICE_PER_1M_OUTPUT_TOKENS
    return input_cost + output_cost


def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    start = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=300,
    )
    elapsed = time.perf_counter() - start

    usage = response.usage
    cost = estimate_cost(usage.prompt_tokens, usage.completion_tokens)

    print(f"Model: {MODEL}")
    print(f"Prompt: {PROMPT}\n")
    print(f"Response:\n{response.choices[0].message.content}\n")
    print("--- Metrics (approximate) ---")
    print(f"Latency:          {elapsed:.2f} s")
    print(f"Input tokens:     {usage.prompt_tokens}")
    print(f"Output tokens:    {usage.completion_tokens}")
    print(f"Total tokens:     {usage.total_tokens}")
    print(f"Estimated cost:   ${cost:.6f}")
    print(
        f"Throughput:       {usage.completion_tokens / elapsed:.1f} output tokens/sec (approx, "
        f"includes network latency)"
    )


if __name__ == "__main__":
    main()
