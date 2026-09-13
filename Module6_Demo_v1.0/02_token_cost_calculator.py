"""
Demo 2: Token costs - input, output, and total.

Sends one prompt, reads the actual token usage from the response, and
breaks the cost down into input tokens, output tokens, and the total -
since most providers price input and output tokens differently.

Setup:
    pip install openai python-dotenv
    Set OPENAI_API_KEY in the .env file in the parent GenAI folder (shared
    across modules).
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "gpt-4o-mini"

# Illustrative pricing, per 1M tokens - check https://openai.com/api/pricing
# before relying on this for real budgeting.
PRICE_PER_1M_INPUT_TOKENS = 0.15
PRICE_PER_1M_OUTPUT_TOKENS = 0.60

PROMPT = "Write a 4-sentence explanation of why the sky appears blue during the day."


def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT}],
    )

    usage = response.usage
    input_cost = usage.prompt_tokens / 1_000_000 * PRICE_PER_1M_INPUT_TOKENS
    output_cost = usage.completion_tokens / 1_000_000 * PRICE_PER_1M_OUTPUT_TOKENS
    total_cost = input_cost + output_cost

    print("Response:", response.choices[0].message.content)
    print(f"\nInput tokens:  {usage.prompt_tokens:>6}   cost: ${input_cost:.6f}")
    print(f"Output tokens: {usage.completion_tokens:>6}   cost: ${output_cost:.6f}")
    print(f"Total tokens:  {usage.total_tokens:>6}   cost: ${total_cost:.6f}")


if __name__ == "__main__":
    main()
