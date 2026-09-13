"""
Demo 1: Context window limits.

Sends inputs of increasing size to a model and tracks the token count
against its documented context window, including deliberately exceeding the
limit once to see how the API responds. The oversized request is rejected
before any generation happens, so it costs nothing.

Setup:
    pip install openai tiktoken python-dotenv
    Set OPENAI_API_KEY in the .env file in the parent GenAI folder (shared
    across modules).
"""

import os

import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "gpt-4o-mini"
CONTEXT_WINDOW_TOKENS = 128_000  # documented context window for gpt-4o-mini

encoding = tiktoken.encoding_for_model(MODEL)


def count_tokens(text):
    return len(encoding.encode(text))


def make_text(approx_tokens):
    # "hello " encodes to ~1 token, so repeating it is a simple way to hit a target count.
    return "hello " * approx_tokens


def try_request(client, text):
    token_count = count_tokens(text)
    pct_of_window = token_count / CONTEXT_WINDOW_TOKENS * 100
    print(f"Input tokens: {token_count:,} ({pct_of_window:.1f}% of the {CONTEXT_WINDOW_TOKENS:,} token window)")

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": text + "\n\nReply with just: OK"}],
            max_tokens=5,
        )
        print(f"Result: succeeded, model replied {response.choices[0].message.content!r}")
    except Exception as exc:
        print(f"Result: failed - {exc}")


def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    print("=== Small input ===")
    try_request(client, make_text(50))

    print("\n=== Medium input ===")
    try_request(client, make_text(5_000))

    print("\n=== Input that exceeds the context window ===")
    try_request(client, make_text(CONTEXT_WINDOW_TOKENS + 5_000))


if __name__ == "__main__":
    main()
