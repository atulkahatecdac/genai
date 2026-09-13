"""
Demo 14: Token usage measurement using tiktoken.

Estimates the prompt token count with tiktoken before ever calling the API
(following OpenAI's per-message overhead formula for chat models), then
compares that estimate against the token count the API actually reports.

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
encoding = tiktoken.encoding_for_model(MODEL)

MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What are three benefits of regular exercise?"},
]

TOKENS_PER_MESSAGE = 3  # per-message overhead (role + separators), per OpenAI's cookbook formula
TOKENS_PER_REPLY_PRIMING = 3  # tokens added to prime the assistant's reply


def count_prompt_tokens(messages):
    total = 0
    for message in messages:
        total += TOKENS_PER_MESSAGE
        for value in message.values():
            total += len(encoding.encode(value))
    total += TOKENS_PER_REPLY_PRIMING
    return total


def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    estimated_tokens = count_prompt_tokens(MESSAGES)
    print(f"Estimated prompt tokens (tiktoken, before sending): {estimated_tokens}")

    response = client.chat.completions.create(model=MODEL, messages=MESSAGES)
    actual_tokens = response.usage.prompt_tokens

    print(f"Actual prompt tokens (reported by the API):        {actual_tokens}")
    print(f"Difference: {abs(estimated_tokens - actual_tokens)} tokens")

    print(f"\nResponse: {response.choices[0].message.content}")
    print(f"Completion tokens: {response.usage.completion_tokens}")


if __name__ == "__main__":
    main()
