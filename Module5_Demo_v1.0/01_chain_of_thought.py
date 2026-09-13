"""
Demo 1: Chain-of-Thought (CoT) prompting.

Answers the same multi-step word problem two ways: asking directly for the
final number, and asking the model to reason step by step first. CoT tends
to be more reliable on multi-step problems because it gives the model room
to work through intermediate steps instead of guessing the final answer.

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

PROBLEM = (
    "A store had 120 apples. They sold 35% of them in the morning and then "
    "40 more in the afternoon. How many apples are left?"
)

DIRECT_PROMPT = f"{PROBLEM}\n\nAnswer with just the number, nothing else."

COT_PROMPT = f"{PROBLEM}\n\nLet's think step by step. Show your reasoning, then finish with a line 'Answer: <number>'."


def call_model(client, prompt):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content


def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    print("=== Direct prompt (no reasoning) ===")
    print(call_model(client, DIRECT_PROMPT))

    print("\n=== Chain-of-thought prompt ===")
    print(call_model(client, COT_PROMPT))


if __name__ == "__main__":
    main()
