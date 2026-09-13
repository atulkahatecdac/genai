"""
Demo 9: Code generation example using Anthropic (Claude).

Asks Claude to generate a small, self-contained Python utility function from
a natural-language spec, then writes the generated code to disk so it can be
inspected or executed.

Setup:
    pip install anthropic python-dotenv
    Set ANTHROPIC_API_KEY in a .env file (see .env.example).
"""

import os
from pathlib import Path

from dotenv import load_dotenv
import anthropic

load_dotenv()

MODEL = "claude-opus-5"
OUTPUT_PATH = Path(__file__).parent / "assets" / "generated_code.py"

TASK_SPEC = (
    "Write a Python function `is_palindrome(text: str) -> bool` that returns "
    "True if the input string is a palindrome, ignoring case, spaces, and "
    "punctuation. Include a short docstring and 3 example calls in a "
    "`if __name__ == '__main__':` block. Return ONLY the code, no explanation, "
    "no markdown code fences."
)


def main():
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system="You are an expert Python developer who writes clean, well-documented code.",
        messages=[{"role": "user", "content": TASK_SPEC}],
    )

    generated_code = "".join(block.text for block in response.content if block.type == "text")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(generated_code)

    print(f"Model: {MODEL}")
    print("Generated code:\n")
    print(generated_code)
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
