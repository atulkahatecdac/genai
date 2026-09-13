"""
Demo 2: Tree-of-Thought (ToT) prompting.

Solves a "Game of 24" puzzle (combine four numbers with +, -, *, / to reach
24) by generating several independent candidate solutions (branches),
evaluating each one, and picking the best. This is a simplified,
single-level tree: generate branches -> evaluate -> select, rather than the
full recursive search from the ToT paper.

Setup:
    pip install openai python-dotenv
    Set OPENAI_API_KEY in the .env file in the parent GenAI folder (shared
    across modules).
"""

import os
import re

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "gpt-4o-mini"
NUMBERS = [4, 9, 10, 13]
TARGET = 24
NUM_BRANCHES = 3

BRANCH_PROMPT = f"""Using the numbers {NUMBERS} exactly once each, and any of +, -, *, /, and parentheses, write an arithmetic expression that evaluates to {TARGET}.

Try a couple of combinations in your head, then on the final line output exactly:
Expression: <your expression>"""


def generate_branch(client):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": BRANCH_PROMPT}],
        temperature=1.0,
    )
    text = response.choices[0].message.content
    match = re.search(r"Expression:\s*(.+)", text)
    return match.group(1).strip() if match else None


def evaluate_branch(expression):
    if not expression or not re.fullmatch(r"[0-9+\-*/(). ]+", expression):
        return False, "invalid characters in expression"

    used_numbers = sorted(int(n) for n in re.findall(r"\d+", expression))
    if used_numbers != sorted(NUMBERS):
        return False, f"uses {used_numbers}, expected {sorted(NUMBERS)}"

    try:
        result = eval(expression)  # noqa: S307 - input is regex-validated above
    except ZeroDivisionError:
        return False, "division by zero"

    if abs(result - TARGET) < 1e-6:
        return True, f"evaluates to {result}"
    return False, f"evaluates to {result}, not {TARGET}"


def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    branches = [generate_branch(client) for _ in range(NUM_BRANCHES)]

    print(f"Generated {NUM_BRANCHES} branches:\n")
    winner = None
    for i, expression in enumerate(branches, start=1):
        is_valid, reason = evaluate_branch(expression)
        status = "VALID" if is_valid else "invalid"
        print(f"Branch {i}: {expression!r} -> [{status}] {reason}")
        if is_valid and winner is None:
            winner = expression

    print("\nSelected solution:", winner or "none of the branches solved it - try running again")


if __name__ == "__main__":
    main()
