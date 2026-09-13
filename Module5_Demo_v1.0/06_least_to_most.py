"""
Demo 6: Least-to-Most prompting.

Breaks a multi-part problem into an ordered list of smaller subproblems,
then solves them one at a time in order, feeding each prior answer forward
as context for the next step, finishing with the overall answer.

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

PROBLEM = (
    "A train travels 60 km in 1.5 hours at a steady speed. It then speeds "
    "up by 20% for the next 2 hours. How far does the train travel in total?"
)

DECOMPOSE_PROMPT = f"""Break the following problem into an ordered list of smaller subproblems that must be solved in sequence to reach the final answer. Do not solve them yet.

Problem: {PROBLEM}

Output only a numbered list, one subproblem per line."""


def call_model(client, prompt):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content


def parse_numbered_list(text):
    lines = re.findall(r"^\s*\d+[.)]\s*(.+)$", text, flags=re.MULTILINE)
    return lines


def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    print("=== Step 1: decompose into subproblems ===")
    decomposition = call_model(client, DECOMPOSE_PROMPT)
    print(decomposition)
    subproblems = parse_numbered_list(decomposition)

    print("\n=== Step 2: solve subproblems in order ===")
    solved_so_far = ""
    for i, subproblem in enumerate(subproblems, start=1):
        prompt = (
            f"Original problem: {PROBLEM}\n\n"
            f"Solved so far:\n{solved_so_far if solved_so_far else '(nothing yet)'}\n\n"
            f"Now solve this subproblem: {subproblem}\n"
            "Give a short, direct answer."
        )
        answer = call_model(client, prompt)
        print(f"\nSubproblem {i}: {subproblem}\nAnswer: {answer}")
        solved_so_far += f"{i}. {subproblem} -> {answer}\n"

    print("\n=== Final answer ===")
    final_prompt = (
        f"Original problem: {PROBLEM}\n\n"
        f"Subproblems and their answers:\n{solved_so_far}\n"
        "Using the above, give the final answer to the original problem."
    )
    print(call_model(client, final_prompt))


if __name__ == "__main__":
    main()
