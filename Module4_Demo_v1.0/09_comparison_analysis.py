"""
Demo 9: Prompt testing and evaluation - comparison analysis.

Runs two prompt variants for the same task (writing a product description)
against a shared set of inputs, then uses a separate model as a judge to
pick the better output for each input, and tallies the results.

Setup:
    pip install openai anthropic python-dotenv
    Set OPENAI_API_KEY and ANTHROPIC_API_KEY in the .env file in the parent
    GenAI folder (shared across modules).
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
import anthropic

load_dotenv()

CANDIDATE_MODEL = "gpt-4o-mini"
JUDGE_MODEL = "claude-opus-5"

PRODUCTS = [
    "a stainless steel reusable water bottle, 750ml, keeps drinks cold for 24 hours",
    "wireless earbuds with active noise cancellation and 30-hour battery life",
]

PROMPT_A = "Write a product description for: {product}"
PROMPT_B = (
    "Write a 2-sentence product description for: {product}. "
    "Sentence 1: the key benefit for the customer. Sentence 2: one standout feature. "
    "Avoid generic marketing filler like 'perfect for everyone'."
)


def generate(openai_client, prompt):
    response = openai_client.chat.completions.create(
        model=CANDIDATE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response.choices[0].message.content


def judge(anthropic_client, product, output_a, output_b):
    judge_prompt = f"""Two product descriptions were written for: "{product}"

Description A:
{output_a}

Description B:
{output_b}

Which is more concise and specific, with less generic filler? Answer with exactly one letter: A or B."""

    response = anthropic_client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=5,
        messages=[{"role": "user", "content": judge_prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text").strip()


def main():
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    wins = {"A": 0, "B": 0}
    for product in PRODUCTS:
        output_a = generate(openai_client, PROMPT_A.format(product=product))
        output_b = generate(openai_client, PROMPT_B.format(product=product))
        winner = judge(anthropic_client, product, output_a, output_b)
        wins[winner] = wins.get(winner, 0) + 1

        print(f"Product: {product}")
        print(f"  A: {output_a}")
        print(f"  B: {output_b}")
        print(f"  Judge picked: {winner}\n")

    print(f"Final tally: A={wins.get('A', 0)}  B={wins.get('B', 0)}")


if __name__ == "__main__":
    main()
