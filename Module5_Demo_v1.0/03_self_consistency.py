"""
Demo 3: Self-consistency prompting.

Runs the same chain-of-thought prompt several times at a higher temperature
(so reasoning paths vary), extracts the final answer from each run, and
takes a majority vote instead of trusting any single sample.

Setup:
    pip install openai python-dotenv
    Set OPENAI_API_KEY in the .env file in the parent GenAI folder (shared
    across modules).
"""

import os
import re
from collections import Counter

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "gpt-4o-mini"
NUM_SAMPLES = 5

PROBLEM = (
    "Sarah has 3 boxes with 8 pencils each. She gives away 5 pencils and "
    "then buys 2 more boxes of 8 pencils. How many pencils does she have now?"
)

COT_PROMPT = f"{PROBLEM}\n\nThink step by step, then finish with a line 'Answer: <number>'."


def sample_answer(client):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": COT_PROMPT}],
        temperature=0.8,
    )
    text = response.choices[0].message.content
    match = re.search(r"Answer:\s*(-?\d+)", text)
    return match.group(1) if match else None


def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    answers = []
    for i in range(NUM_SAMPLES):
        answer = sample_answer(client)
        answers.append(answer)
        print(f"Sample {i + 1}: {answer}")

    valid_answers = [a for a in answers if a is not None]
    tally = Counter(valid_answers)
    majority_answer, votes = tally.most_common(1)[0]

    print(f"\nVote tally: {dict(tally)}")
    print(f"Majority answer: {majority_answer} ({votes}/{len(valid_answers)} votes)")


if __name__ == "__main__":
    main()
