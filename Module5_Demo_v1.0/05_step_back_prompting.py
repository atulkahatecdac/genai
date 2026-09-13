"""
Demo 5: Step-Back prompting.

Answers a specific numeric question two ways: directly, and by first
"stepping back" to ask for the general principle/formula that applies, then
using that principle to solve the specific question.

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

SPECIFIC_QUESTION = (
    "A ball is thrown straight up at 20 m/s from the edge of a 15 m tall "
    "building. Using g = 10 m/s^2, roughly how many seconds until it hits "
    "the ground?"
)

DIRECT_PROMPT = f"{SPECIFIC_QUESTION}\n\nGive a numeric answer with brief reasoning."

STEP_BACK_QUESTION = (
    "What is the general physics principle or formula that relates an "
    "object's height over time when it is thrown upward under gravity, "
    "starting from some initial height and initial velocity?"
)

FINAL_PROMPT_TEMPLATE = """General principle:
{principle}

Now apply that principle to answer this specific question: {question}
Show the substitution and give a numeric answer."""


def call_model(client, prompt):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content


def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    print("=== Direct answer ===")
    print(call_model(client, DIRECT_PROMPT))

    print("\n=== Step back: ask for the general principle first ===")
    principle = call_model(client, STEP_BACK_QUESTION)
    print(principle)

    print("\n=== Apply the principle to the specific question ===")
    final_prompt = FINAL_PROMPT_TEMPLATE.format(principle=principle, question=SPECIFIC_QUESTION)
    print(call_model(client, final_prompt))


if __name__ == "__main__":
    main()
