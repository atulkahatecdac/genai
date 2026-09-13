"""
Demo 1: The four core elements of a prompt - Instruction, Context, Input
data, Output indicator.

Builds one prompt by assembling these four elements as separate pieces, then
sends it to an LLM so you can see how each piece shapes the final prompt text
and the response.

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

INSTRUCTION = "Classify the urgency of the following customer support ticket."
CONTEXT = (
    "You work for a SaaS company. HIGH urgency means the customer cannot use "
    "the product at all or data is at risk. MEDIUM means a feature is broken "
    "but a workaround exists. LOW means it's a question or a minor cosmetic issue."
)
INPUT_DATA = (
    "Ticket: \"Ever since today's update I can't export any reports, and my "
    "board meeting where I need those numbers is tomorrow morning.\""
)
OUTPUT_INDICATOR = "Respond with exactly one word: LOW, MEDIUM, or HIGH."


def build_prompt():
    return "\n\n".join([INSTRUCTION, CONTEXT, INPUT_DATA, OUTPUT_INDICATOR])


def main():
    prompt = build_prompt()

    print("Instruction:      ", INSTRUCTION)
    print("Context:          ", CONTEXT)
    print("Input data:       ", INPUT_DATA)
    print("Output indicator: ", OUTPUT_INDICATOR)
    print("\nAssembled prompt:\n" + "-" * 40)
    print(prompt)
    print("-" * 40)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    print(f"\nModel: {MODEL}")
    print("Response:", response.choices[0].message.content)


if __name__ == "__main__":
    main()
