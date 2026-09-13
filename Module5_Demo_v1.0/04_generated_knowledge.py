"""
Demo 4: Generated Knowledge prompting.

Answers a commonsense question two ways: directly, and by first asking the
model to generate relevant background facts, then answering the question
using those facts as context. Surfacing the relevant knowledge first often
corrects answers that go wrong when the model jumps straight to a guess.

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

QUESTION = "True or False: Part of golf is trying to get a higher point total than others."

DIRECT_PROMPT = f"{QUESTION}\n\nAnswer with True or False and a one-sentence reason."

KNOWLEDGE_PROMPT = "List 3 short facts about how scoring works in golf."

KNOWLEDGE_AUGMENTED_PROMPT_TEMPLATE = """Facts about golf scoring:
{knowledge}

Using only the facts above, answer this question: {question}
Answer with True or False and a one-sentence reason."""


def call_model(client, prompt, temperature=0):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content


def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    print("=== Direct answer (no generated knowledge) ===")
    print(call_model(client, DIRECT_PROMPT))

    print("\n=== Step 1: generate relevant knowledge ===")
    knowledge = call_model(client, KNOWLEDGE_PROMPT)
    print(knowledge)

    print("\n=== Step 2: answer using the generated knowledge ===")
    final_prompt = KNOWLEDGE_AUGMENTED_PROMPT_TEMPLATE.format(knowledge=knowledge, question=QUESTION)
    print(call_model(client, final_prompt))


if __name__ == "__main__":
    main()
