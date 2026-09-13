"""
Demo 11: Auto-prompt generation (meta-prompting).

Gives the model a plain-English task description and asks it to act as a
prompt engineer and design a well-structured prompt for that task, then
uses the generated prompt to actually perform the task.

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

TASK_DESCRIPTION = (
    "Classify a restaurant review as Positive, Negative, or Neutral, and "
    "extract the specific dish mentioned, if any."
)

META_PROMPT = f"""You are an expert prompt engineer. Design a clear, effective prompt that another AI could use to accomplish this task:

Task: {TASK_DESCRIPTION}

The prompt you write should include: a clear instruction, the expected output format, and one worked example. Output only the prompt text itself, nothing else."""

SAMPLE_REVIEW = "The truffle risotto was creamy and rich, but the service was painfully slow and our waiter seemed annoyed the whole time."


def call_model(client, prompt):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content


def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    print("=== Task description ===")
    print(TASK_DESCRIPTION)

    print("\n=== Step 1: auto-generate a prompt for this task ===")
    generated_prompt = call_model(client, META_PROMPT)
    print(generated_prompt)

    print("\n=== Step 2: use the generated prompt on a real review ===")
    final_prompt = f"{generated_prompt}\n\nReview: \"{SAMPLE_REVIEW}\""
    print(call_model(client, final_prompt))


if __name__ == "__main__":
    main()
