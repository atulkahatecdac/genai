"""
Demo 4: Prompting techniques - zero-shot, one-shot, few-shot.

Classifies movie review sentiment three ways to show how the number of
examples in the prompt affects output consistency and format.

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

NEW_REVIEW = "It started strong but the last act felt rushed and left half the subplots unresolved."

ZERO_SHOT = f"""Classify the sentiment of this movie review as Positive, Negative, or Mixed.

Review: "{NEW_REVIEW}"
Sentiment:"""

ONE_SHOT = f"""Classify the sentiment of each movie review as Positive, Negative, or Mixed.

Review: "A visual masterpiece with a story that never lets up."
Sentiment: Positive

Review: "{NEW_REVIEW}"
Sentiment:"""

FEW_SHOT = f"""Classify the sentiment of each movie review as Positive, Negative, or Mixed.

Review: "A visual masterpiece with a story that never lets up."
Sentiment: Positive

Review: "Two hours I'll never get back. Flat acting, no plot."
Sentiment: Negative

Review: "Great cinematography, but the pacing dragged in the middle third."
Sentiment: Mixed

Review: "{NEW_REVIEW}"
Sentiment:"""


def call_model(client, prompt):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=10,
    )
    return response.choices[0].message.content.strip()


def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    for name, prompt in [("Zero-shot", ZERO_SHOT), ("One-shot", ONE_SHOT), ("Few-shot", FEW_SHOT)]:
        print(f"{name}: {call_model(client, prompt)}")


if __name__ == "__main__":
    main()
