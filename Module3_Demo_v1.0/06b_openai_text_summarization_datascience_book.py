"""
Demo 6b: Text summarization example using OpenAI, applied to a real textbook.

Reads an excerpt from the free OpenStax textbook "Principles of Data Science"
(assets/datascience_book_excerpt.txt - Chapter 1, sections 1.1-1.2) and asks
the model to produce a short summary. Same context-stuffing pattern as demo
6, just with a longer, real-world source document.

Setup:
    pip install openai python-dotenv
    Set OPENAI_API_KEY in a .env file (see .env.example).
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "gpt-4o-mini"
DOCUMENT_PATH = Path(__file__).parent / "assets" / "datascience_book_excerpt.txt"


def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    document_text = DOCUMENT_PATH.read_text(encoding="utf-8")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You summarize textbook excerpts into 5-7 concise bullet points.",
            },
            {"role": "user", "content": f"Summarize this document:\n\n{document_text}"},
        ],
        temperature=0.3,
        max_tokens=400,
    )

    print(f"Model: {MODEL}")
    print(f"Source document: {DOCUMENT_PATH.name} ({len(document_text)} chars)")
    print("\nSummary:")
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
