"""
Demo 6c: Interactive text summarization example using OpenAI.

Same textbook excerpt as demo 6b (assets/datascience_book_excerpt.txt), but
instead of producing one fixed summary this version loops, letting you type
how you'd like it summarized (e.g. "in 3 bullet points", "for a 10-year-old",
"focus on the data science cycle") until you decide to stop.

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
QUIT_WORDS = ("quit", "exit", "q")


def summarize(client: OpenAI, document_text: str, instruction: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You summarize the provided textbook excerpt according to the "
                    "user's instructions. Base your summary only on the excerpt."
                ),
            },
            {
                "role": "user",
                "content": f"Document:\n{document_text}\n\nInstruction: {instruction}",
            },
        ],
        temperature=0.3,
        max_tokens=400,
    )
    return response.choices[0].message.content or ""


def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    document_text = DOCUMENT_PATH.read_text(encoding="utf-8")

    print(f"Model: {MODEL}")
    print(f"Source document: {DOCUMENT_PATH.name} ({len(document_text)} chars)")
    print("\nTell me how you'd like the document summarized (type 'quit' to exit).")
    print('Examples: "in 3 bullet points", "for a 10-year-old", "focus on the data science cycle"\n')

    while True:
        instruction = input("How should I summarize it? ").strip()
        if instruction.lower() in QUIT_WORDS:
            print("Goodbye!")
            break
        if not instruction:
            continue

        summary = summarize(client, document_text, instruction)
        print(f"\nSummary:\n{summary}\n")


if __name__ == "__main__":
    main()
