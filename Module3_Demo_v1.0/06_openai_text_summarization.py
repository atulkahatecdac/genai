"""
Demo 6: Text summarization example using OpenAI.

Reads the sample document and asks the model to produce a short summary.

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
DOCUMENT_PATH = Path(__file__).parent / "assets" / "sample_document.txt"


def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    document_text = DOCUMENT_PATH.read_text()

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You summarize documents into 3-5 concise bullet points.",
            },
            {"role": "user", "content": f"Summarize this document:\n\n{document_text}"},
        ],
        temperature=0.3,
        max_tokens=300,
    )

    print(f"Model: {MODEL}")
    print(f"Source document: {DOCUMENT_PATH.name} ({len(document_text)} chars)")
    print("\nSummary:")
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
