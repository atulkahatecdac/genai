"""
Demo 7b: Q&A over a document example using Ollama (Qwen model), applied to a
real textbook.

Loads an excerpt from the free OpenStax textbook "Principles of Data Science"
(assets/datascience_book_excerpt.txt - Chapter 1, sections 1.1-1.2), stuffs it
into the prompt as context, and asks a locally-running Qwen model questions
about it. Same context-stuffing pattern as demo 7, just with a longer,
real-world source document.

Setup:
    1. Install Ollama from https://ollama.com
    2. Pull a model:  ollama pull qwen2.5
    3. pip install ollama
"""

from pathlib import Path

import ollama

MODEL = "qwen2.5"
DOCUMENT_PATH = Path(__file__).parent / "assets" / "datascience_book_excerpt.txt"

QUESTIONS = [
    "What are the five steps of the data science cycle?",
    "According to the Anaconda survey mentioned in the text, which step of the data science cycle takes up the most time?",
    "How did Walmart use data science to prepare for Hurricane Frances?",
]


def ask(document_text: str, question: str) -> str:
    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the user's question using ONLY the information in the "
                    "provided document. If the answer isn't in the document, say so."
                ),
            },
            {
                "role": "user",
                "content": f"Document:\n{document_text}\n\nQuestion: {question}",
            },
        ],
    )
    return response["message"]["content"]


def main():
    document_text = DOCUMENT_PATH.read_text(encoding="utf-8")
    print(f"Model: {MODEL}")
    print(f"Source document: {DOCUMENT_PATH.name}\n")

    for question in QUESTIONS:
        print(f"Q: {question}")
        print(f"A: {ask(document_text, question)}\n")


if __name__ == "__main__":
    main()
