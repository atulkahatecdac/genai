"""
Demo 7: Q&A over a document example using Ollama (Qwen model).

Loads the sample document (assets/sample_document.txt), stuffs it into the
prompt as context, and asks a locally-running Qwen model questions about it.
This is a simple "context stuffing" Q&A pattern - fine for small documents;
larger documents would need chunking + retrieval (RAG), covered separately.

Setup:
    1. Install Ollama from https://ollama.com
    2. Pull a model:  ollama pull qwen2.5
    3. pip install ollama
"""

from pathlib import Path

import ollama

MODEL = "qwen2.5"
DOCUMENT_PATH = Path(__file__).parent / "assets" / "sample_document.txt"

QUESTIONS = [
    "Who leads Project Nimbus?",
    "What is the total approved budget?",
    "What happens in Phase 2 of the timeline?",
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
    document_text = DOCUMENT_PATH.read_text()
    print(f"Model: {MODEL}")
    print(f"Source document: {DOCUMENT_PATH.name}\n")

    for question in QUESTIONS:
        print(f"Q: {question}")
        print(f"A: {ask(document_text, question)}\n")


if __name__ == "__main__":
    main()
