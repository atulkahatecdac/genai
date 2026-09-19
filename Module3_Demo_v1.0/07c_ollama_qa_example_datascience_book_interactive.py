"""
Demo 7c: Interactive Q&A over a document example using Ollama (Qwen model).

Same textbook excerpt and context-stuffing pattern as demo 7b
(assets/datascience_book_excerpt.txt), but instead of a fixed list of
questions this version loops, letting you type your own questions about the
document until you decide to stop.

Setup:
    1. Install Ollama from https://ollama.com
    2. Pull a model:  ollama pull qwen2.5
    3. pip install ollama
"""

from pathlib import Path

import ollama

MODEL = "qwen2.5"
DOCUMENT_PATH = Path(__file__).parent / "assets" / "datascience_book_excerpt.txt"
QUIT_WORDS = ("quit", "exit", "q")


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
    print(f"Source document: {DOCUMENT_PATH.name}")
    print("\nAsk questions about the document (type 'quit' to exit).\n")

    while True:
        question = input("Your question: ").strip()
        if question.lower() in QUIT_WORDS:
            print("Goodbye!")
            break
        if not question:
            continue

        print(f"A: {ask(document_text, question)}\n")


if __name__ == "__main__":
    main()
