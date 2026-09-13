"""
Demo 8: LangChain document loaders.

Loads a plain text file and a CSV file using LangChain's document loaders
and inspects the resulting Document objects (page_content + metadata).
This demo makes no API calls - loaders just read local files.

Setup:
    pip install langchain-community
"""

from pathlib import Path

from langchain_community.document_loaders import CSVLoader, TextLoader

ASSETS_DIR = Path(__file__).parent / "assets"


def main():
    print("=== TextLoader ===")
    text_loader = TextLoader(str(ASSETS_DIR / "sample_notes.txt"), encoding="utf-8")
    text_docs = text_loader.load()
    print(f"Loaded {len(text_docs)} document(s)")
    for doc in text_docs:
        print(f"Metadata: {doc.metadata}")
        print(f"Content preview: {doc.page_content[:150]!r}...")

    print("\n=== CSVLoader ===")
    csv_loader = CSVLoader(str(ASSETS_DIR / "sample_data.csv"))
    csv_docs = csv_loader.load()
    print(f"Loaded {len(csv_docs)} document(s), one per row")
    for doc in csv_docs:
        print(f"Metadata: {doc.metadata}")
        print(f"Content: {doc.page_content}")


if __name__ == "__main__":
    main()
