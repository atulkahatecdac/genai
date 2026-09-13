"""
Demo 10: Recursive chunking.

Compares LangChain's RecursiveCharacterTextSplitter (which tries paragraph
breaks, then line breaks, then spaces, before ever cutting mid-word) against
a naive fixed-size splitter that just slices every N characters regardless
of where that lands. This demo makes no API calls.

Setup:
    pip install langchain-text-splitters
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCUMENT = """Section 1: Getting Started
Install the CLI, then run `init` inside your project folder. This creates a config file with sensible defaults for most setups.

Section 2: Configuration
Most settings can be overridden with environment variables. See the reference table below for the full list of supported options.

Section 3: Troubleshooting
If the CLI fails to start, check that your config file is valid JSON. A common mistake is a trailing comma after the last field."""

CHUNK_SIZE = 120
CHUNK_OVERLAP = 0


def naive_chunk(text, chunk_size):
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def main():
    print("=== Naive fixed-size chunking ===")
    naive_chunks = naive_chunk(DOCUMENT, CHUNK_SIZE)
    for i, chunk in enumerate(naive_chunks, start=1):
        print(f"Chunk {i}: {chunk!r}")

    print("\n=== RecursiveCharacterTextSplitter ===")
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    recursive_chunks = splitter.split_text(DOCUMENT)
    for i, chunk in enumerate(recursive_chunks, start=1):
        print(f"Chunk {i}: {chunk!r}")

    print("\nNaive chunking cuts wherever the character count lands, sometimes mid-word or mid-sentence.")
    print("Recursive chunking prefers to break at paragraph or line boundaries first, keeping each chunk coherent.")


if __name__ == "__main__":
    main()
