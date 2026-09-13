"""
Demo 9: Text splitting for long documents with chunk overlap.

Splits a longer document into fixed-size chunks with LangChain's
CharacterTextSplitter, then highlights the overlapping text shared between
consecutive chunks so the effect of chunk_overlap is visible, not just
described. This demo makes no API calls.

Setup:
    pip install langchain-text-splitters
"""

from langchain_text_splitters import CharacterTextSplitter

DOCUMENT = (
    "Quarterly Report - Product Division. Sales grew 12% year over year, "
    "driven mainly by strong demand in the north region. The sales team "
    "attributes this to the new partner referral program launched in "
    "January, which contributed roughly a third of new deals this quarter. "
    "Customer support ticket volume rose 8%, largely due to onboarding "
    "questions from referral-driven signups. Average resolution time held "
    "steady at 6 hours despite the volume increase, thanks to the new "
    "triage system. Engineering shipped 4 major features this quarter, "
    "including the long-requested bulk export tool. Two features slipped "
    "to next quarter due to unresolved performance issues under high load."
)

CHUNK_SIZE = 150
CHUNK_OVERLAP = 40


def shared_overlap(end_of_prev, start_of_next):
    for length in range(min(len(end_of_prev), len(start_of_next)), 0, -1):
        if end_of_prev[-length:] == start_of_next[:length]:
            return end_of_prev[-length:]
    return ""


def main():
    splitter = CharacterTextSplitter(separator=" ", chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_text(DOCUMENT)

    print(f"Document length: {len(DOCUMENT)} chars -> split into {len(chunks)} chunks\n")
    for i, chunk in enumerate(chunks, start=1):
        print(f"Chunk {i} ({len(chunk)} chars): {chunk!r}")

    print("\n=== Overlap between consecutive chunks ===")
    for i in range(len(chunks) - 1):
        overlap = shared_overlap(chunks[i], chunks[i + 1])
        print(f"Chunk {i + 1} -> Chunk {i + 2}: {overlap!r}")


if __name__ == "__main__":
    main()
