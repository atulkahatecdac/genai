"""
Demo 11: Semantic chunking.

Splits a multi-topic block of text into sentences, embeds each sentence,
and starts a new chunk whenever the similarity between consecutive
sentences drops below a threshold - i.e. whenever the topic shifts -
instead of chunking by a fixed character count.

Setup:
    pip install openai python-dotenv
    Set OPENAI_API_KEY in the .env file in the parent GenAI folder (shared
    across modules).
"""

import math
import os
import re

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
SIMILARITY_THRESHOLD = 0.5

TEXT = (
    "The company was founded in 2014 by two former teachers who wanted to make tutoring more accessible. "
    "It grew from a single classroom in Austin to a platform used in over 40 states. "
    "The founders still review every new curriculum module personally before it ships. "
    "The flagship product is a mobile app that pairs students with tutors for 30-minute sessions. "
    "It supports scheduling, in-app video calls, and progress tracking for parents. "
    "Recent updates added AI-generated practice quizzes based on each session's topics. "
    "Customer support is available by chat from 8am to 10pm Eastern, seven days a week. "
    "Most tickets are resolved within 15 minutes during peak hours. "
    "A dedicated escalation team handles billing disputes within 24 hours."
)


def split_into_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def cosine_similarity(vec_a, vec_b):
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    return dot / (norm_a * norm_b)


def embed_sentences(client, sentences):
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=sentences)
    return [item.embedding for item in response.data]


def semantic_chunk(sentences, embeddings, threshold=SIMILARITY_THRESHOLD):
    chunks = [[sentences[0]]]
    for i in range(1, len(sentences)):
        similarity = cosine_similarity(embeddings[i - 1], embeddings[i])
        marker = "same topic" if similarity >= threshold else "TOPIC SHIFT -> new chunk"
        print(f"  Similarity({i - 1}, {i}) = {similarity:.3f}  [{marker}]")

        if similarity >= threshold:
            chunks[-1].append(sentences[i])
        else:
            chunks.append([sentences[i]])
    return chunks


def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    sentences = split_into_sentences(TEXT)
    print(f"Split into {len(sentences)} sentences\n")

    print("Consecutive-sentence similarities:")
    embeddings = embed_sentences(client, sentences)
    chunks = semantic_chunk(sentences, embeddings)

    print(f"\nResulting in {len(chunks)} semantic chunks:\n")
    for i, chunk in enumerate(chunks, start=1):
        print(f"Chunk {i}: {' '.join(chunk)}\n")


if __name__ == "__main__":
    main()
