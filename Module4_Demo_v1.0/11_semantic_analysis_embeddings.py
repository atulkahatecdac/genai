"""
Demo 11: Prompt testing methods - semantic analysis using embeddings.

Compares an ideal reference answer to the model's actual answer using
embedding cosine similarity, so near-miss wording doesn't get flagged as a
failure the way a strict keyword match would.

Setup:
    pip install openai python-dotenv
    Set OPENAI_API_KEY in the .env file in the parent GenAI folder (shared
    across modules).
"""

import math
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

CHAT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
SIMILARITY_THRESHOLD = 0.75

TEST_CASES = [
    {
        "question": "What is the capital of Australia?",
        "reference_answer": "The capital of Australia is Canberra.",
    },
    {
        "question": "Why is the sky blue?",
        "reference_answer": (
            "The sky looks blue because air molecules scatter shorter blue "
            "wavelengths of sunlight more than other colors."
        ),
    },
]


def cosine_similarity(vec_a, vec_b):
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    return dot / (norm_a * norm_b)


def embed(client, text):
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def ask(client, question):
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": question}],
        temperature=0,
    )
    return response.choices[0].message.content


def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    for case in TEST_CASES:
        actual_answer = ask(client, case["question"])
        similarity = cosine_similarity(
            embed(client, case["reference_answer"]),
            embed(client, actual_answer),
        )
        status = "PASS" if similarity >= SIMILARITY_THRESHOLD else "FAIL"

        print(f"Question: {case['question']}")
        print(f"  Reference: {case['reference_answer']}")
        print(f"  Actual:    {actual_answer}")
        print(f"  Similarity: {similarity:.3f} -> [{status}]\n")


if __name__ == "__main__":
    main()
