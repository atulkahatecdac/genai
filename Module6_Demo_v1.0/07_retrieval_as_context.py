"""
Demo 7: Retrieval as context (mini-RAG).

Embeds a small in-memory knowledge base, retrieves the snippets most
relevant to a user's question via cosine similarity, and injects only those
snippets into the prompt as context - rather than sending the whole
knowledge base every time.

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
TOP_K = 2

KNOWLEDGE_BASE = [
    "Returns are accepted within 30 days of purchase with a valid receipt.",
    "Standard shipping takes 5-7 business days; express shipping takes 1-2 business days.",
    "The Pro plan includes 100,000 API calls per month; the Free plan includes 1,000.",
    "Customer support is available Monday-Friday, 9am-6pm Eastern Time.",
    "User data is retained for 90 days after account deletion, then permanently erased.",
    "API rate limits are 60 requests per minute on the Free plan, 600 on the Pro plan.",
]

QUESTION = "How long do you keep my data if I delete my account?"


def cosine_similarity(vec_a, vec_b):
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    return dot / (norm_a * norm_b)


def embed(client, text):
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def retrieve(client, question, top_k=TOP_K):
    question_embedding = embed(client, question)
    scored = [(cosine_similarity(question_embedding, embed(client, doc)), doc) for doc in KNOWLEDGE_BASE]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored[:top_k]


def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    retrieved = retrieve(client, QUESTION)
    print("Retrieved snippets:")
    for score, doc in retrieved:
        print(f"  ({score:.3f}) {doc}")

    context = "\n".join(doc for _, doc in retrieved)
    prompt = (
        f"Using only the context below, answer the question. If the answer isn't in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {QUESTION}"
    )

    response = client.chat.completions.create(model=CHAT_MODEL, messages=[{"role": "user", "content": prompt}])
    print(f"\nQuestion: {QUESTION}")
    print(f"Answer: {response.choices[0].message.content}")


if __name__ == "__main__":
    main()
