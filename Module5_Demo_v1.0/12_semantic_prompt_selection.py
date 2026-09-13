"""
Demo 12: Semantic search for prompt selection.

Keeps a small library of prompt templates, each with a short description.
Embeds the descriptions once, then embeds an incoming user query and picks
the template whose description is most semantically similar (cosine
similarity), instead of relying on brittle keyword matching or manual
if/else routing.

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

PROMPT_LIBRARY = [
    {
        "name": "summarize",
        "description": "Summarize a long piece of text into key bullet points.",
        "template": "Summarize the following text in 3 bullet points:\n\n{input}",
    },
    {
        "name": "translate",
        "description": "Translate text from English into another language.",
        "template": "Translate the following text to French:\n\n{input}",
    },
    {
        "name": "sentiment",
        "description": "Classify the sentiment of a piece of text as positive, negative, or neutral.",
        "template": "Classify the sentiment of this text as Positive, Negative, or Neutral:\n\n{input}",
    },
    {
        "name": "code_explain",
        "description": "Explain what a piece of code does in plain, simple English.",
        "template": "Explain what the following code does in simple terms:\n\n{input}",
    },
]

USER_QUERY = "Can you tell me what this Python function is doing?"
SAMPLE_INPUT = "def is_even(n):\n    return n % 2 == 0"


def cosine_similarity(vec_a, vec_b):
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    return dot / (norm_a * norm_b)


def embed(client, text):
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def select_template(client, query):
    query_embedding = embed(client, query)
    scored = []
    for entry in PROMPT_LIBRARY:
        similarity = cosine_similarity(query_embedding, embed(client, entry["description"]))
        scored.append((similarity, entry))
    scored.sort(key=lambda pair: pair[0], reverse=True)

    print("Similarity scores:")
    for similarity, entry in scored:
        print(f"  {entry['name']:<14} {similarity:.3f}")

    return scored[0][1]


def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    print(f"User query: {USER_QUERY}\n")
    best_template = select_template(client, USER_QUERY)
    print(f"\nSelected template: {best_template['name']}")

    rendered_prompt = best_template["template"].format(input=SAMPLE_INPUT)
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": rendered_prompt}],
    )
    print(f"\nRendered prompt:\n{rendered_prompt}")
    print(f"\nModel output:\n{response.choices[0].message.content}")


if __name__ == "__main__":
    main()
