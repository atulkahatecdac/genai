"""
Demo 5: Conversational memory types - Buffer, Summary, Vector.

Implements three small memory strategies and runs the same conversation
through each, then compares what context each one sends to the model for
a final question that depends on something said earlier.

- BufferMemory: keeps the full raw conversation.
- SummaryMemory: keeps a running LLM-generated summary instead of raw turns.
- VectorMemory: embeds each turn and retrieves only the most relevant ones
  for the current question.

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
SYSTEM_PROMPT = "You are a helpful personal assistant."

CONVERSATION_TURNS = [
    "My name is Jordan and I live in Seattle.",
    "I'm allergic to shellfish, please keep that in mind for any food suggestions.",
    "I'm planning a small dinner party for 6 people next Saturday.",
    "By the way, my favorite color is green, not that it matters much here.",
    "What's a good seafood-free appetizer I could serve at the party?",
]


def call_model(client, messages):
    response = client.chat.completions.create(model=CHAT_MODEL, messages=messages)
    return response.choices[0].message.content


class BufferMemory:
    def __init__(self):
        self.turns = []

    def add_turn(self, client, user_message, assistant_message):
        self.turns.append(("user", user_message))
        self.turns.append(("assistant", assistant_message))

    def get_context(self, client, query):
        return [{"role": role, "content": content} for role, content in self.turns]


class SummaryMemory:
    def __init__(self):
        self.summary = ""

    def add_turn(self, client, user_message, assistant_message):
        prompt = (
            f"Current summary of the conversation: {self.summary or '(empty)'}\n\n"
            f"New exchange:\nUser: {user_message}\nAssistant: {assistant_message}\n\n"
            "Update the summary to include any new facts worth remembering, in 1-2 sentences."
        )
        self.summary = call_model(client, [{"role": "user", "content": prompt}])

    def get_context(self, client, query):
        return [{"role": "system", "content": f"What you remember about the user: {self.summary}"}]


class VectorMemory:
    def __init__(self):
        self.entries = []  # list of (text, embedding)

    @staticmethod
    def cosine_similarity(vec_a, vec_b):
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        return dot / (norm_a * norm_b)

    def _embed(self, client, text):
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
        return response.data[0].embedding

    def add_turn(self, client, user_message, assistant_message):
        text = f"User: {user_message}\nAssistant: {assistant_message}"
        self.entries.append((text, self._embed(client, text)))

    def get_context(self, client, query, top_k=2):
        query_embedding = self._embed(client, query)
        scored = [(self.cosine_similarity(query_embedding, emb), text) for text, emb in self.entries]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        top_entries = [text for _, text in scored[:top_k]]
        return [{"role": "system", "content": "Relevant past context:\n" + "\n---\n".join(top_entries)}]


def run_with_memory(client, name, memory):
    print(f"=== {name} ===")

    for user_message in CONVERSATION_TURNS[:-1]:
        context = memory.get_context(client, user_message)
        reply = call_model(client, [{"role": "system", "content": SYSTEM_PROMPT}] + context + [{"role": "user", "content": user_message}])
        memory.add_turn(client, user_message, reply)

    final_question = CONVERSATION_TURNS[-1]
    context = memory.get_context(client, final_question)
    print(f"Context sent for final question: {context}")

    final_reply = call_model(
        client,
        [{"role": "system", "content": SYSTEM_PROMPT}] + context + [{"role": "user", "content": final_question}],
    )
    print(f"Final question: {final_question}")
    print(f"Reply: {final_reply}\n")


def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    run_with_memory(client, "Buffer Memory", BufferMemory())
    run_with_memory(client, "Summary Memory", SummaryMemory())
    run_with_memory(client, "Vector Memory", VectorMemory())


if __name__ == "__main__":
    main()
