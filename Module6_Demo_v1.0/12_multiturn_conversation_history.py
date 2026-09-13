"""
Demo 12: Multi-turn conversation history.

Contrasts a stateless bot (each turn sent with no memory of earlier turns)
against a stateful bot (the full message history is kept and resent) on a
conversation where a later turn depends on information from an earlier one.

Setup:
    pip install openai python-dotenv
    Set OPENAI_API_KEY in the .env file in the parent GenAI folder (shared
    across modules).
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = "You are a helpful assistant that suggests snacks."

CONVERSATION = [
    "My name is Priya and I'm allergic to peanuts.",
    "Can you suggest a snack for me?",
]


def call_model(client, messages):
    response = client.chat.completions.create(model=MODEL, messages=messages)
    return response.choices[0].message.content


def stateful_run(client):
    print("=== Stateful (keeps full history) ===")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    reply = None
    for turn in CONVERSATION:
        messages.append({"role": "user", "content": turn})
        reply = call_model(client, messages)
        messages.append({"role": "assistant", "content": reply})
        print(f"User: {turn}")
        print(f"Assistant: {reply}\n")
    return reply


def stateless_run(client):
    print("=== Stateless (no memory between turns) ===")
    reply = None
    for turn in CONVERSATION:
        # Each call starts fresh - the model has no idea what was said before.
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": turn}]
        reply = call_model(client, messages)
        print(f"User: {turn}")
        print(f"Assistant: {reply}\n")
    return reply


def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    stateful_final = stateful_run(client)
    stateless_final = stateless_run(client)

    print("=== Safety check on the final snack suggestion ===")
    print(f"Stateful reply mentions peanuts?  {'peanut' in stateful_final.lower()}")
    print(f"Stateless reply mentions peanuts? {'peanut' in stateless_final.lower()}")


if __name__ == "__main__":
    main()
