"""
Demo 6: Persisting conversational memory.

Saves conversation history to a JSON file on disk after each turn, and
loads it back in on startup if it exists - so memory survives between
separate runs of the script, not just within one process.

Setup:
    pip install openai python-dotenv
    Set OPENAI_API_KEY in the .env file in the parent GenAI folder (shared
    across modules).

Usage:
    Run this script multiple times with different NEW_MESSAGE values (or
    edit it) and watch the history in assets/conversation_memory.json grow.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = "You are a helpful assistant with a good memory for past conversations."
MEMORY_PATH = Path(__file__).parent / "assets" / "conversation_memory.json"

NEW_MESSAGE = "What have we talked about so far?"


def load_memory():
    if MEMORY_PATH.exists():
        return json.loads(MEMORY_PATH.read_text())
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def save_memory(messages):
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    MEMORY_PATH.write_text(json.dumps(messages, indent=2))


def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    messages = load_memory()
    prior_turns = (len(messages) - 1) // 2
    print(f"Loaded {prior_turns} prior turn(s) from {MEMORY_PATH}")

    messages.append({"role": "user", "content": NEW_MESSAGE})
    response = client.chat.completions.create(model=MODEL, messages=messages)
    reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})

    print(f"\nUser: {NEW_MESSAGE}")
    print(f"Assistant: {reply}")

    save_memory(messages)
    print(f"\nSaved {(len(messages) - 1) // 2} total turn(s) to {MEMORY_PATH}")


if __name__ == "__main__":
    main()
