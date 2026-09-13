"""
Demo 13: Continuing context across sessions.

Simulates two separate sessions in one run to prove memory actually
survives a "restart": session 1 states a fact and saves the conversation to
disk, then session 2 reloads that file from scratch (as a fresh process
would) and correctly recalls the fact when asked a follow-up question.

Setup:
    pip install openai python-dotenv
    Set OPENAI_API_KEY in the .env file in the parent GenAI folder (shared
    across modules).
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = "You are a helpful assistant that remembers what users tell you across sessions."
SESSION_MEMORY_PATH = Path(__file__).parent / "assets" / "session_memory.json"

SESSION_1_MESSAGE = "My name is Alex and I prefer vegetarian recipes for dinner."
SESSION_2_MESSAGE = "Can you suggest a dinner recipe for me tonight?"


def load_memory():
    if SESSION_MEMORY_PATH.exists():
        return json.loads(SESSION_MEMORY_PATH.read_text())
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def save_memory(messages):
    SESSION_MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SESSION_MEMORY_PATH.write_text(json.dumps(messages, indent=2))


def run_session(client, session_name, user_message):
    messages = load_memory()
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(model=MODEL, messages=messages)
    reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})

    save_memory(messages)

    print(f"=== {session_name} ===")
    print(f"User: {user_message}")
    print(f"Assistant: {reply}\n")
    return reply


def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Start from a clean file so this demo is repeatable each time it's run.
    if SESSION_MEMORY_PATH.exists():
        SESSION_MEMORY_PATH.unlink()

    run_session(client, "Session 1", SESSION_1_MESSAGE)

    print("(simulating a fresh process - session 2 only ever reads the saved file from disk)\n")

    final_reply = run_session(client, "Session 2 (reloaded from disk)", SESSION_2_MESSAGE)

    print("=== Continuity check ===")
    print(f"Session 2 reply mentions vegetarian? {'veget' in final_reply.lower()}")


if __name__ == "__main__":
    main()
