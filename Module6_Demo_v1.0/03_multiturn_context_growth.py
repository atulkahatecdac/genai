"""
Demo 3: Multi-turn conversations and context growth.

Runs a scripted 5-turn conversation, appending each user message and the
model's reply to the running message list, and prints the cumulative token
count before each call so the growth trend is visible.

Setup:
    pip install openai tiktoken python-dotenv
    Set OPENAI_API_KEY in the .env file in the parent GenAI folder (shared
    across modules).
"""

import os

import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "gpt-4o-mini"
encoding = tiktoken.encoding_for_model(MODEL)

SYSTEM_PROMPT = "You are a helpful travel planning assistant. Keep replies to 2-3 sentences."

USER_TURNS = [
    "I'm planning a 5-day trip to Japan in April. Any general tips?",
    "I'll be traveling with my 70-year-old parents, does that change anything?",
    "We're most interested in gardens and quiet temples, not crowded tourist spots.",
    "What about food - my mom is vegetarian.",
    "Can you summarize the plan so far in a short itinerary?",
]


def count_messages_tokens(messages):
    total = 0
    for message in messages:
        total += 3  # per-message overhead (role, separators) - approximate
        total += len(encoding.encode(message["content"]))
    total += 3  # priming tokens for the reply
    return total


def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for turn_number, user_message in enumerate(USER_TURNS, start=1):
        messages.append({"role": "user", "content": user_message})

        token_count = count_messages_tokens(messages)
        print(f"--- Turn {turn_number} ---")
        print(f"User: {user_message}")
        print(f"Context size before this call: {token_count} tokens")

        response = client.chat.completions.create(model=MODEL, messages=messages)
        reply = response.choices[0].message.content
        print(f"Assistant: {reply}\n")

        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
