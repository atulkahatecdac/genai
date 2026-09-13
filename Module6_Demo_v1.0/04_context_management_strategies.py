"""
Demo 4: Context management strategies - chunking, summarization, windowing.

Takes the same long, fixed conversation history and prepares it for a new
question three different ways: sending the full history, windowing it down
to only the last few turns, and summarizing the older turns into one short
note. Compares the resulting token counts for each strategy.

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

SYSTEM_PROMPT = "You are a technical support assistant for a home printer."

HISTORY = [
    ("user", "My printer won't turn on at all, the power light is off."),
    ("assistant", "Let's check the basics - is the power cable firmly connected at both ends?"),
    ("user", "Yes, I checked that. Still nothing."),
    ("assistant", "Try a different power outlet, and if you have one, a different power cable."),
    ("user", "Different outlet worked! It's on now, model number is HP-2210."),
    ("assistant", "Good to hear. The HP-2210 sometimes needs a firmware update after being unplugged for a while."),
    ("user", "How do I check for a firmware update?"),
    ("assistant", "Go to Settings > About > Check for Updates on the printer's touchscreen."),
    ("user", "Found an update, installing it now. Should take 5 minutes it says."),
    ("assistant", "Sounds right - don't unplug it during the update."),
]

NEW_QUESTION = "It's back on after the update, but what was the model number again, and is it still under warranty?"

WINDOW_SIZE = 4  # keep only the last N raw turns


def to_messages(history):
    return [{"role": role, "content": content} for role, content in history]


def count_messages_tokens(messages):
    total = 0
    for message in messages:
        total += 3
        total += len(encoding.encode(message["content"]))
    total += 3
    return total


def full_history_strategy():
    return [{"role": "system", "content": SYSTEM_PROMPT}] + to_messages(HISTORY)


def windowed_strategy():
    windowed = HISTORY[-WINDOW_SIZE:]
    return [{"role": "system", "content": SYSTEM_PROMPT}] + to_messages(windowed)


def summarized_strategy(client):
    older = HISTORY[:-2]
    recent = HISTORY[-2:]
    transcript = "\n".join(f"{role}: {content}" for role, content in older)

    summary_response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": f"Summarize this support conversation in 1-2 sentences, keeping any specific facts (model numbers, steps taken):\n\n{transcript}"}],
        temperature=0,
    )
    summary = summary_response.choices[0].message.content

    return (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + [{"role": "system", "content": f"Summary of earlier conversation: {summary}"}]
        + to_messages(recent)
    )


def run_and_report(client, name, messages):
    messages_with_question = messages + [{"role": "user", "content": NEW_QUESTION}]
    token_count = count_messages_tokens(messages_with_question)
    response = client.chat.completions.create(model=MODEL, messages=messages_with_question)
    reply = response.choices[0].message.content

    print(f"=== {name} ===")
    print(f"Context size: {token_count} tokens")
    print(f"Reply: {reply}\n")


def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    run_and_report(client, "Full history", full_history_strategy())
    run_and_report(client, f"Windowed (last {WINDOW_SIZE} turns)", windowed_strategy())
    run_and_report(client, "Summarized older turns", summarized_strategy(client))


if __name__ == "__main__":
    main()
