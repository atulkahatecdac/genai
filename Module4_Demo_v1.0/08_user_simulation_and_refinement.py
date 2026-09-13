"""
Demo 8: Prompt testing and evaluation - user simulation and iterative
refinement.

Part 1 uses one LLM call to simulate different user personas talking to a
support-bot prompt. Part 2 shows iterative refinement: a prompt fails a
known tricky case, gets tightened, and is retested.

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

SUPPORT_BOT_SYSTEM_PROMPT = "You are a support bot for a cloud storage app. Answer briefly and helpfully."

PERSONAS = [
    "an impatient user whose files won't sync and needs a fast answer",
    "a confused first-time user who doesn't know technical terms",
    "a technical power user asking about API rate limits",
]

TRICKY_CASE = "How do I delete my account and ALL my data permanently?"


def simulate_user_message(client, persona):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": f"Write one realistic support chat message from {persona}. Only output the message.",
            }
        ],
        temperature=0.9,
    )
    return response.choices[0].message.content.strip()


def ask_support_bot(client, system_prompt, user_message):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content


def run_user_simulation(client):
    print("=== User simulation ===")
    for persona in PERSONAS:
        user_message = simulate_user_message(client, persona)
        reply = ask_support_bot(client, SUPPORT_BOT_SYSTEM_PROMPT, user_message)
        print(f"\nPersona: {persona}")
        print(f"Simulated message: {user_message}")
        print(f"Bot reply: {reply}")


def response_mentions_confirmation(text):
    lowered = text.lower()
    return "confirm" in lowered or "are you sure" in lowered


def run_iterative_refinement(client):
    print("\n=== Iterative refinement ===")

    reply_v1 = ask_support_bot(client, SUPPORT_BOT_SYSTEM_PROMPT, TRICKY_CASE)
    print(f"\nPrompt v1 reply:\n{reply_v1}")
    print(f"Asks for confirmation before a destructive action? {response_mentions_confirmation(reply_v1)}")

    prompt_v2 = (
        SUPPORT_BOT_SYSTEM_PROMPT
        + " Before describing any destructive or irreversible action (like deleting an account or data), "
        "explicitly ask the user to confirm first."
    )
    reply_v2 = ask_support_bot(client, prompt_v2, TRICKY_CASE)
    print(f"\nPrompt v2 reply:\n{reply_v2}")
    print(f"Asks for confirmation before a destructive action? {response_mentions_confirmation(reply_v2)}")


def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    run_user_simulation(client)
    run_iterative_refinement(client)


if __name__ == "__main__":
    main()
