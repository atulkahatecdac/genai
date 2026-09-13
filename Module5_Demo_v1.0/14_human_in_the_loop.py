"""
Demo 14: Human in the loop.

A support assistant drafts a reply to a customer, but the draft is never
sent automatically - a human reviewer must approve it, edit it, or reject
it first. This is an interactive demo: run it in a terminal so you can
respond to the prompts.

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

SYSTEM_PROMPT = "You are a customer support assistant. Draft a short, empathetic reply to the customer's message."
CUSTOMER_MESSAGE = "I want a refund for my last order, it's been broken since day one."


def draft_reply(client, message):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ],
    )
    return response.choices[0].message.content


def send_email(body):
    print("\n--- EMAIL SENT (simulated) ---")
    print(body)
    print("-------------------------------")


def human_review(draft):
    print("\nProposed reply:\n" + draft)
    choice = input("\nApprove and send [a] / Edit before sending [e] / Reject [r]? ").strip().lower()

    if choice == "a":
        return draft
    if choice == "e":
        print("Enter your edited reply (press Enter when done):")
        return input("> ")
    print("Action cancelled by human reviewer - nothing was sent.")
    return None


def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    print(f"Customer message: {CUSTOMER_MESSAGE}")
    draft = draft_reply(client, CUSTOMER_MESSAGE)

    approved_text = human_review(draft)
    if approved_text:
        send_email(approved_text)


if __name__ == "__main__":
    main()
