"""
Demo 2: Additional prompt elements - Persona, Examples, Tone and
Constraints, Delimiters and Syntax.

Runs the same task (drafting a reply to a customer email) twice: once with
only the core elements, and once with the additional elements layered on
top, so you can compare how much they change the output.

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

CUSTOMER_EMAIL = (
    "Hi, I was charged twice for my subscription this month and I'm pretty "
    "annoyed. Can you fix this?"
)

BASIC_PROMPT = f"Write a reply to this customer email:\n{CUSTOMER_EMAIL}"

ENHANCED_PROMPT = f"""You are a senior customer support specialist known for being warm, empathetic, and precise.

Example:
Customer: "My order arrived damaged."
Reply: "I'm really sorry your order arrived damaged - that's not the experience we want for you. I've started a replacement, no need to send anything back, and it should reach you within 3-5 business days."

Tone: warm but professional. Constraints: under 60 words, do not promise a specific refund date, do not admit fault on behalf of the billing system.

<customer_email>
{CUSTOMER_EMAIL}
</customer_email>

Write only the reply text, no subject line."""


def call_model(client, prompt):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response.choices[0].message.content


def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    print("=== Core elements only ===")
    print(call_model(client, BASIC_PROMPT))

    print("\n=== Core + Persona + Examples + Tone/Constraints + Delimiters ===")
    print(call_model(client, ENHANCED_PROMPT))


if __name__ == "__main__":
    main()
