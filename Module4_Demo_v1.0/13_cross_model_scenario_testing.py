"""
Demo 13: Prompt testing methods - cross-model and scenario-based testing.

Runs the same realistic support scenarios through the same prompt on three
different providers (OpenAI, Anthropic, Gemini) so their responses can be
compared side by side.

Setup:
    pip install openai anthropic google-genai python-dotenv
    Set OPENAI_API_KEY, ANTHROPIC_API_KEY, and GOOGLE_API_KEY in the .env
    file in the parent GenAI folder (shared across modules).
"""

import os

from dotenv import load_dotenv
from openai import OpenAI
import anthropic
from google import genai

load_dotenv()

OPENAI_MODEL = "gpt-4o-mini"
ANTHROPIC_MODEL = "claude-opus-5"
GEMINI_MODEL = "gemini-2.5-flash"

SYSTEM_PROMPT = "You are a support agent. Reply in at most 3 sentences, be empathetic but concise."

SCENARIOS = [
    {
        "name": "Angry customer requesting a refund",
        "message": "This is the third time your product has failed. I want a full refund immediately, no exceptions.",
    },
    {
        "name": "Confused first-time user",
        "message": "I don't understand how to reset my password, the link in the email doesn't seem to work.",
    },
    {
        "name": "Billing question in French",
        "message": "Bonjour, pourquoi ai-je ete facture deux fois ce mois-ci ?",
    },
]


def ask_openai(client, message):
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ],
    )
    return response.choices[0].message.content


def ask_anthropic(client, message):
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=200,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": message}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def ask_gemini(client, message):
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=f"{SYSTEM_PROMPT}\n\nCustomer message: {message}",
    )
    return response.text


def main():
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    for scenario in SCENARIOS:
        print(f"=== Scenario: {scenario['name']} ===")
        print(f"Message: {scenario['message']}\n")
        print(f"OpenAI ({OPENAI_MODEL}):\n{ask_openai(openai_client, scenario['message'])}\n")
        print(f"Anthropic ({ANTHROPIC_MODEL}):\n{ask_anthropic(anthropic_client, scenario['message'])}\n")
        print(f"Gemini ({GEMINI_MODEL}):\n{ask_gemini(gemini_client, scenario['message'])}\n")


if __name__ == "__main__":
    main()
