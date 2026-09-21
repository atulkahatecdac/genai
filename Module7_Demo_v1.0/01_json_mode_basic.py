"""
Demo 1: JSON mode - valid JSON, but not guaranteed to have the keys you want.

Uses OpenAI's JSON mode (response_format={"type": "json_object"}) to turn a
raw customer support message into JSON, with no schema telling the model
what fields to use. Two different ticket messages are asked about the same
way, and the key names the model picks can still drift between them - JSON
mode guarantees valid JSON, not consistent or correct JSON.

Setup:
    pip install openai python-dotenv
    Set OPENAI_API_KEY in the .env file in the parent GenAI folder (shared
    across modules).
"""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "gpt-4o-mini"

RAW_TICKETS = [
    "Hi, I'm Sarah. My invoice for CloudSync Pro is wrong - I was charged twice this month.",
    "This is Miguel. CloudSync keeps failing to sync files over 2GB, very urgent, please help fast.",
]

PROMPT_TEMPLATE = (
    "Turn this customer support message into a JSON object with whatever "
    "fields you think are useful:\n\n{message}"
)


def to_json(client, message):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(message=message)}],
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content or "{}"


def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    for raw in RAW_TICKETS:
        print(f"Raw message: {raw}")
        raw_json = to_json(client, raw)
        print(f"Model JSON: {raw_json}")
        parsed = json.loads(raw_json)
        print(f"Keys used: {sorted(parsed.keys())}\n")

    print(
        "Notice the key names aren't guaranteed to match between calls - "
        "JSON mode guarantees valid JSON, not a consistent or correct shape."
    )


if __name__ == "__main__":
    main()
