"""
Demo 11: Handling malformed or partial model output.

Wraps a ticket-classification call in a retry loop that catches JSON
parse errors and Pydantic validation errors (including rejecting
unexpected extra keys), and re-prompts the model with the specific error
message so it can correct itself.

Setup:
    pip install openai pydantic python-dotenv
    Set OPENAI_API_KEY in the .env file in the parent GenAI folder (shared
    across modules).
"""

import json
import os
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, ValidationError

load_dotenv()

MODEL = "gpt-4o-mini"

TICKET_TEXT = "yo my thing is broken again lol, kinda annoying, idk what plan im on"


class TicketClassification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    issue_type: Literal["billing", "technical", "other"]
    urgency: Literal["low", "medium", "high"]


BASE_PROMPT = (
    "Classify this customer message into JSON with fields issue_type "
    f"(billing/technical/other) and urgency (low/medium/high):\n\n{TICKET_TEXT}"
)


def classify(client, prompt, max_retries=2):
    for attempt in range(max_retries + 1):
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        try:
            data = json.loads(raw)
            return TicketClassification.model_validate(data)
        except json.JSONDecodeError as exc:
            print(f"  Attempt {attempt + 1}: invalid JSON ({exc}), retrying...")
            prompt = f"{BASE_PROMPT}\n\nYour last reply wasn't valid JSON. Return only a JSON object."
        except ValidationError as exc:
            errors = ", ".join(f"{e['loc']}: {e['msg']}" for e in exc.errors())
            print(f"  Attempt {attempt + 1}: validation failed ({errors}), retrying...")
            prompt = f"{BASE_PROMPT}\n\nYour last reply had these problems: {errors}. Fix them and return only JSON."
    raise RuntimeError("Could not get valid output after retries")


def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print(f"Message: {TICKET_TEXT}\n")
    result = classify(client, BASE_PROMPT)
    print(f"\nFinal result: {result}")


if __name__ == "__main__":
    main()
