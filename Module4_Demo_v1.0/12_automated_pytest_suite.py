"""
Demo 12: Prompt testing methods - automated testing with pytest.

Wraps a prompt in a callable and writes pytest assertions against its
output, so it can be re-run automatically as a regression suite whenever
the prompt changes.

Setup:
    pip install openai pytest python-dotenv
    Set OPENAI_API_KEY in the .env file in the parent GenAI folder (shared
    across modules).

Run with:
    pytest 12_automated_pytest_suite.py -v
"""

import os

import pytest
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = (
    "Extract the order number and issue type from the customer message. "
    "Respond with exactly two lines:\n"
    "Order: <order number or UNKNOWN>\n"
    "Issue: <one of DAMAGED, LATE, WRONG_ITEM, OTHER>"
)


def extract(message):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ],
        temperature=0,
    )
    return response.choices[0].message.content


def test_damaged_item():
    output = extract("Order #4471 arrived with a cracked screen.")
    assert "4471" in output
    assert "DAMAGED" in output


def test_late_delivery():
    output = extract("It's been 3 weeks and order 9981 still hasn't shipped.")
    assert "9981" in output
    assert "LATE" in output


def test_missing_order_number():
    output = extract("My package never arrived, I don't have the order number handy.")
    assert "UNKNOWN" in output


def test_output_has_two_lines():
    output = extract("Order 1002 came with the wrong color shirt.")
    lines = [line for line in output.splitlines() if line.strip()]
    assert len(lines) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
