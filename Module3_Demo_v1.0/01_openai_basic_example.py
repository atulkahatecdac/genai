"""
Demo 1: Basic example using the OpenAI LLM.

Sends a single prompt to an OpenAI chat model and prints the response.

Setup:
    pip install openai python-dotenv
    Set OPENAI_API_KEY in a .env file (see .env.example).
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "gpt-4o-mini"


def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a concise, helpful assistant."},
            {"role": "user", "content": "In two sentences, explain what a large language model is."},
        ],
        temperature=0.7,
        max_tokens=20,
    )

    print(f"Model: {MODEL}")
    print("Response:")
    print(response.choices[0].message.content)
    print("\nToken usage:", response.usage)


if __name__ == "__main__":
    main()
