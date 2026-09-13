"""
Demo 3: Basic example using the Google Gemini LLM.

Sends a single prompt to Gemini and prints the response.

Setup:
    pip install google-genai python-dotenv
    Set GOOGLE_API_KEY in a .env file (see .env.example).
"""

import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

MODEL = "gemini-2.5-flash"


def main():
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    response = client.models.generate_content(
        model=MODEL,
        contents="In two sentences, explain what a large language model is.",
    )

    print(f"Model: {MODEL}")
    print("Response:")
    print(response.text)
    print("\nUsage metadata:", response.usage_metadata)


if __name__ == "__main__":
    main()
