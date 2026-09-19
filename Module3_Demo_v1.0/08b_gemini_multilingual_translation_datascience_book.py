"""
Demo 8b: Multilingual translation example using Gemini, applied to a real
textbook.

Translates the opening definition of "data science" from the free OpenStax
textbook "Principles of Data Science" into several target languages. Same
pattern as demo 8, just with a real-world source sentence instead of a
made-up one.

Setup:
    pip install google-genai python-dotenv
    Set GOOGLE_API_KEY in a .env file (see .env.example).
"""

import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

MODEL = "gemini-2.5-flash"

SOURCE_TEXT = (
    "Data science is a field of study that investigates how to collect, manage, "
    "and analyze data of all types in order to retrieve meaningful information. "
    "You can consider data to be any pieces of evidence or observations that can "
    "be analyzed to provide some insights."
)
SOURCE_CITATION = 'Principles of Data Science (OpenStax), Section 1.1 "What Is Data Science?"'
TARGET_LANGUAGES = ["French", "Hindi", "Japanese", "Spanish"]


def translate(client: genai.Client, text: str, target_language: str) -> str:
    prompt = (
        f"Translate the following English text into {target_language}. "
        f"Return only the translation, with no extra commentary.\n\n"
        f"Text: {text}"
    )
    response = client.models.generate_content(model=MODEL, contents=prompt)
    return response.text.strip()


def main():
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    print(f"Model: {MODEL}")
    print(f"Source: {SOURCE_CITATION}")
    print(f"Source (English): {SOURCE_TEXT}\n")

    for language in TARGET_LANGUAGES:
        translation = translate(client, SOURCE_TEXT, language)
        print(f"{language}: {translation}")


if __name__ == "__main__":
    main()
