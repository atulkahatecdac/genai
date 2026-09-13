"""
Demo 8: Multilingual translation example using Gemini.

Translates a single English sentence into several target languages.

Setup:
    pip install google-genai python-dotenv
    Set GOOGLE_API_KEY in a .env file (see .env.example).
"""

import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

MODEL = "gemini-2.5-flash"

SOURCE_TEXT = "The weather is beautiful today, and I am learning about generative AI."
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
    print(f"Source (English): {SOURCE_TEXT}\n")

    for language in TARGET_LANGUAGES:
        translation = translate(client, SOURCE_TEXT, language)
        print(f"{language}: {translation}")


if __name__ == "__main__":
    main()
