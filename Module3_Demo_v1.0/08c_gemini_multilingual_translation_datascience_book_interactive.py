"""
Demo 8c: Interactive multilingual translation example using Gemini.

Same idea as demo 8b, but instead of a fixed sentence and language list this
version loops, letting you type the text to translate and the target
language each round, until you decide to stop. Press Enter at the text
prompt to reuse the example sentence from "Principles of Data Science".

Setup:
    pip install google-genai python-dotenv
    Set GOOGLE_API_KEY in a .env file (see .env.example).
"""

import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

MODEL = "gemini-2.5-flash"
QUIT_WORDS = ("quit", "exit", "q")

EXAMPLE_TEXT = (
    "Data science is a field of study that investigates how to collect, manage, "
    "and analyze data of all types in order to retrieve meaningful information."
)
EXAMPLE_CITATION = 'Principles of Data Science (OpenStax), Section 1.1 "What Is Data Science?"'


def translate(client: genai.Client, text: str, target_language: str) -> str:
    prompt = (
        f"Translate the following text into {target_language}. "
        f"Return only the translation, with no extra commentary.\n\n"
        f"Text: {text}"
    )
    response = client.models.generate_content(model=MODEL, contents=prompt)
    return response.text.strip()


def main():
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    print(f"Model: {MODEL}")
    print(f"Example text ({EXAMPLE_CITATION}):\n{EXAMPLE_TEXT}\n")
    print("Enter text to translate, or press Enter to use the example above.")
    print("Type 'quit' at either prompt to exit.\n")

    while True:
        text = input("Text to translate: ").strip()
        if text.lower() in QUIT_WORDS:
            print("Goodbye!")
            break
        if not text:
            text = EXAMPLE_TEXT
            print("(using example text)")

        target_language = input("Target language: ").strip()
        if target_language.lower() in QUIT_WORDS:
            print("Goodbye!")
            break
        if not target_language:
            print("No target language entered - try again.\n")
            continue

        translation = translate(client, text, target_language)
        print(f"\n{target_language}: {translation}\n")


if __name__ == "__main__":
    main()
