"""
Demo 5: Prompt template styles - string, chat, and messages.

Builds the same translation task three ways: a flat string template, a
single chat call with a system + user turn, and a multi-turn messages
template that carries conversation history.

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


def string_template(language, sentence):
    template = 'Translate the following sentence to {language}: "{sentence}"'
    return template.format(language=language, sentence=sentence)


def chat_template(language, sentence):
    return [
        {"role": "system", "content": "You are a professional translator. Reply with only the translation."},
        {"role": "user", "content": f'Translate to {language}: "{sentence}"'},
    ]


def messages_template(language, sentence):
    # Multi-turn history: the model already agreed to a tone correction once,
    # and we want that correction to carry forward into the new translation.
    return [
        {"role": "system", "content": "You are a professional translator. Reply with only the translation."},
        {"role": "user", "content": 'Translate to French: "Where is the nearest train station?"'},
        {"role": "assistant", "content": "Ou est la gare la plus proche ?"},
        {"role": "user", "content": "Keep it informal from now on, use 'tu' instead of 'vous' where relevant."},
        {"role": "assistant", "content": "Understood, I'll use informal French going forward."},
        {"role": "user", "content": f'Translate to {language}: "{sentence}"'},
    ]


def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    language, sentence = "French", "Can you help me find my hotel?"

    print("=== String template ===")
    prompt = string_template(language, sentence)
    print("Rendered:", prompt)
    response = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}])
    print("Output:", response.choices[0].message.content)

    print("\n=== Chat template (system + user) ===")
    messages = chat_template(language, sentence)
    print("Rendered:", messages)
    response = client.chat.completions.create(model=MODEL, messages=messages)
    print("Output:", response.choices[0].message.content)

    print("\n=== Messages template (multi-turn history) ===")
    messages = messages_template(language, sentence)
    print("Rendered:", messages)
    response = client.chat.completions.create(model=MODEL, messages=messages)
    print("Output:", response.choices[0].message.content)


if __name__ == "__main__":
    main()
