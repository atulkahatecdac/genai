"""
Demo 10: Demonstrating temperature, top_p, top_k, and max_tokens.

Uses Gemini because its GenerateContentConfig exposes all four sampling
parameters directly (temperature, top_p, top_k, max_output_tokens), which
makes it easy to see their individual effects in one place.

- temperature: randomness/creativity. 0 = deterministic, higher = more varied.
- top_p (nucleus sampling): only sample from the smallest set of tokens whose
  cumulative probability >= top_p.
- top_k: only sample from the k most likely next tokens.
- max_output_tokens: hard cap on response length.

Setup:
    pip install google-genai python-dotenv
    Set GOOGLE_API_KEY in a .env file (see .env.example).
"""

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

MODEL = "gemini-2.5-flash"
PROMPT = "Write one creative opening sentence for a story about a lighthouse keeper."

CONFIGS = [
    {"label": "Low temperature (deterministic)", "temperature": 0.0, "top_p": 1.0, "top_k": 1, "max_output_tokens": 60},
    {"label": "High temperature (creative)", "temperature": 1.2, "top_p": 0.95, "top_k": 40, "max_output_tokens": 60},
    {"label": "Narrow top_k (few token choices)", "temperature": 0.9, "top_p": 1.0, "top_k": 3, "max_output_tokens": 60},
    {"label": "Tight max_output_tokens (truncated)", "temperature": 0.9, "top_p": 0.95, "top_k": 40, "max_output_tokens": 12},
]


def main():
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    print(f"Model: {MODEL}")
    print(f"Prompt: {PROMPT}\n")

    for raw_config in CONFIGS:
        config = dict(raw_config)
        label = config.pop("label")
        response = client.models.generate_content(
            model=MODEL,
            contents=PROMPT,
            config=types.GenerateContentConfig(**config),
        )
        print(f"--- {label} ---")
        print(f"params: {config}")
        print(f"output: {response.text.strip()}\n")


if __name__ == "__main__":
    main()
