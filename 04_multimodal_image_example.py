"""
Demo 4: Multimodal example (image understanding).

Sends a photo to an OpenAI vision-capable model and asks it to identify the
animals present in the image.

Setup:
    pip install openai python-dotenv
    Set OPENAI_API_KEY in a .env file (see .env.example).
    Place a photo containing one or more animals at assets/animals.png
"""

import base64
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "gpt-4o-mini"
IMAGE_PATH = Path(__file__).parent / "assets" / "animals.png"

PROMPT = (
    "Identify every animal you can see in this image. For each one, name the "
    "species (or general type if the exact species isn't clear), roughly "
    "where it is in the frame, and what it appears to be doing."
)


def encode_image(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def main():
    if not IMAGE_PATH.exists():
        print(f"Image not found: {IMAGE_PATH}")
        print("Add a photo containing one or more animals at that path and re-run.")
        sys.exit(1)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    image_b64 = encode_image(IMAGE_PATH)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            }
        ],
        max_tokens=300,
    )

    print(f"Model: {MODEL}")
    print(f"Image: {IMAGE_PATH}")
    print("Response:")
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
