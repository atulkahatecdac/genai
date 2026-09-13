"""
Demo 2: Basic example using the Anthropic (Claude) LLM.

Sends a single prompt to Claude and prints the response.

Setup:
    pip install anthropic python-dotenv
    Set ANTHROPIC_API_KEY in a .env file (see .env.example).
"""

import os
from dotenv import load_dotenv
import anthropic

load_dotenv()

# claude-opus-5 is Anthropic's current flagship model. For high-volume or
# latency-sensitive workloads, claude-sonnet-5 or claude-haiku-4-5 are cheaper
# alternatives - see 12_cost_latency_calculator_openai.py for how to reason
# about that tradeoff.
MODEL = "claude-opus-5"


def main():
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system="You are a concise, helpful assistant.",
        messages=[
            {"role": "user", "content": "In two sentences, explain what a large language model is."}
        ],
    )

    text = "".join(block.text for block in response.content if block.type == "text")

    print(f"Model: {MODEL}")
    print("Response:")
    print(text)
    print("\nToken usage:", response.usage)


if __name__ == "__main__":
    main()
