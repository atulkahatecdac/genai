"""
Demo 5: Basic example using Ollama (local Llama model).

Sends a single prompt to a locally-running Llama model via Ollama and prints
the response. No API key needed - everything runs on your machine.

Setup:
    1. Install Ollama from https://ollama.com
    2. Pull a model:  ollama pull llama3
    3. pip install ollama
"""

import ollama

MODEL = "llama3.2:latest"


def main():
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a concise, helpful assistant."},
            {"role": "user", "content": "In two sentences, explain what a large language model is."},
        ],
    )

    print(f"Model: {MODEL}")
    print("Response:")
    print(response["message"]["content"])
    print("\nEval count (output tokens, approx):", response.get("eval_count"))


if __name__ == "__main__":
    main()
