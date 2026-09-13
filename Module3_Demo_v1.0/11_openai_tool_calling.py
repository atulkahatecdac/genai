"""
Demo 11: Tool calling (function calling) example using OpenAI.

Defines a local `get_current_weather` function, lets the model decide when
to call it, executes it locally, and feeds the result back so the model can
produce a final natural-language answer.

Setup:
    pip install openai python-dotenv
    Set OPENAI_API_KEY in a .env file (see .env.example).
"""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "gpt-4o-mini"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name, e.g. 'Bengaluru'"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        },
    }
]


def get_current_weather(city: str, unit: str = "celsius") -> str:
    """Fake weather lookup - stands in for a real weather API call."""
    fake_data = {"city": city, "temperature": 27, "unit": unit, "condition": "Partly cloudy"}
    return json.dumps(fake_data)


def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    messages = [{"role": "user", "content": "What's the weather like in Bengaluru right now?"}]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
    )

    message = response.choices[0].message
    tool_calls = message.tool_calls

    if not tool_calls:
        print("Model answered directly (no tool call):")
        print(message.content)
        return

    messages.append(message)

    for tool_call in tool_calls:
        args = json.loads(tool_call.function.arguments)
        print(f"Model called tool: {tool_call.function.name}({args})")

        result = get_current_weather(**args)

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            }
        )

    final_response = client.chat.completions.create(model=MODEL, messages=messages)

    print(f"\nModel: {MODEL}")
    print("Final response:")
    print(final_response.choices[0].message.content)


if __name__ == "__main__":
    main()
