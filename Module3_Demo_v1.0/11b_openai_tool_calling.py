"""
Demo 11b: Tool calling (function calling) example using OpenAI, with a real
API behind the tool.

Same pattern as demo 11, but instead of a fake weather lookup this version
defines a local `get_stock_price` function that fetches a real, live stock
price from Yahoo Finance's public chart endpoint, lets the model decide when
to call it, executes it locally, and feeds the result back so the model can
produce a final natural-language answer.

Setup:
    pip install openai requests python-dotenv
    Set OPENAI_API_KEY in a .env file (see .env.example).

Note: this uses Yahoo Finance's unofficial, undocumented chart API (no API
key required), so it may break or rate-limit without notice - it's meant to
show a real network call, not to be relied on for production trading.
"""

import json
import os

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "gpt-4o-mini"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current stock price for a given ticker symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol, e.g. 'AAPL' for Apple, 'MSFT' for Microsoft",
                    },
                },
                "required": ["ticker"],
            },
        },
    }
]


def get_stock_price(ticker: str) -> str:
    """Look up the latest price for a ticker symbol from Yahoo Finance."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        meta = response.json()["chart"]["result"][0]["meta"]
        data = {
            "ticker": meta["symbol"],
            "price": meta["regularMarketPrice"],
            "currency": meta["currency"],
            "previous_close": meta.get("chartPreviousClose"),
        }
    except (requests.RequestException, KeyError, IndexError, TypeError) as exc:
        data = {"ticker": ticker, "error": f"Could not fetch price ({exc})"}

    return json.dumps(data)


def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    messages = [
        {
            "role": "user",
            "content": "What's Apple's current stock price, and how does it compare to its previous close?",
        }
    ]

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

        result = get_stock_price(**args)
        print(f"Tool result: {result}")

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
