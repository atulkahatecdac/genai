"""
Demo 6: How function calling works - the basic loop.

Defines one local get_ticket_status tool backed by a small dummy support
ticket database, lets the model decide when to call it, executes it
locally, and feeds the result back so the model can produce a final
natural-language answer. The model never touches the database - your code
does.

Setup:
    pip install openai python-dotenv
    Set OPENAI_API_KEY in the .env file in the parent GenAI folder (shared
    across modules).
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "gpt-4o-mini"
TICKETS_PATH = Path(__file__).parent / "assets" / "support_tickets.json"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_ticket_status",
            "description": "Look up the status of a support ticket by its ticket ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_id": {"type": "string", "description": "Ticket ID, e.g. 'TCK-1001'"},
                },
                "required": ["ticket_id"],
            },
        },
    }
]


def load_tickets():
    return json.loads(TICKETS_PATH.read_text())


def get_ticket_status(ticket_id: str) -> str:
    for ticket in load_tickets():
        if ticket["ticket_id"] == ticket_id:
            return json.dumps({"ticket_id": ticket_id, "status": ticket["status"], "urgency": ticket["urgency"]})
    return json.dumps({"ticket_id": ticket_id, "error": "not found"})


def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    messages = [{"role": "user", "content": "What's the status of ticket TCK-1002?"}]

    response = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS)
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
        result = get_ticket_status(**args)
        print(f"Tool result: {result}")
        messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})

    final_response = client.chat.completions.create(model=MODEL, messages=messages)
    print(f"\nFinal response:\n{final_response.choices[0].message.content}")


if __name__ == "__main__":
    main()
