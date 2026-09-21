"""
Demo 9: Designing safe tool input schemas.

A query_tickets tool constrained with Literal types, numeric bounds, and a
regex pattern - so the model (or any caller) can't request something the
schema doesn't allow, before it ever reaches the "database".

Setup:
    pip install langchain langchain-openai pydantic python-dotenv
    Set OPENAI_API_KEY in the .env file in the parent GenAI folder (shared
    across modules).
"""

import json
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

MODEL = "gpt-4o-mini"
TICKETS_PATH = Path(__file__).parent / "assets" / "support_tickets.json"


def load_tickets():
    return json.loads(TICKETS_PATH.read_text())


class QueryTicketsArgs(BaseModel):
    issue_type: Optional[Literal["billing", "technical", "other"]] = None
    status: Optional[Literal["open", "in_progress", "resolved"]] = None
    limit: int = Field(default=10, ge=1, le=50, description="max rows to return, 1-50")
    ticket_id: Optional[str] = Field(default=None, pattern=r"^TCK-\d{4}$")


@tool(args_schema=QueryTicketsArgs)
def query_tickets(issue_type=None, status=None, limit=10, ticket_id=None):
    """Query support tickets by issue type, status, or a specific ticket ID."""
    tickets = load_tickets()
    if ticket_id:
        tickets = [t for t in tickets if t["ticket_id"] == ticket_id]
    if issue_type:
        tickets = [t for t in tickets if t["issue_type"] == issue_type]
    if status:
        tickets = [t for t in tickets if t["status"] == status]
    return tickets[:limit]


def main():
    print("Valid call: query_tickets(issue_type='billing', limit=2)")
    print(query_tickets.invoke({"issue_type": "billing", "limit": 2}))

    print("\nInvalid call: query_tickets(limit=500) - exceeds the le=50 bound")
    try:
        QueryTicketsArgs.model_validate({"limit": 500})
    except ValidationError as exc:
        print(f"Rejected before it ever reached the database:\n{exc}")

    print("\nInvalid call: a ticket_id that doesn't match the expected pattern")
    try:
        QueryTicketsArgs.model_validate({"ticket_id": "hack'; DROP TABLE tickets;"})
    except ValidationError as exc:
        print(f"Rejected before it ever reached the database:\n{exc}")

    print("\nLetting the model use the tool:")
    llm = ChatOpenAI(model=MODEL)
    llm_with_tools = llm.bind_tools([query_tickets])
    response = llm_with_tools.invoke("Show me up to 3 open tickets.")
    for call in response.tool_calls:
        print(f"Model called: {call['name']}({call['args']})")
        print(f"Result: {query_tickets.invoke(call['args'])}")


if __name__ == "__main__":
    main()
