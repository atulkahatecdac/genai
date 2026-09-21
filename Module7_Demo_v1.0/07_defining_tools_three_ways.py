"""
Demo 7: Three ways to define a tool in LangChain.

Defines the same capability - look up a support ticket by ID - three
different ways: the @tool decorator (simplest), StructuredTool.from_function
(wrap an existing function with an explicit schema), and a BaseTool
subclass (for tools that need more structure). All three are bound to the
model together to show they're interchangeable at the model boundary.

Setup:
    pip install langchain langchain-openai python-dotenv
    Set OPENAI_API_KEY in the .env file in the parent GenAI folder (shared
    across modules).
"""

import json
from pathlib import Path
from typing import Type

from dotenv import load_dotenv
from langchain_core.tools import BaseTool, StructuredTool, tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

MODEL = "gpt-4o-mini"
TICKETS_PATH = Path(__file__).parent / "assets" / "support_tickets.json"


def load_tickets():
    return json.loads(TICKETS_PATH.read_text())


def _lookup(ticket_id: str) -> str:
    for ticket in load_tickets():
        if ticket["ticket_id"] == ticket_id:
            return f"{ticket_id} is {ticket['status']} ({ticket['urgency']} urgency)"
    return f"{ticket_id} not found"


class LookupArgs(BaseModel):
    ticket_id: str = Field(description="Ticket ID, e.g. 'TCK-1001'")


# 1. @tool decorator - simplest path, schema inferred from the type hints.
@tool
def lookup_ticket_decorator(ticket_id: str) -> str:
    """Look up a support ticket's status by its ticket ID, e.g. 'TCK-1001'."""
    return _lookup(ticket_id)


# 2. StructuredTool.from_function - wraps an existing function with an explicit schema.
lookup_ticket_structured = StructuredTool.from_function(
    func=_lookup,
    name="lookup_ticket_structured",
    args_schema=LookupArgs,
    description="Look up a support ticket's status by its ticket ID.",
)


# 3. BaseTool subclass - for tools that need state or more structure.
class LookupTicketTool(BaseTool):
    name: str = "lookup_ticket_class"
    description: str = "Look up a support ticket's status by its ticket ID."
    args_schema: Type[BaseModel] = LookupArgs

    def _run(self, ticket_id: str) -> str:
        return _lookup(ticket_id)


def main():
    print("Direct invocation, to prove all three do the same thing:")
    print(" @tool:          ", lookup_ticket_decorator.invoke({"ticket_id": "TCK-1002"}))
    print(" StructuredTool: ", lookup_ticket_structured.invoke({"ticket_id": "TCK-1002"}))
    print(" BaseTool:       ", LookupTicketTool().invoke({"ticket_id": "TCK-1002"}))

    print("\nBinding all three to the model together:")
    llm = ChatOpenAI(model=MODEL)
    llm_with_tools = llm.bind_tools(
        [lookup_ticket_decorator, lookup_ticket_structured, LookupTicketTool()]
    )
    response = llm_with_tools.invoke("What's the status of ticket TCK-1004?")
    for call in response.tool_calls:
        print(f"Model chose tool: {call['name']}({call['args']})")


if __name__ == "__main__":
    main()
