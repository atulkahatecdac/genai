"""
Demo 8: Routing user intent across multiple tools.

Three different tools over the same dummy support ticket database - a
status lookup, an issue-type counter, and a per-customer ticket list - so
the model has to pick the right one based on each query.

Setup:
    pip install langchain langchain-openai python-dotenv
    Set OPENAI_API_KEY in the .env file in the parent GenAI folder (shared
    across modules).
"""

import json
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()

MODEL = "gpt-4o-mini"
TICKETS_PATH = Path(__file__).parent / "assets" / "support_tickets.json"


def load_tickets():
    return json.loads(TICKETS_PATH.read_text())


@tool
def get_ticket_status(ticket_id: str) -> str:
    """Look up a single support ticket's status by its ticket ID, e.g. 'TCK-1001'."""
    for ticket in load_tickets():
        if ticket["ticket_id"] == ticket_id:
            return f"{ticket_id}: {ticket['status']}"
    return f"{ticket_id} not found"


@tool
def count_tickets_by_issue_type(issue_type: str) -> str:
    """Count how many tickets exist for a given issue type: billing, technical, or other."""
    tickets = load_tickets()
    count = sum(1 for t in tickets if t["issue_type"] == issue_type)
    return f"{count} ticket(s) with issue_type={issue_type}"


@tool
def list_open_tickets_for_customer(customer_name: str) -> str:
    """List open tickets for a specific customer by name."""
    tickets = load_tickets()
    open_ids = [t["ticket_id"] for t in tickets if t["customer_name"] == customer_name and t["status"] == "open"]
    return f"Open tickets for {customer_name}: {open_ids or 'none'}"


QUERIES = [
    "What's the status of ticket TCK-1004?",
    "How many billing tickets do we have?",
    "Does Sarah Chen have any open tickets?",
]


def main():
    llm = ChatOpenAI(model=MODEL)
    tools = [get_ticket_status, count_tickets_by_issue_type, list_open_tickets_for_customer]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful support assistant."),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

    for query in QUERIES:
        print(f"Query: {query}")
        result = executor.invoke({"input": query})
        print(f"Answer: {result['output']}\n")


if __name__ == "__main__":
    main()
