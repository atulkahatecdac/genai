"""
Demo 10: Tool permissions and scope limiting.

Builds a customer-scoped set of tools via a closure - the customer's name
is captured when the tools are created, never supplied by the model - plus
an "escalate_ticket" action that only returns a preview and requires an
explicit confirm() call before anything actually happens. Every call is
logged.

Setup:
    pip install langchain langchain-openai python-dotenv
    Set OPENAI_API_KEY in the .env file in the parent GenAI folder (shared
    across modules).
"""

import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()

MODEL = "gpt-4o-mini"
TICKETS_PATH = Path(__file__).parent / "assets" / "support_tickets.json"

CALL_LOG = []


def load_tickets():
    return json.loads(TICKETS_PATH.read_text())


def make_tools_for_customer(customer_name: str):
    @tool
    def list_my_tickets() -> str:
        """List the current customer's own support tickets."""
        CALL_LOG.append(("list_my_tickets", {}))
        tickets = [t for t in load_tickets() if t["customer_name"] == customer_name]
        return json.dumps(tickets)

    @tool
    def escalate_ticket(ticket_id: str) -> str:
        """Request escalation of one of the current customer's tickets to high priority."""
        CALL_LOG.append(("escalate_ticket", {"ticket_id": ticket_id}))
        owned = any(t["ticket_id"] == ticket_id and t["customer_name"] == customer_name for t in load_tickets())
        if not owned:
            return json.dumps({"status": "denied", "reason": "ticket does not belong to this customer"})
        return json.dumps(
            {
                "status": "pending_confirm",
                "preview": f"Will escalate {ticket_id} to high priority for {customer_name}.",
            }
        )

    return [list_my_tickets, escalate_ticket]


def confirm_escalation(ticket_id: str) -> str:
    """The actual irreversible action - only called after a human/caller confirms."""
    CALL_LOG.append(("confirm_escalation", {"ticket_id": ticket_id}))
    return f"{ticket_id} escalated to high priority."


def main():
    tools = make_tools_for_customer("Sarah Chen")
    tools_by_name = {t.name: t for t in tools}

    llm = ChatOpenAI(model=MODEL)
    llm_with_tools = llm.bind_tools(tools)

    print("Sarah asks to see her own tickets:")
    response = llm_with_tools.invoke("What tickets do I have open?")
    for call in response.tool_calls:
        print(f"  {call['name']}({call['args']}) -> {tools_by_name[call['name']].invoke(call['args'])}")

    print("\nSarah asks to escalate one of her own tickets:")
    response = llm_with_tools.invoke("Please escalate ticket TCK-1004.")
    for call in response.tool_calls:
        result = tools_by_name[call["name"]].invoke(call["args"])
        print(f"  {call['name']}({call['args']}) -> {result}")
        parsed = json.loads(result)
        if parsed.get("status") == "pending_confirm":
            print(f"  Preview shown to a human reviewer: {parsed['preview']}")
            print(f"  Human confirms -> {confirm_escalation(call['args']['ticket_id'])}")

    print("\nSarah tries to escalate someone else's ticket (TCK-1002, belongs to Miguel):")
    response = llm_with_tools.invoke("Please escalate ticket TCK-1002.")
    for call in response.tool_calls:
        print(f"  {call['name']}({call['args']}) -> {tools_by_name[call['name']].invoke(call['args'])}")

    print(f"\nCall log ({len(CALL_LOG)} entries): {CALL_LOG}")


if __name__ == "__main__":
    main()
