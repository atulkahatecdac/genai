"""
Demo 3: Extracting structured fields from a messy support ticket email.

Pulls specific fields out of an unstructured, email-style customer message
into a typed Ticket object, leaving fields as None when the text doesn't
mention them.

Setup:
    pip install langchain langchain-openai python-dotenv
    Set OPENAI_API_KEY in the .env file in the parent GenAI folder (shared
    across modules).
"""

from typing import Optional, cast

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

MODEL = "gpt-4o-mini"

TICKET_EMAIL = """
Subject: Ongoing sync issue

Hi team,

This is James O'Brien writing in again. Our CloudSync Team plan keeps
dropping large uploads and I haven't heard back since Tuesday. Could
someone take a look? Not urgent, but it's been open a while now.

Thanks,
James
"""


class Ticket(BaseModel):
    customer_name: Optional[str] = Field(default=None, description="the customer's full name, if mentioned")
    product_mentioned: Optional[str] = Field(default=None, description="the product or plan name, if mentioned")
    issue_type: str = Field(description="billing, technical, or other")
    urgency: str = Field(description="low, medium, or high")


def main():
    llm = ChatOpenAI(model=MODEL)
    extractor = llm.with_structured_output(Ticket)

    print(f"Raw email:\n{TICKET_EMAIL}")
    result = cast(
        Ticket,
        extractor.invoke(f"Extract the following fields from this support email:\n\n{TICKET_EMAIL}"),
    )
    print("Extracted:")
    print(f"  customer_name:     {result.customer_name}")
    print(f"  product_mentioned: {result.product_mentioned}")
    print(f"  issue_type:        {result.issue_type}")
    print(f"  urgency:           {result.urgency}")


if __name__ == "__main__":
    main()
