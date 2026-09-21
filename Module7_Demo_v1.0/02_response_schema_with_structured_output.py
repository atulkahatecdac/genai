"""
Demo 2: Binding a response schema with LangChain's with_structured_output().

Same two raw customer support messages as demo 1, but this time a Pydantic
schema is bound to the model with with_structured_output(), so every call
returns a typed object with the same fields - no more guessing what keys
the model decided to use.

Setup:
    pip install langchain langchain-openai python-dotenv
    Set OPENAI_API_KEY in the .env file in the parent GenAI folder (shared
    across modules).
"""

from typing import cast

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

MODEL = "gpt-4o-mini"

RAW_TICKETS = [
    "Hi, I'm Sarah. My invoice for CloudSync Pro is wrong - I was charged twice this month.",
    "This is Miguel. CloudSync keeps failing to sync files over 2GB, very urgent, please help fast.",
]


class TicketClassification(BaseModel):
    customer_name: str = Field(description="the customer's first name, if mentioned")
    issue_type: str = Field(description="billing, technical, or other")
    urgency: str = Field(description="low, medium, or high")
    summary: str = Field(description="one-sentence summary of the issue")


def main():
    llm = ChatOpenAI(model=MODEL)
    structured_llm = llm.with_structured_output(TicketClassification)

    for raw in RAW_TICKETS:
        result = cast(TicketClassification, structured_llm.invoke(raw))
        print(f"Raw message: {raw}")
        print(
            f"customer_name={result.customer_name!r} issue_type={result.issue_type!r} "
            f"urgency={result.urgency!r}"
        )
        print(f"summary: {result.summary}\n")


if __name__ == "__main__":
    main()
