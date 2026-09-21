"""
Demo 4: Pydantic's three roles in an LLM pipeline - schema, validation, safety.

Uses one Pydantic model, TicketPriority, to show all three roles Pydantic
plays around model output:
1. Schema definition - the model class doubles as the structured-output
   schema handed to the LLM.
2. Output validation - a real model call is validated against it.
3. Runtime safety - a deliberately bad, hand-built response is validated
   too, to show the ValidationError you get to catch instead of a crash.

Setup:
    pip install langchain langchain-openai pydantic python-dotenv
    Set OPENAI_API_KEY in the .env file in the parent GenAI folder (shared
    across modules).
"""

from typing import Literal, cast

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

MODEL = "gpt-4o-mini"

TICKET_TEXT = "This is Miguel. CloudSync keeps failing to sync files over 2GB, very urgent, please help fast."


class TicketPriority(BaseModel):
    issue_type: Literal["billing", "technical", "other"]
    priority: int = Field(ge=1, le=5, description="1 = lowest, 5 = most urgent")


def main():
    print("=== 1. Schema definition ===")
    print("TicketPriority fields:", list(TicketPriority.model_fields))

    print("\n=== 2. Output validation (real model call) ===")
    llm = ChatOpenAI(model=MODEL)
    structured_llm = llm.with_structured_output(TicketPriority)
    result = cast(TicketPriority, structured_llm.invoke(TICKET_TEXT))
    print(f"Validated result: {result}")

    print("\n=== 3. Runtime safety (a deliberately bad response) ===")
    bad_response = {"issue_type": "urgent-ish", "priority": 11}
    try:
        TicketPriority.model_validate(bad_response)
    except ValidationError as exc:
        print(f"Caught ValidationError instead of a crash:\n{exc}")


if __name__ == "__main__":
    main()
