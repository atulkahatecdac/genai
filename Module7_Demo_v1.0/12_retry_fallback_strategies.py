"""
Demo 12: Layered retry and fallback for structured output.

Parses a ticket resolution summary with a PydanticOutputParser wrapped in
a RetryWithErrorOutputParser, which automatically re-prompts the model on
a validation failure - and falls back to a safe default object if it still
can't get valid output.

Setup:
    pip install langchain langchain-openai pydantic python-dotenv
    Set OPENAI_API_KEY in the .env file in the parent GenAI folder (shared
    across modules).
"""

from typing import Literal

from dotenv import load_dotenv
from langchain.output_parsers import RetryWithErrorOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

load_dotenv()

MODEL = "gpt-4o-mini"

TICKET_TEXT = "Ticket TCK-1006 - refund issued for the duplicate CloudSync Pro charge, customer confirmed receipt."


class TicketResolution(BaseModel):
    ticket_id: str
    resolution_status: Literal["resolved", "unresolved"]
    summary: str


def main():
    llm = ChatOpenAI(model=MODEL)
    parser = PydanticOutputParser(pydantic_object=TicketResolution)

    prompt = PromptTemplate(
        template="Summarize this support note as structured data.\n{format_instructions}\n\nNote: {note}",
        input_variables=["note"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    retry_parser = RetryWithErrorOutputParser.from_llm(parser=parser, llm=llm, max_retries=2)

    chain = prompt | llm
    raw_output = chain.invoke({"note": TICKET_TEXT})

    print(f"Note: {TICKET_TEXT}")
    print(f"Raw model output: {raw_output.content}\n")

    try:
        result = retry_parser.parse_with_prompt(str(raw_output.content), prompt.format_prompt(note=TICKET_TEXT))
        print(f"Parsed result: {result}")
    except OutputParserException as exc:
        print(f"Giving up after retries ({exc}); using hard fallback.")
        result = TicketResolution(ticket_id="unknown", resolution_status="unresolved", summary="parse failed")
        print(f"Fallback result: {result}")


if __name__ == "__main__":
    main()
