"""
Demo 13: End-to-end tool-enabled support agent.

Combines everything from this module into one flow: a Pydantic-validated
tool for ticket lookup, a tool-calling agent that decides when to call it,
and a Pydantic-validated final answer - with a retry layer in case the
agent's final summary doesn't parse cleanly.

Setup:
    pip install langchain langchain-openai pydantic python-dotenv
    Set OPENAI_API_KEY in the .env file in the parent GenAI folder (shared
    across modules).
"""

import json
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.output_parsers import RetryWithErrorOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

MODEL = "gpt-4o-mini"
TICKETS_PATH = Path(__file__).parent / "assets" / "support_tickets.json"


def load_tickets():
    return json.loads(TICKETS_PATH.read_text())


class LookupArgs(BaseModel):
    ticket_id: str = Field(pattern=r"^TCK-\d{4}$", description="Ticket ID, e.g. 'TCK-1001'")


@tool(args_schema=LookupArgs)
def lookup_ticket(ticket_id: str) -> str:
    """Look up a support ticket by its ID."""
    for ticket in load_tickets():
        if ticket["ticket_id"] == ticket_id:
            return json.dumps(ticket)
    return json.dumps({"ticket_id": ticket_id, "error": "not found"})


class AgentAnswer(BaseModel):
    status: Literal["found", "not_found"]
    message: str
    ticket_id: Optional[str] = None


def main():
    llm = ChatOpenAI(model=MODEL)
    tools = [lookup_ticket]

    agent_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a support assistant. Use tools to look up tickets when asked."),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    agent = create_tool_calling_agent(llm, tools, agent_prompt)
    executor = AgentExecutor(agent=agent, tools=tools)

    query = "What's the status of ticket TCK-1004, and who is it for?"
    raw_answer = executor.invoke({"input": query})["output"]
    print(f"Query: {query}")
    print(f"Agent's raw answer: {raw_answer}\n")

    parser = PydanticOutputParser(pydantic_object=AgentAnswer)
    format_prompt = PromptTemplate(
        template="Reformat this answer as structured data.\n{format_instructions}\n\nAnswer: {answer}",
        input_variables=["answer"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    retry_parser = RetryWithErrorOutputParser.from_llm(parser=parser, llm=llm, max_retries=2)

    format_chain = format_prompt | llm
    structured_raw = format_chain.invoke({"answer": raw_answer})

    try:
        result = retry_parser.parse_with_prompt(
            str(structured_raw.content), format_prompt.format_prompt(answer=raw_answer)
        )
        print(f"Structured final answer: {result}")
    except OutputParserException as exc:
        print(f"Giving up after retries ({exc}); using hard fallback.")
        result = AgentAnswer(status="not_found", message="Could not parse agent output.")
        print(f"Fallback result: {result}")


if __name__ == "__main__":
    main()
