"""
Demo 9: ReAct - reasoning and acting in a loop.

A DSPy ReAct agent answers a question about the ticket training dataset by
interleaving Thought -> Action -> Observation steps, calling two small
tools (a keyword search and a counter) until it has enough information to
answer.

Setup:
    pip install dspy python-dotenv
    Set OPENAI_API_KEY in the .env file in the parent GenAI folder (shared
    across modules).
"""

import json
from pathlib import Path

import dspy
from dotenv import load_dotenv

load_dotenv()

MODEL = "openai/gpt-4o-mini"
DATA_PATH = Path(__file__).parent / "assets" / "ticket_training_data.json"


def load_tickets():
    return json.loads(DATA_PATH.read_text())


def search_tickets_by_keyword(keyword: str) -> str:
    """Search training tickets for a keyword and return how many matched, by urgency."""
    tickets = load_tickets()
    matches = [t for t in tickets if keyword.lower() in t["ticket"].lower()]
    by_urgency = {}
    for t in matches:
        by_urgency[t["urgency"]] = by_urgency.get(t["urgency"], 0) + 1
    return f"{len(matches)} ticket(s) mention '{keyword}': {by_urgency}"


def count_tickets_by_urgency(urgency: str) -> str:
    """Count how many training tickets have a given urgency level."""
    tickets = load_tickets()
    count = sum(1 for t in tickets if t["urgency"] == urgency)
    return f"{count} ticket(s) with urgency={urgency}"


class AnswerQuestion(dspy.Signature):
    """Answer a question about the support ticket dataset."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


def main():
    dspy.configure(lm=dspy.LM(MODEL))

    agent = dspy.ReAct(AnswerQuestion, tools=[search_tickets_by_keyword, count_tickets_by_urgency])

    question = "What fraction of tickets that mention the word 'down' are high urgency?"
    result = agent(question=question)

    print(f"Question: {question}")
    print(f"Answer: {result.answer}\n")
    print("Trajectory:")
    for key, value in result.trajectory.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
