"""
Demo 3: Predict, ChainOfThought, and ReAct - three reasoning strategies for one Signature.

Runs the exact same urgency-classification Signature through DSPy's three
core modules and prints what each one returns:
- Predict: one LLM call, straight to the answer.
- ChainOfThought: adds a reasoning field before the answer.
- ReAct: can call a tool (a keyword checker here) before answering.

Setup:
    pip install dspy python-dotenv
    Set OPENAI_API_KEY in the .env file in the parent GenAI folder (shared
    across modules).
"""

import dspy
from dotenv import load_dotenv

load_dotenv()

MODEL = "openai/gpt-4o-mini"

TICKET_TEXT = "The whole company's CloudSync Team workspace has been down for two hours, nobody can work."

URGENT_KEYWORDS = ["down", "locked out", "lost", "corrupt", "security", "immediately", "disaster"]


class ClassifyUrgency(dspy.Signature):
    """Classify a support ticket's urgency."""

    ticket: str = dspy.InputField()
    urgency: str = dspy.OutputField(desc="low, medium, or high")


def check_urgent_keywords(text: str) -> str:
    """Check the ticket text for known urgent keywords like 'down' or 'lost'."""
    found = [w for w in URGENT_KEYWORDS if w in text.lower()]
    return f"Found urgent keywords: {found}" if found else "No urgent keywords found"


def main():
    dspy.configure(lm=dspy.LM(MODEL))

    print(f"Ticket: {TICKET_TEXT}\n")

    predict_result = dspy.Predict(ClassifyUrgency)(ticket=TICKET_TEXT)
    print(f"Predict:        urgency={predict_result.urgency}")

    cot_result = dspy.ChainOfThought(ClassifyUrgency)(ticket=TICKET_TEXT)
    print(f"ChainOfThought: urgency={cot_result.urgency}")
    print(f"  reasoning: {cot_result.reasoning}")

    react = dspy.ReAct(ClassifyUrgency, tools=[check_urgent_keywords])
    react_result = react(ticket=TICKET_TEXT)
    print(f"ReAct:          urgency={react_result.urgency}")
    print(f"  trajectory steps: {len(react_result.trajectory) // 4}")


if __name__ == "__main__":
    main()
