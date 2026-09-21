"""
Demo 2: DSPy modules and signatures.

A Signature declares a typed input/output contract; a Module wraps it with
a reasoning strategy and composes cleanly as plain Python. Here,
TicketProcessor summarizes a support ticket and classifies its urgency in
one call, using ChainOfThought so the model reasons before answering.

Setup:
    pip install dspy python-dotenv
    Set OPENAI_API_KEY in the .env file in the parent GenAI folder (shared
    across modules).
"""

import dspy
from dotenv import load_dotenv

load_dotenv()

MODEL = "openai/gpt-4o-mini"

TICKET_TEXT = (
    "CloudSync Pro keeps failing to sync files over 2GB, it's slowing "
    "down my work but I can still use smaller files."
)


class SummarizeAndClassify(dspy.Signature):
    """Summarize a support ticket and classify its urgency."""

    ticket: str = dspy.InputField()
    summary: str = dspy.OutputField(desc="one-sentence summary")
    urgency: str = dspy.OutputField(desc="low / medium / high")


class TicketProcessor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.process = dspy.ChainOfThought(SummarizeAndClassify)

    def forward(self, ticket):
        return self.process(ticket=ticket)


def main():
    dspy.configure(lm=dspy.LM(MODEL))

    proc = TicketProcessor()
    result = proc(ticket=TICKET_TEXT)

    print(f"Ticket: {TICKET_TEXT}\n")
    print(f"Summary: {result.summary}")
    print(f"Urgency: {result.urgency}")
    print(f"\nReasoning (added automatically by ChainOfThought): {result.reasoning}")


if __name__ == "__main__":
    main()
