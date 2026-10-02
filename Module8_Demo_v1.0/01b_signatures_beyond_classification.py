"""
Demo 1b: Signatures are not just for classification.

Demo 1 used a Signature to classify a ticket's urgency. This demo runs two
more Signatures on the same CloudSync ticket:
  - Extraction: one Signature with several output fields, filled in one call.
  - Inline: a one-line string Signature ("ticket -> summary"), no class needed.

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
    "Our entire team is locked out of CloudSync Pro right now, we can't "
    "access any files for an important client deadline in an hour."
)


class ExtractTicketDetails(dspy.Signature):
    """Extract key details from a support ticket."""

    ticket: str = dspy.InputField()
    product: str = dspy.OutputField(desc="product name mentioned in the ticket")
    issue: str = dspy.OutputField(desc="the problem, in a few words")
    deadline: str = dspy.OutputField(desc="any time pressure mentioned, or 'none'")


def main():
    dspy.configure(lm=dspy.LM(MODEL))

    print("=== Extraction: several outputs from one Signature ===")
    extractor = dspy.Predict(ExtractTicketDetails)
    details = extractor(ticket=TICKET_TEXT)
    print(f"product:  {details.product}")
    print(f"issue:    {details.issue}")
    print(f"deadline: {details.deadline}")

    print("\n=== Inline Signature: one line, no class ===")
    summarizer = dspy.Predict("ticket -> summary")
    result = summarizer(ticket=TICKET_TEXT)
    print(f"summary: {result.summary}")


if __name__ == "__main__":
    main()
