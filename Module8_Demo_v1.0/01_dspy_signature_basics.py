"""
Demo 1: DSPy in one paragraph - prompts are generated, not hand-written.

Compares a traditional hand-written prompt string against the DSPy
equivalent: a typed Signature run through dspy.Predict. Same task -
classify a support ticket's urgency - but DSPy generates the actual prompt
text from the Signature, and an optimizer can later tune it automatically
(see demos 4 and 5).

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

TRADITIONAL_PROMPT = f"""
Classify this support ticket's urgency as low, medium, or high.
Ticket: {TICKET_TEXT}
Answer:
"""


class ClassifyUrgency(dspy.Signature):
    """Classify a support ticket's urgency."""

    ticket: str = dspy.InputField()
    urgency: str = dspy.OutputField(desc="low, medium, or high")


def main():
    dspy.configure(lm=dspy.LM(MODEL))

    print("=== Traditional hand-written prompt ===")
    print(TRADITIONAL_PROMPT)

    print("=== DSPy equivalent ===")
    classifier = dspy.Predict(ClassifyUrgency)
    result = classifier(ticket=TICKET_TEXT)
    print(f"urgency: {result.urgency}")
    print(
        "\n(DSPy generated the actual prompt text from the Signature above - "
        "an optimizer can later tune it automatically, see demos 4 and 5.)"
    )


if __name__ == "__main__":
    main()
