"""
Demo 10: DSPy with multiple LLMs in one pipeline.

Configures a cheap default model globally, then overrides it with a
stronger model for just the expensive step of a two-stage pipeline - cheap
intent classification, then higher-quality response drafting.

Setup:
    pip install dspy python-dotenv
    Set OPENAI_API_KEY in the .env file in the parent GenAI folder (shared
    across modules).
"""

import dspy
from dotenv import load_dotenv

load_dotenv()

CHEAP_MODEL = "openai/gpt-4o-mini"
STRONG_MODEL = "openai/gpt-4o"

TICKET_TEXT = (
    "My CloudSync Pro subscription was cancelled without warning and I "
    "lost access to critical files before a client meeting."
)


class ClassifyIntent(dspy.Signature):
    """Classify a support ticket into a category."""

    ticket: str = dspy.InputField()
    category: str = dspy.OutputField(desc="billing, technical, or other")


class DraftResponse(dspy.Signature):
    """Draft an empathetic support reply to a customer ticket."""

    ticket: str = dspy.InputField()
    category: str = dspy.InputField()
    reply: str = dspy.OutputField()


class SupportPipeline(dspy.Module):
    def __init__(self, strong_lm):
        super().__init__()
        self.classify = dspy.Predict(ClassifyIntent)
        self.draft = dspy.ChainOfThought(DraftResponse)
        self.strong_lm = strong_lm

    def forward(self, ticket):
        intent = self.classify(ticket=ticket)
        with dspy.context(lm=self.strong_lm):
            response = self.draft(ticket=ticket, category=intent.category)
        return dspy.Prediction(category=intent.category, reply=response.reply)


def main():
    cheap_lm = dspy.LM(CHEAP_MODEL)
    strong_lm = dspy.LM(STRONG_MODEL)

    dspy.configure(lm=cheap_lm)  # global default: cheap model

    pipeline = SupportPipeline(strong_lm=strong_lm)
    result = pipeline(ticket=TICKET_TEXT)

    print(f"Ticket: {TICKET_TEXT}\n")
    print(f"Category (classified with {CHEAP_MODEL}): {result.category}")
    print(f"Reply (drafted with {STRONG_MODEL}):\n{result.reply}")


if __name__ == "__main__":
    main()
