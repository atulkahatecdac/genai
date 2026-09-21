"""
Demo 6: Compiling, inspecting, and reloading a DSPy program.

Compiles a fresh program, inspects exactly which examples the optimizer
selected, saves it, then reloads it into a brand-new module instance
without touching the optimizer again - the pattern you'd use in
production.

Setup:
    pip install dspy python-dotenv
    Set OPENAI_API_KEY in the .env file in the parent GenAI folder (shared
    across modules).
"""

import json
from pathlib import Path

import dspy
from dotenv import load_dotenv
from dspy.teleprompt import BootstrapFewShot

load_dotenv()

MODEL = "openai/gpt-4o-mini"
DATA_PATH = Path(__file__).parent / "assets" / "ticket_training_data.json"
COMPILED_PATH = Path(__file__).parent / "assets" / "ticket_classifier_compiled.json"


class ClassifyUrgency(dspy.Signature):
    """Classify a support ticket's urgency."""

    ticket: str = dspy.InputField()
    urgency: str = dspy.OutputField(desc="low, medium, or high")


def load_split(split):
    rows = json.loads(DATA_PATH.read_text())
    return [
        dspy.Example(ticket=row["ticket"], urgency=row["urgency"]).with_inputs("ticket")
        for row in rows
        if row["split"] == split
    ]


def accuracy(example, pred, trace=None):
    return pred.urgency.lower() == example.urgency.lower()


def main():
    dspy.configure(lm=dspy.LM(MODEL))

    print("=== Compile ===")
    optimizer = BootstrapFewShot(metric=accuracy, max_bootstrapped_demos=4)
    compiled = optimizer.compile(student=dspy.Predict(ClassifyUrgency), trainset=load_split("train"))

    print("=== Inspect what was chosen ===")
    for demo in compiled.demos:
        print(f"  Input: {demo['ticket'][:60]}...")
        print(f"  Output: {demo['urgency']}")
        print("  ---")

    compiled.save(str(COMPILED_PATH))
    print(f"Saved to {COMPILED_PATH}\n")

    print("=== Reload in a fresh module, no optimizer involved ===")
    prod = dspy.Predict(ClassifyUrgency)
    prod.load(str(COMPILED_PATH))

    test_ticket = "Could you tell me whether CloudSync Basic supports two-factor authentication?"
    result = prod(ticket=test_ticket)
    print(f"Ticket: {test_ticket}")
    print(f"Urgency (from reloaded program, no LLM calls to the optimizer): {result.urgency}")


if __name__ == "__main__":
    main()
