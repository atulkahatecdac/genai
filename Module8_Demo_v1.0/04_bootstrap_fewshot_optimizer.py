"""
Demo 4: BootstrapFewShot - automatic few-shot example selection.

Runs a Predict-based urgency classifier on the training split of the
ticket dataset, keeps the traces that scored correctly against a simple
exact-match metric, and bakes the best ones in as few-shot examples - no
manual example curation.

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
COMPILED_PATH = Path(__file__).parent / "assets" / "ticket_classifier_bootstrap.json"


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

    trainset = load_split("train")
    print(f"Training examples: {len(trainset)}")

    optimizer = BootstrapFewShot(metric=accuracy, max_bootstrapped_demos=4, max_labeled_demos=8)
    trained = optimizer.compile(student=dspy.Predict(ClassifyUrgency), trainset=trainset)

    print(f"\nBootstrapped {len(trained.demos)} demo(s) into the prompt:")
    for demo in trained.demos:
        print(f"  {demo['ticket'][:60]}... -> {demo['urgency']}")

    trained.save(str(COMPILED_PATH))
    print(f"\nSaved compiled program to {COMPILED_PATH}")


if __name__ == "__main__":
    main()
