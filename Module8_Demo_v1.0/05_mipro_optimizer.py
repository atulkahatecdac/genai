"""
Demo 5: MIPRO - co-optimizing the instruction text and the few-shot examples.

Goes further than BootstrapFewShot: MIPRO also rewrites the Signature's
instruction text, evaluating candidate phrasings against a held-out
validation set before picking the best (instruction, examples) combination.

Setup:
    pip install dspy python-dotenv
    Set OPENAI_API_KEY in the .env file in the parent GenAI folder (shared
    across modules).

Note: MIPRO makes many more LLM calls than BootstrapFewShot (tens to a few
hundred, depending on settings) - this demo uses auto="light" and a small
dataset to keep it fast and cheap to run.
"""

import json
from pathlib import Path

import dspy
from dotenv import load_dotenv
from dspy.teleprompt import MIPROv2

load_dotenv()

MODEL = "openai/gpt-4o-mini"
DATA_PATH = Path(__file__).parent / "assets" / "ticket_training_data.json"
COMPILED_PATH = Path(__file__).parent / "assets" / "ticket_classifier_mipro.json"


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
    valset = trainset[-4:]  # small held-out slice used only for scoring candidates

    optimizer = MIPROv2(metric=accuracy, auto="light")
    trained = optimizer.compile(
        student=dspy.Predict(ClassifyUrgency),
        trainset=trainset,
        valset=valset,
        requires_permission_to_run=False,
    )

    print("What MIPRO optimized:")
    print(f"  Instruction: {trained.signature.instructions}")
    print(f"  Demos baked in: {len(trained.demos)}")

    trained.save(str(COMPILED_PATH))
    print(f"\nSaved compiled program to {COMPILED_PATH}")


if __name__ == "__main__":
    main()
