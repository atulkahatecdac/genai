"""
Demo 8: Validating a compiled program - baseline vs. optimized, and failure analysis.

Compares an uncompiled baseline classifier against a BootstrapFewShot-
compiled version on the same held-out test set, then prints the specific
test cases the compiled version still gets wrong.

Setup:
    pip install dspy python-dotenv
    Set OPENAI_API_KEY in the .env file in the parent GenAI folder (shared
    across modules).
"""

import json
from pathlib import Path

import dspy
from dotenv import load_dotenv
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShot

load_dotenv()

MODEL = "openai/gpt-4o-mini"
DATA_PATH = Path(__file__).parent / "assets" / "ticket_training_data.json"


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


def urgency_match(example, pred, trace=None):
    return pred.urgency.lower() == example.urgency.lower()


def main():
    dspy.configure(lm=dspy.LM(MODEL))

    trainset = load_split("train")
    testset = load_split("test")
    evaluator = Evaluate(devset=testset, metric=urgency_match, num_threads=4, display_progress=False)

    print("=== Baseline (no optimization) ===")
    baseline = dspy.Predict(ClassifyUrgency)
    baseline_result = evaluator(baseline)
    print(f"Baseline accuracy: {baseline_result.score:.1f}%")

    print("\n=== After BootstrapFewShot ===")
    optimizer = BootstrapFewShot(metric=urgency_match, max_bootstrapped_demos=4)
    compiled = optimizer.compile(student=dspy.Predict(ClassifyUrgency), trainset=trainset)
    compiled_result = evaluator(compiled)
    print(f"Compiled accuracy: {compiled_result.score:.1f}%")

    print("\n=== Failure cases (compiled program) ===")
    failures = [(example, pred) for example, pred, score in compiled_result.results if not score]
    if not failures:
        print("  None - the compiled program got every test case right.")
    for example, pred in failures:
        print(f"  Ticket: {example.ticket[:70]}...")
        print(f"  Expected: {example.urgency}  Got: {pred.urgency}")


if __name__ == "__main__":
    main()
