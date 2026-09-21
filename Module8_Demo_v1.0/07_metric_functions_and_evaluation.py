"""
Demo 7: Writing metric functions and evaluating with dspy.Evaluate.

Two metrics - exact match for the urgency classifier, and token-level F1
for a free-text extraction task - then dspy.Evaluate runs the urgency
classifier over the held-out test split and reports an aggregate score.

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
    """Exact-match metric: did the predicted urgency match the label?"""
    return pred.urgency.lower() == example.urgency.lower()


def token_f1(gold_text, pred_text):
    """F1 metric over shared words - useful for free-text extraction tasks."""
    gold = set(gold_text.lower().split())
    pred = set(pred_text.lower().split())
    if not pred:
        return 0.0
    precision = len(gold & pred) / len(pred)
    recall = len(gold & pred) / len(gold)
    return 2 * precision * recall / (precision + recall + 1e-9)


def main():
    dspy.configure(lm=dspy.LM(MODEL))

    print("Token F1 sanity check (no model call needed):")
    print(f"  {token_f1('billing issue with duplicate charge', 'duplicate billing charge issue'):.2f}")

    testset = load_split("test")
    classifier = dspy.Predict(ClassifyUrgency)

    evaluator = Evaluate(devset=testset, metric=urgency_match, num_threads=4, display_progress=True)
    result = evaluator(classifier)
    print(f"\nTest accuracy: {result.score:.1f}%")


if __name__ == "__main__":
    main()
