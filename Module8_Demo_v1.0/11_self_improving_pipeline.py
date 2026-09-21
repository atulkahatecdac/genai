"""
Demo 11: A self-improving pipeline - log, accumulate, recompile, gate.

Simulates the production loop from the deck: predictions get logged, the
test split stands in for "newly accumulated" production examples, the
program is recompiled, and the new version is only "deployed" if it beats
the current one on a fixed test set by a minimum margin.

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
PROD_PATH = Path(__file__).parent / "assets" / "ticket_classifier_prod.json"
IMPROVEMENT_THRESHOLD = 2.0  # percentage points

PRODUCTION_LOG = []


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


def log_prediction(ticket, predicted_urgency, true_urgency):
    PRODUCTION_LOG.append({"ticket": ticket, "predicted": predicted_urgency, "true": true_urgency})


def recompile_if_improved(current_program, trainset, evaluator):
    current_score = evaluator(current_program).score
    print(f"Current production accuracy: {current_score:.1f}%")

    optimizer = BootstrapFewShot(metric=urgency_match, max_bootstrapped_demos=4)
    candidate = optimizer.compile(student=dspy.Predict(ClassifyUrgency), trainset=trainset)
    candidate_score = evaluator(candidate).score
    print(f"Candidate accuracy:          {candidate_score:.1f}%")

    if candidate_score > current_score + IMPROVEMENT_THRESHOLD:
        candidate.save(str(PROD_PATH))
        print(f"Deployed new version (+{candidate_score - current_score:.1f} points) -> {PROD_PATH}")
        return candidate
    print("Not enough improvement - keeping current production version.")
    return current_program


def main():
    dspy.configure(lm=dspy.LM(MODEL))

    trainset = load_split("train")
    testset = load_split("test")
    evaluator = Evaluate(devset=testset, metric=urgency_match, num_threads=4, display_progress=False)

    print("=== Simulating production traffic and logging ===")
    current_program = dspy.Predict(ClassifyUrgency)
    for example in testset[:2]:
        pred = current_program(ticket=example.ticket)
        log_prediction(example.ticket, pred.urgency, example.urgency)
    print(f"Logged {len(PRODUCTION_LOG)} production prediction(s).\n")

    print("=== Weekly recompile check ===")
    recompile_if_improved(current_program, trainset, evaluator)


if __name__ == "__main__":
    main()
