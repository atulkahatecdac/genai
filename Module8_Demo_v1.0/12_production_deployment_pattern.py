"""
Demo 12: Production deployment - save once, load at startup, serve many requests.

Compiles a program once (the expensive, offline step), saves it as a
versioned JSON file, then simulates a running server: load the compiled
program once at "startup" and reuse it across multiple incoming requests,
logging each prediction along the way.

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
MODEL_VERSION = "ticket_classifier_v1"
COMPILED_PATH = Path(__file__).parent / "assets" / f"{MODEL_VERSION}.json"

REQUEST_LOG = []


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


def compile_and_save():
    """One-time offline step - would normally run in a training job, not the server."""
    optimizer = BootstrapFewShot(metric=urgency_match, max_bootstrapped_demos=4)
    compiled = optimizer.compile(student=dspy.Predict(ClassifyUrgency), trainset=load_split("train"))
    compiled.save(str(COMPILED_PATH))
    print(f"Compiled and saved {MODEL_VERSION} to {COMPILED_PATH}")


def handle_request(program, ticket_text: str):
    result = program(ticket=ticket_text)
    REQUEST_LOG.append({"input": ticket_text, "output": result.urgency, "model_version": MODEL_VERSION})
    return result


def main():
    dspy.configure(lm=dspy.LM(MODEL))

    if not COMPILED_PATH.exists():
        print("No compiled program found - running the one-time offline compile step first.\n")
        compile_and_save()

    print("\n=== Server startup: load compiled program once ===")
    program = dspy.Predict(ClassifyUrgency)
    program.load(str(COMPILED_PATH))

    print("\n=== Serving requests (reusing the loaded program, no recompiling) ===")
    incoming_tickets = [
        "Wondering if there's a referral program for CloudSync Pro customers.",
        "Our CloudSync Team workspace just went down for the whole company.",
    ]
    for ticket_text in incoming_tickets:
        result = handle_request(program, ticket_text)
        print(f"  Ticket: {ticket_text[:60]}...  -> urgency={result.urgency}")

    print(f"\nRequest log ({len(REQUEST_LOG)} entries): {REQUEST_LOG}")


if __name__ == "__main__":
    main()
