"""
Demo 6: Model fine-tuning.

Fine-tunes gpt-4o-mini on a tiny toy dataset that always answers in haiku
form, then compares the base model's response to the fine-tuned model's
response for the same question.

WARNING: this submits a REAL fine-tuning job to OpenAI. It costs a small
amount of money and can take anywhere from several minutes to a few hours to
complete depending on queue load - it will not finish instantly.

Setup:
    pip install openai python-dotenv
    Set OPENAI_API_KEY in the .env file in the parent GenAI folder (shared
    across modules).

Usage:
    First run:  python 06_fine_tuning_job.py
                (uploads the dataset, creates the job, prints the job ID, and
                polls for a few minutes)
    If it's still running when the script exits, set EXISTING_JOB_ID below to
    the printed job ID and re-run to just check status / continue polling.
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE_MODEL = "gpt-4o-mini-2024-07-18"
DATASET_PATH = Path(__file__).parent / "assets" / "fine_tune_dataset.jsonl"
TEST_QUESTION = "What should I eat for breakfast?"

# Set to a job ID (e.g. "ftjob-abc123") to check/resume an existing job
# instead of creating a new one.
EXISTING_JOB_ID = None

SYSTEM_PROMPT = "You are a helpful assistant who always answers in the form of a haiku."

TRAINING_EXAMPLES = [
    ("What should I eat for breakfast?", "Warm bowl of oats steams\nHoney drizzled, morning sun\nQuiet before work"),
    ("How do I learn to code?", "Small steps, one line first\nErrors teach more than success\nPractice every day"),
    ("What's the weather like today?", "Clouds drift overhead now\nCheck the sky before you leave\nBring a coat, just in case"),
    ("Can you recommend a book?", "Pages hold new worlds\nPick the one that calls to you\nStart on any page"),
    ("How do I stay productive?", "Rest before you push\nOne task at a time, unrushed\nEvening brings you peace"),
    ("What's a good workout routine?", "Stretch before you move\nSmall sets build a steady strength\nRest days count as work"),
    ("How do I make new friends?", "Say hello, then wait\nShared moments build quiet trust\nFriendship grows with time"),
    ("What should I watch tonight?", "Choose a story slow\nLet the evening screen unwind\nRest is earned tonight"),
    ("How do I save money?", "Small coins, steady jar\nSkip the thing you don't yet need\nFuture self says thanks"),
    ("What's the best way to travel?", "Pack light, wander far\nMaps forget the best of streets\nGet lost, then get found"),
]


def build_dataset():
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DATASET_PATH, "w") as f:
        for question, answer in TRAINING_EXAMPLES:
            record = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ]
            }
            f.write(json.dumps(record) + "\n")
    print(f"Wrote {len(TRAINING_EXAMPLES)} training examples to {DATASET_PATH}")


def create_job(client):
    with open(DATASET_PATH, "rb") as f:
        uploaded = client.files.create(file=f, purpose="fine-tune")
    print(f"Uploaded training file: {uploaded.id}")

    job = client.fine_tuning.jobs.create(training_file=uploaded.id, model=BASE_MODEL)
    print(f"Created fine-tuning job: {job.id}")
    return job.id


def poll_job(client, job_id, max_checks=10, wait_seconds=30):
    for i in range(max_checks):
        job = client.fine_tuning.jobs.retrieve(job_id)
        print(f"[{i + 1}/{max_checks}] Job status: {job.status}")
        if job.status == "succeeded":
            return job.fine_tuned_model
        if job.status in ("failed", "cancelled"):
            raise RuntimeError(f"Fine-tuning job ended with status: {job.status}")
        time.sleep(wait_seconds)

    print(
        f"\nJob is still running after {max_checks * wait_seconds}s. "
        f'Set EXISTING_JOB_ID = "{job_id}" at the top of this script and re-run to keep checking.'
    )
    return None


def ask(client, model, question):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": question}],
    )
    return response.choices[0].message.content


def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    if EXISTING_JOB_ID:
        job_id = EXISTING_JOB_ID
    else:
        build_dataset()
        job_id = create_job(client)

    fine_tuned_model = poll_job(client, job_id)
    if not fine_tuned_model:
        return

    print(f"\nFine-tuned model ready: {fine_tuned_model}")
    print("\n=== Base model ===")
    print(ask(client, BASE_MODEL, TEST_QUESTION))
    print("\n=== Fine-tuned model ===")
    print(ask(client, fine_tuned_model, TEST_QUESTION))


if __name__ == "__main__":
    main()
