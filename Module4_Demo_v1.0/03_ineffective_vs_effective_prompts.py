"""
Demo 3: Ineffective vs. effective prompts.

Runs a vague, underspecified prompt and a tightened, well-specified prompt
on the same summarization task, so you can compare the difference in output
quality and consistency.

Setup:
    pip install openai python-dotenv
    Set OPENAI_API_KEY in the .env file in the parent GenAI folder (shared
    across modules).
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "gpt-4o-mini"

ARTICLE = (
    "Remote work adoption has plateaued over the past year after surging "
    "during the pandemic. A recent survey of 2,000 companies found that 41% "
    "now require at least three days a week in the office, up from 22% two "
    "years ago. Employees cite flexibility and reduced commute time as the "
    "main benefits of remote work, while employers point to collaboration "
    "and mentorship as reasons for wanting people back in the office. "
    "Productivity data remains mixed: some studies show no measurable "
    "difference, while others report a slight dip in output for fully "
    "remote teams beyond their first year."
)

INEFFECTIVE_PROMPT = f"Summarize this.\n\n{ARTICLE}"

EFFECTIVE_PROMPT = f"""Summarize the article below in exactly 3 bullet points:
- One bullet on the current trend
- One bullet on the employee perspective
- One bullet on the employer perspective

Each bullet must be under 20 words and based only on facts stated in the article - do not add outside opinions.

<article>
{ARTICLE}
</article>"""


def call_model(client, prompt):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content


def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    print("=== Ineffective prompt ===")
    print("Prompt: Summarize this. <article>")
    print("Output:\n", call_model(client, INEFFECTIVE_PROMPT))

    print("\n=== Effective prompt ===")
    print("Prompt: 3 labeled bullets, word limit, no outside opinions, delimited <article>")
    print("Output:\n", call_model(client, EFFECTIVE_PROMPT))


if __name__ == "__main__":
    main()
