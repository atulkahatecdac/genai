"""
Demo 6d: Text summarization example using OpenAI, served as a simple Flask
web app.

Same textbook excerpt and logic as demo 6c, but instead of a terminal loop
this version shows a single plain HTML page (no CSS framework, no
JavaScript) where you type how you'd like the document summarized and
submit a form.

Setup:
    pip install openai flask python-dotenv
    Set OPENAI_API_KEY in a .env file (see .env.example).

Run:
    python 06d_openai_text_summarization_datascience_book_flask.py
    Then open http://127.0.0.1:5001 in a browser.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, render_template_string, request
from openai import OpenAI

load_dotenv()

MODEL = "gpt-4o-mini"
DOCUMENT_PATH = Path(__file__).parent / "assets" / "datascience_book_excerpt.txt"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
document_text = DOCUMENT_PATH.read_text(encoding="utf-8")

app = Flask(__name__)

PAGE = """
<!doctype html>
<title>Demo 6d: Data Science Book Summarizer</title>
<h1>Data Science Book Summarizer</h1>
<p>Source: {{ source_name }} ({{ source_len }} chars) - OpenAI model: {{ model }}</p>
<form method="post">
    <label for="instruction">How should the document be summarized?</label><br>
    <input type="text" id="instruction" name="instruction" size="60"
           value="{{ instruction }}">
    <input type="submit" value="Summarize">
</form>
{% if summary %}
    <h2>Summary</h2>
    <pre>{{ summary }}</pre>
{% endif %}
"""


def summarize(instruction: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You summarize the provided textbook excerpt according to the "
                    "user's instructions. Base your summary only on the excerpt."
                ),
            },
            {
                "role": "user",
                "content": f"Document:\n{document_text}\n\nInstruction: {instruction}",
            },
        ],
        temperature=0.3,
        max_tokens=400,
    )
    return response.choices[0].message.content or ""


@app.route("/", methods=["GET", "POST"])
def index():
    summary = None
    instruction = "Summarize in 5 bullet points"

    if request.method == "POST":
        instruction = request.form.get("instruction", "").strip()
        if instruction:
            summary = summarize(instruction)

    return render_template_string(
        PAGE,
        instruction=instruction,
        summary=summary,
        source_name=DOCUMENT_PATH.name,
        source_len=len(document_text),
        model=MODEL,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5001)
