"""
Demo 7d: Q&A over a document example using Ollama (Qwen model), served as a
simple Flask web app.

Same textbook excerpt and logic as demo 7c, but instead of a terminal loop
this version shows a single plain HTML page (no CSS framework, no
JavaScript) where you type a question and submit a form to get an answer.

Setup:
    1. Install Ollama from https://ollama.com
    2. Pull a model:  ollama pull qwen2.5
    3. pip install ollama flask

Run:
    python 07d_ollama_qa_example_datascience_book_flask.py
    Then open http://127.0.0.1:5002 in a browser.
"""

from pathlib import Path

import ollama
from flask import Flask, render_template_string, request

MODEL = "qwen2.5"
DOCUMENT_PATH = Path(__file__).parent / "assets" / "datascience_book_excerpt.txt"

document_text = DOCUMENT_PATH.read_text(encoding="utf-8")

app = Flask(__name__)

PAGE = """
<!doctype html>
<title>Demo 7d: Data Science Book Q&A</title>
<h1>Data Science Book Q&A</h1>
<p>Source: {{ source_name }} - Ollama model: {{ model }}</p>
<form method="post">
    <label for="question">Your question:</label><br>
    <input type="text" id="question" name="question" size="60" value="{{ question }}">
    <input type="submit" value="Ask">
</form>
{% if answer %}
    <h2>Answer</h2>
    <pre>{{ answer }}</pre>
{% endif %}
"""


def ask(question: str) -> str:
    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the user's question using ONLY the information in the "
                    "provided document. If the answer isn't in the document, say so."
                ),
            },
            {
                "role": "user",
                "content": f"Document:\n{document_text}\n\nQuestion: {question}",
            },
        ],
    )
    return response["message"]["content"]


@app.route("/", methods=["GET", "POST"])
def index():
    answer = None
    question = "What are the five steps of the data science cycle?"

    if request.method == "POST":
        question = request.form.get("question", "").strip()
        if question:
            answer = ask(question)

    return render_template_string(
        PAGE,
        question=question,
        answer=answer,
        source_name=DOCUMENT_PATH.name,
        model=MODEL,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5002)
