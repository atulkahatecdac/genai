"""
Demo 8d: Multilingual translation example using Gemini, served as a simple
Flask web app.

Same logic as demo 8c, but instead of a terminal loop this version shows a
single plain HTML page (no CSS framework, no JavaScript) with two fields -
text to translate and target language - and a submit button.

Setup:
    pip install google-genai flask python-dotenv
    Set GOOGLE_API_KEY in a .env file (see .env.example).

Run:
    python 08d_gemini_multilingual_translation_datascience_book_flask.py
    Then open http://127.0.0.1:5003 in a browser.
"""

import os

from dotenv import load_dotenv
from flask import Flask, render_template_string, request
from google import genai

load_dotenv()

MODEL = "gemini-2.5-flash"

EXAMPLE_TEXT = (
    "Data science is a field of study that investigates how to collect, manage, "
    "and analyze data of all types in order to retrieve meaningful information."
)
EXAMPLE_CITATION = 'Principles of Data Science (OpenStax), Section 1.1 "What Is Data Science?"'

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

app = Flask(__name__)

PAGE = """
<!doctype html>
<title>Demo 8d: Data Science Book Translator</title>
<h1>Data Science Book Translator</h1>
<p>Example text ({{ citation }}): {{ example_text }}</p>
<form method="post">
    <label for="text">Text to translate:</label><br>
    <textarea id="text" name="text" rows="4" cols="60">{{ text }}</textarea><br>
    <label for="target_language">Target language:</label><br>
    <input type="text" id="target_language" name="target_language" value="{{ target_language }}"><br>
    <input type="submit" value="Translate">
</form>
{% if translation %}
    <h2>Translation ({{ target_language }})</h2>
    <pre>{{ translation }}</pre>
{% endif %}
"""


def translate(text: str, target_language: str) -> str:
    prompt = (
        f"Translate the following text into {target_language}. "
        f"Return only the translation, with no extra commentary.\n\n"
        f"Text: {text}"
    )
    response = client.models.generate_content(model=MODEL, contents=prompt)
    return response.text.strip()


@app.route("/", methods=["GET", "POST"])
def index():
    translation = None
    text = EXAMPLE_TEXT
    target_language = "French"

    if request.method == "POST":
        text = request.form.get("text", "").strip()
        target_language = request.form.get("target_language", "").strip()
        if text and target_language:
            translation = translate(text, target_language)

    return render_template_string(
        PAGE,
        text=text,
        target_language=target_language,
        translation=translation,
        example_text=EXAMPLE_TEXT,
        citation=EXAMPLE_CITATION,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5003)
