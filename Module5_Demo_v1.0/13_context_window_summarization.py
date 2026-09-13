"""
Demo 13: Context window summarization strategies.

Three techniques for keeping large inputs within a model's context window:
1. Prune irrelevant input - drop conversation turns unrelated to the
   current question before sending anything to the model.
2. Summarization prompts - compress older conversation turns into a short
   summary instead of sending the full transcript every time.
3. Chunk large documents with overlap and merge results later - split a
   long document into overlapping chunks, summarize each chunk, then merge
   the chunk summaries into one final summary (map-reduce style).

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

CONVERSATION_TURNS = [
    "User: What's your return policy for electronics?",
    "Assistant: Electronics can be returned within 30 days with a receipt.",
    "User: By the way, do you sell gift cards?",
    "Assistant: Yes, gift cards are available in $25, $50, and $100 denominations.",
    "User: Great, what about the weather forecast for this weekend?",
    "Assistant: I don't have weather data, but I can help with store-related questions.",
    "User: Okay, back to returns - does the 30 day policy apply to opened items?",
]

CURRENT_QUESTION = "Does the 30 day return policy apply to opened electronics?"

LONG_DOCUMENT = (
    "Quarterly Report - Product Division\n\n"
    "Section 1: Sales grew 12% year over year, driven mainly by strong demand in the "
    "north region. The sales team attributes this to the new partner referral program "
    "launched in January, which contributed roughly a third of new deals this quarter.\n\n"
    "Section 2: Customer support ticket volume rose 8%, largely due to onboarding "
    "questions from the referral-driven signups. Average resolution time held steady "
    "at 6 hours despite the volume increase, thanks to the new triage system.\n\n"
    "Section 3: Engineering shipped 4 major features this quarter, including the "
    "long-requested bulk export tool. Two features slipped to next quarter due to "
    "unresolved performance issues under high load.\n\n"
    "Section 4: Looking ahead, the team plans to expand the referral program to the "
    "south region and invest further in the triage system to keep resolution times "
    "flat as volume continues to grow."
)


def call_model(client, prompt):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content


def prune_irrelevant(turns, question):
    question_keywords = {word.lower().strip(".,?") for word in question.split() if len(word) > 3}
    relevant = []
    for turn in turns:
        turn_keywords = {word.lower().strip(".,?") for word in turn.split() if len(word) > 3}
        if question_keywords & turn_keywords:
            relevant.append(turn)
    return relevant


def summarize_history(client, turns):
    transcript = "\n".join(turns)
    prompt = f"Summarize this conversation so far in 2 sentences, keeping only facts relevant to future questions:\n\n{transcript}"
    return call_model(client, prompt)


def chunk_text(text, chunk_size=400, overlap=80):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def summarize_chunks_and_merge(client, chunks):
    chunk_summaries = []
    for i, chunk in enumerate(chunks, start=1):
        summary = call_model(client, f"Summarize this excerpt in 1 sentence:\n\n{chunk}")
        chunk_summaries.append(summary)
        print(f"  Chunk {i}/{len(chunks)} summary: {summary}")

    merge_prompt = "Combine these partial summaries into one coherent final summary:\n\n" + "\n".join(chunk_summaries)
    return call_model(client, merge_prompt)


def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    print("=== 1. Prune irrelevant input ===")
    print(f"Full history: {len(CONVERSATION_TURNS)} turns")
    pruned = prune_irrelevant(CONVERSATION_TURNS, CURRENT_QUESTION)
    print(f"Pruned to {len(pruned)} relevant turns:")
    for turn in pruned:
        print(f"  {turn}")

    print("\n=== 2. Summarization prompt for older turns ===")
    summary = summarize_history(client, CONVERSATION_TURNS)
    print(f"Conversation summary: {summary}")
    answer = call_model(client, f"Context: {summary}\n\nQuestion: {CURRENT_QUESTION}")
    print(f"Answer using summary as context: {answer}")

    print("\n=== 3. Chunk large document with overlap, then merge ===")
    chunks = chunk_text(LONG_DOCUMENT)
    print(f"Split document ({len(LONG_DOCUMENT)} chars) into {len(chunks)} overlapping chunks")
    final_summary = summarize_chunks_and_merge(client, chunks)
    print(f"\nFinal merged summary:\n{final_summary}")


if __name__ == "__main__":
    main()
