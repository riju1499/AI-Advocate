
import csv
import os
import datetime
import json

import rag_pipeline
from rag_pipeline import llm as judge_llm, rag_retriever, llm, rag_with_context

OUT_CSV = "data/rag_evaluation_log.csv"


def judge_against_context(question: str, answer: str, context: str):
    """
    Use the same LLM as a 'judge' to score correctness & faithfulness
    based only on the retrieved context (which comes from your PDFs).
    """
    prompt = f"""You are evaluating a legal RAG system for Nepali law.

Question:
{question}

Retrieved legal context (from official Nepali law PDFs):
{context}

Assistant's answer:
{answer}

Evaluate:

1. correctness (0-2):
   - 0: Answer is wrong or does not address the question.
   - 1: Answer is partially correct or incomplete.
   - 2: Answer is mostly or fully correct.

2. faithfulness (0-2):
   - 0: Answer contradicts the context or uses facts not present in context.
   - 1: Answer is partly supported by the context, but some parts are vague or may be hallucinated.
   - 2: Answer is clearly supported by the context and does not invent new legal claims.

Return ONLY a JSON object, no explanation, like:
{{"correctness": 2, "faithfulness": 1}}
"""
    resp = judge_llm.invoke(prompt)
    try:
        scores = json.loads(resp.content)
    except Exception:
        scores = {"correctness": 0, "faithfulness": 0}
    return scores


def main():
    os.makedirs("data", exist_ok=True)
    file_exists = os.path.exists(OUT_CSV)

    with open(OUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Write header if file is new
        if not file_exists:
            writer.writerow([
                "timestamp",
                "question",
                "arena_used",
                "answer",
                "context_snippet",
                "correctness",
                "faithfulness",
            ])

        while True:
            q = input("\nEnter a question to evaluate (or 'q' to quit): ")
            if q.lower() == "q":
                break

            arena = "All (auto)"  # you can change for fixed-Act testing

            answer, context = rag_with_context(
                q,
                rag_retriever,
                llm,
                top_k=6,
                arena=arena,
            )

            print("\n--- ANSWER ---\n", answer)
            print("\n--- CONTEXT (first 500 chars) ---\n", context[:500])

            if not context:
                print("\nNo context; skipping LLM judge.")
                correctness = 0
                faithfulness = 0
            else:
                scores = judge_against_context(q, answer, context)
                correctness = scores.get("correctness", 0)
                faithfulness = scores.get("faithfulness", 0)

            print("\nScores -> Correctness:", correctness, "Faithfulness:", faithfulness)

            timestamp = datetime.datetime.now().isoformat(timespec="seconds")
            writer.writerow([
                timestamp,
                q,
                arena,
                answer,
                context[:300],  # store a short snippet to keep CSV manageable
                correctness,
                faithfulness,
            ])
            f.flush()
            print(f"Logged evaluation to {OUT_CSV}")


if __name__ == "__main__":
    main()