import random
from finetune.data.dataset_loader import ALL_QA


def search_qa(query="", qtype=None, difficulty=None, limit=10):
    """Filter ALL_QA and print matches. CLI replacement for the Gradio search tab."""
    results = ALL_QA
    if qtype:
        results = [q for q in results if q.get("type") == qtype]
    if difficulty:
        results = [q for q in results if q.get("difficulty") == difficulty]
    if query:
        ql = query.lower()
        results = [q for q in results
                   if ql in q["question"].lower() or ql in q["answer"].lower()]

    print(f"{min(len(results), limit)} of {len(results)} matches (total {len(ALL_QA)})\n")
    for q in results[:limit]:
        print(f"[{q.get('type','?')}/{q.get('difficulty','?')}]")
        print(f"Q: {q['question']}")
        print(f"A: {q['answer'][:300]}{'…' if len(q['answer'])>300 else ''}")
        print("-" * 60)

def random_quiz(difficulty=None):
    pool = ALL_QA
    if difficulty:
        pool = [q for q in pool if q.get("difficulty") == difficulty]
    if not pool:
        return None
    q = random.choice(pool)
    print(f"Q: {q['question']}\n\nA: {q['answer']}")
    return q
