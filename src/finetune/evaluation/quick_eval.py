import time
import json
from finetune.models.inference import chat
from finetune.utils.settings import BASE_DIR

test_questions = [
    "What is variance?",
    "Explain Type I and Type II errors",
    "What is the Central Limit Theorem?",
    "How does ANOVA work?",
    "What is Bayes' theorem?",
]


def run_quick_eval():
    results = []
    for q in test_questions:
        print("=" * 60)
        print("Q:", q)
        t0 = time.time()
        a = chat(q, max_new_tokens=256)
        elapsed = time.time() - t0
        print(f"\nA ({elapsed:.1f}s):", a)
        results.append({"question": q, "answer": a, "elapsed_s": round(elapsed, 1)})

    eval_path = BASE_DIR / "eval_results.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Eval log saved → {eval_path}")
