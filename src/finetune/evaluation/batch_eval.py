import json
import time
from typing import List, Dict, Any
from finetune.models.inference import chat


def ask_finetuned(question: str) -> str:
    """
    Ask your fine-tuned model.
    Automatically uses Ollama if available, else local PEFT model.
    """
    return chat(question, max_new_tokens=512, temperature=0.2)


def process_qa_file(
    input_path: str,
    output_path: str,
) -> List[Dict[str, Any]]:

    def load_data(path: str) -> List[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # supports:
        # [{"question": ...}]
        # OR {"questions": [...]}
        if isinstance(data, dict):
            return data.get("questions", [])

        return data

    def process_item(
        item: Dict[str, Any],
        i: int,
        total: int
    ) -> Dict[str, Any] | None:

        question = item.get("question", "").strip()

        if not question:
            print(f"⚠️ Skipping empty question at index {i}")
            return None

        print(f"🔍 Processing ({i}/{total}): {question[:60]}...")

        try:
            t0 = time.time()

            answer = ask_finetuned(question)

            elapsed = round(time.time() - t0, 1)

        except Exception as e:
            print(f"❌ Error at question {i}: {e}")
            answer = f"ERROR: {e}"
            elapsed = -1

        return {
            "type": item.get("type", ""),
            "difficulty": item.get("difficulty", ""),
            "question": question,
            "answer": answer,
            "elapsed_s": elapsed,
        }

    # ---------------------------------------------------
    # Main flow
    # ---------------------------------------------------

    data = load_data(input_path)
    total = len(data)

    results = [
        result
        for i, item in enumerate(data, 1)
        if (result := process_item(item, i, total)) is not None
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Done! Results saved to: {output_path}")

    return results
