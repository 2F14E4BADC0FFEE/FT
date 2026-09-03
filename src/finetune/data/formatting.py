def format_examples(dataset_entries, tokenizer):
    """Apply the model's chat template (with ChatML fallback)."""
    texts = []
    for entry in dataset_entries:
        try:
            text = tokenizer.apply_chat_template(
                entry["messages"], tokenize=False, add_generation_prompt=False
            )
        except Exception:
            text = "".join(
                f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
                for m in entry["messages"]
            )
        texts.append(text)
    return texts
