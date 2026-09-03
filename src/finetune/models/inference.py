import json
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from finetune.utils.settings import MODEL_NAME, OUTPUT_DIR, SYSTEM_PROMPT, OLLAMA_URL, OLLAMA_MODEL_NAME
from finetune.training.state import train_state


def _ensure_local_model():
    """Load the fine-tuned model into train_state if not already there."""
    if train_state.get("model") is not None:
        return train_state["model"], train_state["tokenizer"]

    if not OUTPUT_DIR.exists():
        raise RuntimeError(
            f"No adapter at {OUTPUT_DIR}. Run Cell 5 first or point OUTPUT_DIR at an existing adapter."
        )

    print(f"Loading base + adapter from disk ({OUTPUT_DIR})…")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16,
        device_map=None, trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, str(OUTPUT_DIR))
    tokenizer = AutoTokenizer.from_pretrained(str(OUTPUT_DIR))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_state["model"], train_state["tokenizer"] = model, tokenizer
    return model, tokenizer


def chat_local(question, max_new_tokens=256, temperature=0.2):
    """Real model inference on GPU — typically 5-30s per answer for 1.5B on T4."""
    model, tokenizer = _ensure_local_model()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        prompt = "".join(
            f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
            for m in messages
        ) + "<|im_start|>assistant\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    model.eval()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
        )
    answer = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return answer.strip()


def chat_ollama(question, model=OLLAMA_MODEL_NAME, temperature=0.2, num_predict=1024):
    """Use Ollama (after Cell 8 export). Fast."""
    if not question.strip():
        return "Please enter a question."
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": model, "prompt": question,
                  "options": {"temperature": temperature, "num_predict": num_predict}},
            stream=True, timeout=300,
        )
        out = ""
        for line in r.iter_lines():
            if line:
                data = json.loads(line.decode("utf-8"))
                out += data.get("response", "")
                if data.get("done"):
                    break
        return out.strip() or "[Empty response]"
    except requests.exceptions.ConnectionError:
        return "❌ Ollama not running. Start with: ollama serve"
    except Exception as e:
        return f"❌ Error: {e}"


def chat(question, **kwargs):
    """Auto: prefer Ollama if up, else local model."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        names = [m["name"] for m in r.json().get("models", [])]
        if any(OLLAMA_MODEL_NAME in n for n in names):
            return chat_ollama(question, **kwargs)
    except Exception:
        pass
    return chat_local(question, **kwargs)
