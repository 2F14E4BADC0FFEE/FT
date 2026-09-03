import os
from pathlib import Path
from huggingface_hub import snapshot_download, model_info
from transformers import AutoTokenizer
from finetune.utils.settings import LOCAL_MODEL_DIR, HF_REPO_ID, MODEL_NAME


_REQUIRED_FILES = ["config.json", "tokenizer_config.json"]

def _gemma_install_valid(d):
    """True only if LOCAL_MODEL_DIR holds the CORRECT Gemma 4 model."""
    d = Path(d)
    if not d.exists():
        return False
    # Basischecks: config, tokenizer, Gewichte vorhanden?
    if not all((d / f).exists() for f in _REQUIRED_FILES):
        return False
    has_tok = (d / "tokenizer.json").exists() or (d / "tokenizer.model").exists()
    has_weights = any(d.glob("*.safetensors"))
    if not (has_tok and has_weights):
        return False

    # NEU: Identitätscheck – ist es wirklich Gemma 4?
    try:
        import json
        with open(d / "config.json", "r") as f:
            cfg = json.load(f)
        # Methode 1: _name_or_path enthält die Repo-ID
        if HF_REPO_ID in cfg.get("_name_or_path", ""):
            return True
        # Methode 2: Architektur-Liste enthält "Gemma4"
        if any("Gemma4" in arch for arch in cfg.get("architectures", [])):
            return True
        return False
    except Exception:
        return False


def model_available():
    """Cheap Hub reachability check — confirms gated model is accessible."""
    try:
        _info = model_info(HF_REPO_ID, token=os.environ.get("HF_TOKEN"))
        print(f"✅ Model available on the Hub: {HF_REPO_ID} (sha {_info.sha[:8]})")
    except Exception as e:
        raise RuntimeError(
            f"❌ Cannot access {HF_REPO_ID} on the Hugging Face Hub: {e}\n"
            "   → Accept the Gemma license on the model page and ensure HF_TOKEN is attached."
        )


def ensure_base_model():
    """
    Ensures the base model is available locally. Downloads if missing,
    validates integrity, and sets up the chat template.
    """
    # ── Step 1: Check if valid local copy already exists ──
    if _gemma_install_valid(LOCAL_MODEL_DIR):
        print("ℹ️  Local Gemma installation detected.")
        print(f"   → {LOCAL_MODEL_DIR}  (skipping download)")
    else:
        # ── Step 2: Verify Hub access BEFORE the large download ──
        try:
            model_info(HF_REPO_ID, token=os.environ.get("HF_TOKEN"))
        except Exception as e:
            raise RuntimeError(
                f"❌ Cannot access {HF_REPO_ID}: {e}\n"
                "   → Accept the Gemma license and attach HF_TOKEN."
            )

        # ── Step 3: Download the complete model ──
        print(f"📥 Downloading {HF_REPO_ID} → {LOCAL_MODEL_DIR} (one-time)…")
        LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=HF_REPO_ID,
            local_dir=str(LOCAL_MODEL_DIR),
            token=os.environ.get("HF_TOKEN"),
        )

        # ── Step 4: Verify all required files exist ──
        if not _gemma_install_valid(LOCAL_MODEL_DIR):
            missing = [f for f in _REQUIRED_FILES if not (LOCAL_MODEL_DIR / f).exists()]
            raise RuntimeError(
                "❌ Gemma download incomplete — missing/invalid files in "
                f"{LOCAL_MODEL_DIR} (e.g. {missing or 'tokenizer/weights'})."
            )
        print(f"✅ Gemma installed locally: {LOCAL_MODEL_DIR}")

    # ── Step 5: Verify MODEL_NAME points to local copy ──
    assert str(LOCAL_MODEL_DIR) == str(MODEL_NAME), (
        f"MODEL_NAME ({MODEL_NAME}) must equal LOCAL_MODEL_DIR ({LOCAL_MODEL_DIR})."
    )
    print(f"   MODEL_NAME → {MODEL_NAME}")

    # ── Step 6: Ensure chat template is set (Gemma compatibility) ──
    tok = AutoTokenizer.from_pretrained(str(LOCAL_MODEL_DIR), trust_remote_code=True)
    
    if not tok.chat_template:
        # Set the Gemma chat template (system + user/assistant turns)
        tok.chat_template = """\
{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content'] %}
    {%- set messages = messages[1:] %}
{%- else %}
    {%- set system_message = '' %}
{%- endif %}
{%- for message in messages %}
    {%- if message['role'] == 'user' %}
        {%- if loop.index0 == 0 and system_message != '' %}
            {{- '<start_of_turn>user\\n' + system_message + '\\n' + message['content'] + '<end_of_turn>\\n' }}
        {%- else %}
            {{- '<start_of_turn>user\\n' + message['content'] + '<end_of_turn>\\n' }}
        {%- endif %}
    {%- elif message['role'] == 'assistant' %}
        {{- '<start_of_turn>model\\n' + message['content'] + '<end_of_turn>\\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<start_of_turn>model\\n' }}
{%- endif %}
"""
        tok.save_pretrained(str(LOCAL_MODEL_DIR))
        print("✅ Chat template set and saved.")
    else:
        print("ℹ️  Chat template already configured.")

    print("✅ Base model ready for training.")