# preflight.py – an die neuen Splits angepasst
# ============================================================================
# Enthält:
#   run_preflight_checks()    – 6 Integritätschecks (4‑Bit QLoRA)
#   bootstrap_gate()          – Umgebungs- und Datei‑Prüfung (jetzt mit train/val/test)
# ============================================================================

import os
import gc
import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from finetune.utils.settings import (
    MODEL_NAME, TARGET_MODULES, SYSTEM_PROMPT, MAX_SEQ_LENGTH,
    LOCAL_MODEL_DIR, BASE_DIR, DATA_SOURCE_DIR, OUTPUT_DIR, MERGED_DIR,
)
from finetune.models.compat import validate_lora_targets
from finetune.models.download import _gemma_install_valid
from finetune.utils.checks import check_versions


def run_preflight_checks(model_name=MODEL_NAME, target_modules=TARGET_MODULES):
    """
    Führt 6 Integritätschecks am echten Modell durch (4‑Bit QLoRA).
    Gibt GPU‑Speicher danach frei.
    """
    print("=" * 60)
    print("PREFLIGHT CHECKS (4‑Bit QLoRA)")
    print("=" * 60)

    # ── Tokenizer ────────────────────────────────────────────
    try:
        tok = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=True,
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        print("✅ (2/6) tokenizer loaded")
    except Exception as e:
        raise RuntimeError(f"❌ (2/6) tokenizer failed: {e}")

    # ── Chat‑Template (muss system/user/assistant unterstützen) ─
    probe = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "What is variance?"},
        {"role": "assistant", "content": "Variance measures spread."},
    ]
    try:
        tok.apply_chat_template(probe, tokenize=False, add_generation_prompt=False)
        print("✅ (3/6) chat template ok (system/user/assistant)")
    except Exception as e:
        raise RuntimeError(
            f"❌ (3/6) chat template rejected system/user/assistant: {e}"
        )

    # ── 4‑Bit‑Modell laden ───────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        print("✅ (1/6) model loaded (4‑bit QLoRA)")
    except Exception as e:
        raise RuntimeError(f"❌ (1/6) model load failed: {e}")

    # ── LoRA‑Targets prüfen ──────────────────────────────────
    validate_lora_targets(model, target_modules)
    print("✅ (4/6) LoRA targets verified")

    # ── Tokenisierung ─────────────────────────────────────────
    try:
        enc = tok("Design For Six Sigma uses statistics.",
                  truncation=True, max_length=MAX_SEQ_LENGTH, return_tensors="pt")
        assert enc["input_ids"].shape[-1] > 0
        print(f"✅ (5/6) tokenization ok ({enc['input_ids'].shape[-1]} tokens)")
    except Exception as e:
        raise RuntimeError(f"❌ (5/6) tokenization failed: {e}")

    # ── Ein Inferenz‑Schritt ──────────────────────────────────
    try:
        model.eval()
        dev = next(model.parameters()).device
        with torch.no_grad():
            out = model.generate(**{k: v.to(dev) for k, v in enc.items()},
                                 max_new_tokens=8, pad_token_id=tok.pad_token_id)
        assert out.shape[-1] >= enc["input_ids"].shape[-1]
        print("✅ (6/6) inference pass ok")
    except Exception as e:
        raise RuntimeError(f"❌ (6/6) inference failed: {e}")

    # ── Speicher freigeben ────────────────────────────────────
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("=" * 60)
    print("✅ PREFLIGHT PASSED — training may proceed.")
    print("=" * 60)
    return True


def bootstrap_gate(dataset_entries=None):
    """
    Zentrale go/no‑go‑Prüfung VOR dem Training.
    Prüft Pakete, GPU, HF‑Token, lokales Modell, Datensplits und Schreibrechte.
    """
    problems = []

    # 1) Notwendige Pakete
    try:
        import transformers, peft, trl, datasets, accelerate  # noqa: F401
    except Exception as e:
        problems.append(f"package import failed: {e}")

    # 2) CUDA‑GPU
    if not torch.cuda.is_available():
        problems.append("no CUDA GPU – set Kaggle Accelerator to GPU T4 x2")

    # 3) Hugging‑Face‑Token (für gated Models wie Gemma)
    if not os.environ.get("HF_TOKEN"):
        problems.append("HF_TOKEN not set (run Cell 1b)")

    # 4) Lokale Gemma‑Installation gültig?
    if not _gemma_install_valid(LOCAL_MODEL_DIR):
        problems.append(f"local Gemma install missing/invalid at {LOCAL_MODEL_DIR}")

    # 5) Trainings‑ und Validierungs‑Splits vorhanden (Datei‑Check)
    #    dataset_entries kann als Fallback dienen, aber wir erzwingen die neuen Split‑Dateien.
    train_exists = any(
        (root / "train_qa.json").exists()
        for root in (BASE_DIR, DATA_SOURCE_DIR)
    )
    val_exists = any(
        (root / "validation_qa.json").exists()
        for root in (BASE_DIR, DATA_SOURCE_DIR)
    )

    if not (train_exists and val_exists):
        problems.append(
            "Training/Validation splits missing – "
            "train_qa.json and validation_qa.json not found in BASE_DIR or DATA_SOURCE_DIR"
        )
    # (Optional: zusätzlich prüfen, ob dataset_entries übergeben wurde – 
    #  aber das wäre nur relevant, wenn jemand die Daten im Speicher hält.)

    # 6) Ausgabeverzeichnisse schreibbar?
    for d in (OUTPUT_DIR, MERGED_DIR, BASE_DIR / "logs"):
        try:
            Path(d).mkdir(parents=True, exist_ok=True)
            (Path(d) / ".w").write_text("ok")
            (Path(d) / ".w").unlink()
        except Exception as e:
            problems.append(f"output dir not writable: {d} ({e})")

    if problems:
        raise RuntimeError(
            "❌ BOOTSTRAP FAILED — training aborted:\n   - " +
            "\n   - ".join(problems)
        )

    # 7) Versionskompatibilität
    check_versions()
    print("=" * 60)
    print("✅ BOOTSTRAP OK — all stages passed; training may start.")
    print("=" * 60)
    return True