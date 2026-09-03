from pathlib import Path


def check_versions():
    """REQ 9: verify the transformers/peft stack actually supports Gemma 4."""
    from packaging.version import parse as _P
    import transformers, peft
    tv, pv = transformers.__version__, peft.__version__
    problems = []
    if _P(tv) < _P("5.5.0"):
        problems.append(f"transformers {tv} < 5.5.0 (model_type 'gemma4' unsupported)")
    if _P(pv) < _P("0.19.0"):
        problems.append(f"peft {pv} < 0.19.0 (Gemma 4 LoRA target handling missing)")
    try:
        from transformers import Gemma4Config  # noqa: F401  (Gemma 4 compatibility layer)
    except Exception:
        problems.append("transformers has no Gemma4Config (Gemma 4 compatibility layer absent)")
    if problems:
        raise RuntimeError("❌ Version/compatibility check FAILED:\n   - "
                           + "\n   - ".join(problems)
                           + "\n   Reinstall the pinned stack (Cell 1).")
    print(f"✅ Versions OK: transformers {tv}, peft {pv}; Gemma4 layer present.")
    return True

def has_valid_adapter(d):
    """REQ 5: True if OUTPUT_DIR holds a finished LoRA adapter."""
    d = Path(d)
    if not (d / "adapter_config.json").exists():
        return False
    return any((d / f).exists() for f in ("adapter_model.safetensors", "adapter_model.bin"))

def has_valid_merged(d):
    """REQ 5: True if MERGED_DIR holds a merged full model."""
    d = Path(d)
    if not (d / "config.json").exists():
        return False
    return bool(any(d.glob("*.safetensors")) or (d / "pytorch_model.bin").exists())
