_NON_LM_MARKERS = ("vision_tower", "audio_tower", "embed_vision", "embed_audio",
                   "multi_modal_projector", "vision_model", "audio_model")


def validate_lora_targets(model, target_modules):
    """Task 3: confirm every requested LoRA target exists in the model.

    Inspects model.named_modules() and checks each target appears as the
    suffix of at least one module (exactly how PEFT matches targets).
    Raises RuntimeError listing any missing targets; prints success if all
    are present. Training must not proceed if this raises.
    """
    present = {name.split(".")[-1] for name, _ in model.named_modules() if name}
    missing = [t for t in target_modules if t not in present]
    if missing:
        raise RuntimeError(
            "❌ LoRA target validation FAILED — missing in "
            f"{getattr(model.config, 'model_type', '?')}: {missing}\n"
            f"   Projection-like modules present: "
            f"{sorted(p for p in present if p.endswith('_proj'))}"
        )
    print(f"✅ LoRA targets present in "
          f"'{getattr(model.config, 'model_type', '?')}': {list(target_modules)}")
    return True

def resolve_lora_targets(model, target_modules):
    """Return concrete target_modules for LoraConfig.

    Multimodal model -> fully-qualified names INSIDE the language model only
    (towers excluded) so PEFT never touches Gemma4ClippableLinear. Otherwise
    -> the original suffix list, unchanged. Same 7 projections either way;
    LoRA r/alpha/dropout are untouched.
    """
    has_towers = any(any(mk in name for mk in _NON_LM_MARKERS)
                     for name, _ in model.named_modules())
    if not has_towers:
        return list(target_modules)
    wanted = set(target_modules)
    scoped = [name for name, _ in model.named_modules()
              if name.split(".")[-1] in wanted
              and not any(mk in name for mk in _NON_LM_MARKERS)]
    if not scoped:
        raise RuntimeError(
            "❌ Could not resolve any language-model LoRA targets matching "
            f"{sorted(wanted)} outside the multimodal towers."
        )
    print(f"ℹ️  Multimodal model detected → scoping LoRA to {len(scoped)} "
          f"language-model projections (towers excluded).")
    return scoped
