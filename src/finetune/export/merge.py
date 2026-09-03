# finetune/export/merge.py
import gc
import shutil
import torch
import subprocess   # <-- wurde in der vorherigen Version vergessen
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from finetune.utils.settings import MODEL_NAME, OUTPUT_DIR, MERGED_DIR, FORCE_REMERGE


def merge_adapter():
    """
    CPU-Merge genau wie in der manuellen Erfolgszelle:
    - Existierenden Merge wiederverwenden, wenn nicht FORCE_REMERGE
    - Basismodell-Ordner nach dem Laden löschen
    - Speichern in 1 GB-Shards
    """
    if MERGED_DIR.exists() and (MERGED_DIR / "config.json").exists() and not FORCE_REMERGE:
        print("✅ Bereits gemergtes Modell gefunden – Merge wird übersprungen.")
        return

    # Alte Reste löschen
    for p in [Path("/kaggle/working/src"), MERGED_DIR]:
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)

    adapter_dir = Path(OUTPUT_DIR)
    if adapter_dir.exists():
        for ckpt in adapter_dir.glob("checkpoint-*"):
            shutil.rmtree(ckpt)

    # Base-Modell sicherstellen (download nur wenn nötig)
    if not Path(MODEL_NAME).exists():
        from finetune.models.download import ensure_base_model
        ensure_base_model()
    else:
        print("✅ Basismodell gefunden.")

    # Base auf CPU laden (wie in Zelle)
    print("Lade Basismodell auf CPU (FP16) …")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        trust_remote_code=True,
        device_map="cpu",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )

    # Basismodell-Ordner SOFORT löschen (Platz freigeben)
    print("Lösche Basismodell-Ordner (Platz für Merge) …")
    shutil.rmtree(MODEL_NAME, ignore_errors=True)

    # Adapter fusionieren
    print("Füge LoRA-Adapter hinzu und fusioniere …")
    peft_model = PeftModel.from_pretrained(base, str(OUTPUT_DIR))
    merged = peft_model.merge_and_unload()
    del base, peft_model
    gc.collect()

    # Merged speichern (1 GB-Shards)
    print("Speichere gemergtes Modell (max. 1 GB pro Shard) …")
    merged.save_pretrained(MERGED_DIR, max_shard_size="1GB", safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(MERGED_DIR))
    del merged
    gc.collect()
    print(f"✅ Merge abgeschlossen: {MERGED_DIR}")
    
    subprocess.run(["df", "-h", "/kaggle/working"])