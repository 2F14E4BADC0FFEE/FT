import re, subprocess
from pathlib import Path
from finetune.utils.settings import BASE_DIR


def write_requirements():
    REQ_DIR = BASE_DIR
    REQ_DIR.mkdir(parents=True, exist_ok=True)

    USED = ["transformers", "peft", "trl", "datasets", "accelerate", "sentencepiece",
            "protobuf", "huggingface-hub", "requests", "evaluate", "bert-score",
            "rouge-score", "sacrebleu", "numpy", "torch"]

    freeze = subprocess.run(["pip", "freeze"], capture_output=True, text=True).stdout
    (REQ_DIR / "requirements.freeze.txt").write_text(freeze, encoding="utf-8")

    ver = {}
    for line in freeze.splitlines():
        m = re.match(r"^([A-Za-z0-9_.\-]+)==(.+)$", line.strip())
        if m:
            ver[m.group(1).lower()] = m.group(2)

    lines = [
        "# Auto-generated from the live environment (pip freeze).",
        "# transformers>=5.5.0 is the one mandatory upgrade: Gemma 4 (model_type",
        "# 'gemma4') is unknown to transformers 4.x and will not load.",
        "# torch: use the Kaggle-preinstalled CUDA build (do NOT reinstall).",
    ]
    for pkg in USED:
        v = ver.get(pkg.lower())
        lines.append(f"{pkg}=={v}" if v else f"# {pkg}: not detected via pip freeze")
    (REQ_DIR / "requirements.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("✅ wrote", REQ_DIR / "requirements.txt")
    print("✅ wrote", REQ_DIR / "requirements.freeze.txt", f"({len(freeze.splitlines())} pkgs)")
    print("\n".join(lines))
