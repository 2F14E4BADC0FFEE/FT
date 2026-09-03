from pathlib import Path
from finetune.utils.settings import BASE_DIR, OUTPUT_DIR, MERGED_DIR, DATA_SOURCE_DIR


def check_packages():
    pkgs = {}
    for name in ["torch", "transformers", "peft", "datasets", "trl", "accelerate"]:
        try:
            mod = __import__(name)
            pkgs[name] = getattr(mod, "__version__", "OK")
        except ImportError:
            pkgs[name] = "NOT INSTALLED"
    return pkgs


def report_gpu():
    import torch
    for k, v in check_packages().items():
        icon = "✅" if v != "NOT INSTALLED" else "❌"
        print(f"  {icon} {k}: {v}")

    print(f"\nGPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            mem  = torch.cuda.get_device_properties(i).total_memory / 1024**3
            cap  = torch.cuda.get_device_capability(i)
            print(f"  GPU {i}: {name} ({mem:.1f} GB, sm_{cap[0]}{cap[1]})")
    else:
        print("  ⚠️  No GPU detected. Set Kaggle Accelerator to 'GPU T4 x2' or 'GPU P100' under Settings → Accelerator.")


    # FIX #4: startup validation — print installed versions and confirm the
    # Gemma 4 layer is importable (validation only; no version/install changes).
    import transformers, peft
    print(f"transformers: {transformers.__version__}")
    print(f"peft:         {peft.__version__}")
    try:
        from transformers import Gemma4Config  # noqa: F401
        print("✅ Gemma4Config importable.")
    except Exception as e:
        raise RuntimeError(f"❌ Gemma4Config not importable: {e}")


def ensure_directories():
    # (a) create every writable output directory up front
    for d in [BASE_DIR, OUTPUT_DIR, MERGED_DIR, BASE_DIR / "logs"]:
        Path(d).mkdir(parents=True, exist_ok=True)


def assert_writable():
    # (b) verify write permissions on every writable directory
    for d in [BASE_DIR, OUTPUT_DIR, MERGED_DIR, BASE_DIR / "logs"]:
        probe = Path(d) / ".write_test"
        try:
            probe.write_text("ok"); probe.unlink()
        except Exception as e:
            raise RuntimeError(f"❌ Output dir not writable: {d} ({e})")
    print("✅ Output directories exist and are writable.")
    print(f"   READ-ONLY input : {DATA_SOURCE_DIR}  (exists={DATA_SOURCE_DIR.exists()})")
    print(f"   WRITABLE output : {BASE_DIR}")
