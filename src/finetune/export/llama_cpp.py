# finetune/export/llama_cpp.py
import subprocess
from pathlib import Path
from finetune.utils.settings import BASE_DIR

LLAMA_CPP_DIR = BASE_DIR / "llama.cpp"
BUILD_DIR = LLAMA_CPP_DIR / "build"


def assert_llamacpp_supports_gemma4(llama_dir=None):
    """
    Prüft, ob das convert_hf_to_gguf.py-Skript vorhanden ist.
    Optional kann ein abweichendes Verzeichnis übergeben werden.
    """
    if llama_dir is None:
        llama_dir = LLAMA_CPP_DIR
    converter = Path(llama_dir) / "convert_hf_to_gguf.py"
    if not converter.exists():
        raise FileNotFoundError(f"Konverter nicht gefunden: {converter}")
    print("✅ GGUF-Konverter gefunden.")


def build_llama_cpp():
    # (unverändert)
    if LLAMA_CPP_DIR.exists():
        print("llama.cpp bereits vorhanden – Build wird übersprungen.")
        return
    print("Klone llama.cpp …")
    subprocess.run(["git", "clone", "https://github.com/ggerganov/llama.cpp", str(LLAMA_CPP_DIR)], check=True)
    print("Installiere Build-Abhängigkeiten …")
    subprocess.run("apt-get update -qq && apt-get install -y -qq cmake build-essential", shell=True, check=True)
    BUILD_DIR.mkdir(exist_ok=True)
    subprocess.run(["cmake", "-B", str(BUILD_DIR), str(LLAMA_CPP_DIR)], check=True)
    subprocess.run(["cmake", "--build", str(BUILD_DIR), "-j4"], check=True)
    print("✅ llama.cpp built successfully")
    
    subprocess.run(["df", "-h", "/kaggle/working"])