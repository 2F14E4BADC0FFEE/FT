# finetune/export/gguf.py
import os, shutil, subprocess, sys
from pathlib import Path
from finetune.utils.settings import BASE_DIR, MERGED_DIR, GGUF_OUTPUT, LLAMA_CPP_DIR


def export_gguf():
    """
    Konvertierung + Quantisierung exakt wie in der Zelle:
    - Ausgabe nach /tmp
    - Erst danach Merge-Ordner löschen und GGUF verschieben
    """
    if GGUF_OUTPUT.exists():
        print("✅ GGUF bereits vorhanden – Export wird übersprungen.")
        return

    # Pfade wie in der Zelle
    LLAMA_DIR = LLAMA_CPP_DIR
    TMP_FP16_GGUF = Path("/tmp/gemma-dfss-f16.gguf")
    TMP_Q4_GGUF   = Path("/tmp/gemma-dfss-q4_k_m.gguf")

    converter = LLAMA_DIR / "convert_hf_to_gguf.py"
    if not converter.exists():
        raise FileNotFoundError(f"Konverter nicht gefunden: {converter}")

    # llama-quantize bauen, falls nötig
    quantize_bin = LLAMA_DIR / "build" / "bin" / "llama-quantize"
    if not quantize_bin.exists():
        print("Baue llama-quantize …")
        build_dir = LLAMA_DIR / "build"
        build_dir.mkdir(exist_ok=True)
        subprocess.run(["cmake", "..", "-DLLAMA_CURL=OFF"], cwd=str(build_dir),
                       capture_output=True, text=True, check=True)
        subprocess.run(["cmake", "--build", ".", "--target", "llama-quantize", "-j2"],
                       cwd=str(build_dir), capture_output=True, text=True, check=True)
        if not quantize_bin.exists():
            raise RuntimeError("Build von llama-quantize fehlgeschlagen.")
        print("✅ llama-quantize gebaut.")

    # FP16 konvertieren
    print("Konvertiere nach FP16 (Ausgabe in /tmp) …")
    res = subprocess.run(
        [sys.executable, str(converter), str(MERGED_DIR), "--outtype", "f16",
         "--outfile", str(TMP_FP16_GGUF)],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(LLAMA_DIR)}
    )
    if res.returncode != 0:
        raise RuntimeError(f"FP16-Konvertierung fehlgeschlagen:\n{res.stderr}")
    print(f"✅ FP16 GGUF erzeugt: {TMP_FP16_GGUF}")

    # Quantisieren
    print("Quantisiere nach Q4_K_M …")
    res = subprocess.run(
        [str(quantize_bin), str(TMP_FP16_GGUF), str(TMP_Q4_GGUF), "q4_k_m"],
        capture_output=True, text=True
    )
    if res.returncode != 0:
        raise RuntimeError(f"Quantisierung fehlgeschlagen:\n{res.stderr}")
    print(f"✅ Q4_K_M GGUF erzeugt: {TMP_Q4_GGUF}")

    # Platz schaffen: Merge-Ordner löschen
    if MERGED_DIR.exists():
        print("Lösche Merge-Ordner (wird nicht mehr benötigt) …")
        shutil.rmtree(MERGED_DIR, ignore_errors=True)

    # Verschiebe finales GGUF
    print("Verschiebe GGUF nach /kaggle/working …")
    shutil.move(str(TMP_Q4_GGUF), str(GGUF_OUTPUT))
    print(f"✅ GGUF gespeichert unter: {GGUF_OUTPUT}")

    # Aufräumen
    TMP_FP16_GGUF.unlink(missing_ok=True)
    
    subprocess.run(["df", "-h", "/kaggle/working"])