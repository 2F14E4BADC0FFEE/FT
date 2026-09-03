# finetune/export/ollama_import.py

import os
import sys
import time
import subprocess

from finetune.utils.settings import (
    GGUF_OUTPUT,
    MODELFILE,
    OLLAMA_MODEL_NAME,
)


def import_ollama():
    """
    Ollama-Import:
    - Ollama installieren (falls nötig)
    - Server starten (pkill + neu)
    - Modell nur importieren, wenn nicht vorhanden
    """

    # Ollama installieren
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "ollama"],
        check=True,
    )

    # Vorherigen Server beenden
    subprocess.run(
        ["pkill", "-f", "ollama serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(1)

    # Ollama-Modellverzeichnis
    os.environ["OLLAMA_MODELS"] = "/tmp/ollama_models"
    os.makedirs("/tmp/ollama_models", exist_ok=True)

    # Ollama-Server starten
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(5)

    # Bereits importierte Modelle prüfen
    existing = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True,
    )

    if OLLAMA_MODEL_NAME not in existing.stdout:
        MODELFILE.write_text(f"FROM {GGUF_OUTPUT}\n")

        imp = subprocess.run(
            ["ollama", "create", OLLAMA_MODEL_NAME, "-f", str(MODELFILE)],
            capture_output=True,
            text=True,
        )

        if imp.returncode == 0:
            print("✅ Ollama-Modell importiert.")
        else:
            print("❌ Fehler beim Import:")
            print(imp.stderr)
    else:
        print("✅ Ollama-Modell bereits importiert.")


# Optional: Speicherplatz anzeigen
subprocess.run(["df", "-h", "/kaggle/working"])