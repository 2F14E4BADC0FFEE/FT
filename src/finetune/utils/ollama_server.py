"""Ollama daemon install/start/readiness. Extracted verbatim from notebook cell 29."""

import os, time, subprocess, requests
from pathlib import Path

ollama_proc = None


def install_ollama():
    # Persistente Modellablage  (F1: OLLAMA_MODELS set + dir created before daemon start)
    os.environ["OLLAMA_MODELS"] = "/kaggle/working/ollama_models"
    Path(os.environ["OLLAMA_MODELS"]).mkdir(parents=True, exist_ok=True)

    print("Installing Ollama...")

    # zstd wird vom Installer benötigt
    subprocess.run(
        """
        apt-get update -qq &&
        apt-get install -y -qq curl zstd &&
        curl -fsSL https://ollama.com/install.sh | sh
        """,
        shell=True,
        check=True
    )


def start_ollama():
    global ollama_proc
    print("Starting Ollama daemon...")

    # Hintergrundprozess starten
    ollama_proc = subprocess.Popen(
        ["ollama", "serve"],
        stdout=open("/tmp/ollama.log", "w"),
        stderr=subprocess.STDOUT,
    )
    return ollama_proc


def wait_for_ollama():
    # Warten bis API antwortet
    for i in range(60):
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=2)
            if r.ok:
                print("✅ Ollama is running")
                break
        except Exception:
            pass

        time.sleep(1)

    else:
        raise RuntimeError(
            "❌ Ollama failed to start. Check: /tmp/ollama.log"
        )

    print("Ollama ready.")
