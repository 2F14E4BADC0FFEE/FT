"""
Driver — thin orchestration that replaces the notebook's "Run All" behavior.

Environment setup (run BEFORE this script; from notebook cells 4 and 39):
    pip install -q "transformers==5.5.4" "peft==0.19.0" "trl==1.1.0" datasets accelerate sentencepiece protobuf huggingface-hub requests
    pip install -q evaluate bert-score rouge-score sacrebleu

Each call below corresponds to one original notebook code cell, in the same order.
"""

# ── A2: CUDA_VISIBLE_DEVICES must be set BEFORE importing torch / any torch-pulling module ──
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ── Import-order rule: finetune.utils.gpu before any module that pulls in torch ──
import finetune.utils.gpu  # noqa: F401

# ── Public API ──
from finetune.utils.environment import report_gpu, ensure_directories, assert_writable
from finetune.auth.huggingface import authenticate
from finetune.models.download import model_available, ensure_base_model
from finetune.data.dataset_loader import load_qa_dataset
from finetune.preflight import run_preflight_checks, bootstrap_gate
from finetune.training.trainer import train_model
from finetune.utils.ollama_server import install_ollama, start_ollama, wait_for_ollama
from finetune.models.inference import chat
from finetune.evaluation.quick_eval import run_quick_eval
from finetune.export.llama_cpp import build_llama_cpp, LLAMA_CPP_DIR
from finetune.export.gguf import assert_llamacpp_supports_gemma4, export_gguf
from finetune.export.merge import merge_adapter
from finetune.export.ollama_import import import_ollama
from finetune.evaluation.batch_eval import process_qa_file
from finetune.evaluation.scientific import run_evaluation
from finetune.utils.reproducibility import write_requirements
from finetune.utils.settings import BASE_DIR, DATA_SOURCE_DIR


def main():
    # Cell 4 — package/GPU report
    report_gpu()

    # Cell 1b — Hugging Face authentication
    authenticate()

    # Cell 2 — writable directories, then gated-model reachability
    ensure_directories()
    assert_writable()
    model_available()

    # Cell 3 — base model (offline reuse or one-time download)
    ensure_base_model()

    # Cell 4 — dataset load
    dataset_entries = load_qa_dataset()

    # Cell — preflight checks on the actual target model
    run_preflight_checks()

    # Cell — bootstrap gate (A5: dataset_entries passed explicitly)
    bootstrap_gate(dataset_entries=dataset_entries)

    # Cell 5 — fine-tuning (run-control: skip / resume / smoke / full)
    train_model()

    # Cell — Ollama daemon (F1: OLLAMA_MODELS set in install_ollama, before the daemon)
    install_ollama()
    start_ollama()
    wait_for_ollama()

    # Cell tail — smoke query (F2: RNG-load-bearing; exact position preserved)
    q = "What is the formula for sample variance?"
    print("Q:", q)
    print("\nA:", chat(q))

    # Cell — quick eval (A6: before the Ollama model import)
    run_quick_eval()

    # Cell 8 — build llama.cpp
    build_llama_cpp()

    # A4: verify this llama.cpp build supports Gemma 4 BEFORE merging
    assert_llamacpp_supports_gemma4(LLAMA_CPP_DIR)

    # Cell 8 — merge / convert / import (exact cell-35 run-block order + separators)
    merge_adapter()
    print()
    export_gguf()
    print()
    import_ollama()

    # Cell — batch QA over the dataset (A6: after the Ollama model import)
    QA_FILENAME = "/kaggle/input/datasets/feyssal/finetuning/FineTuning/data_fixtures/QA.json"

    INPUT_FILE = QA_FILENAME

    # Search in writable + readonly dirs
    for root in [BASE_DIR, DATA_SOURCE_DIR]:
        candidate = root / QA_FILENAME

        if candidate.exists():
            INPUT_FILE = str(candidate)
            break

    OUTPUT_FILE = str(BASE_DIR / "finetuned_answers.json")

    results = process_qa_file(
        INPUT_FILE,
        OUTPUT_FILE,
    )

    # Cell 9 — scientific evaluation
    run_evaluation()

    # Cell — requirements.txt / requirements.freeze.txt
    write_requirements()


if __name__ == "__main__":
    main()
