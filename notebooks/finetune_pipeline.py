# %% [markdown]
# # 🚀 Fine-Tuning, Export, and Scientific Evaluation Pipeline
#
# **Gemma 4 · LoRA/PEFT · GGUF + Ollama · rigorous, reproducible evaluation**
#
# This notebook is the *orchestrator* for a fully modular pipeline. Every piece of real logic
# lives in the installable package **`src/finetune/`** — a set of small, **definition-only**
# modules with zero import-time side effects (no file writes, no network calls, no model loading,
# and no CUDA initialization happen merely by importing them). The notebook's only job is to call
# that public API **in the correct order** and to narrate what is happening at each step.
#
# ### Why this shape?
# The original project was a single monolithic notebook. Lifting the heavy work into a package
# buys us testability, reuse, and — crucially — *deterministic execution order*. A handful of
# ordering constraints are genuinely load-bearing and are called out inline as we reach them
# (for example, setting `CUDA_VISIBLE_DEVICES` **before** `torch` is ever imported).
#
# ### End-to-end flow
# 1. **Environment & auth** — verify the GPU, authenticate to the Hugging Face Hub, prepare directories.
# 2. **Model & data** — fetch the gated Gemma 4 weights (offline-first) and load the QA dataset.
# 3. **Preflight & gate** — prove the model, tokenizer, chat template, and LoRA targets all work *before* training.
# 4. **Fine-tuning** — memory-efficient LoRA on a single GPU.
# 5. **Ollama service** — stand up a local inference daemon and run a smoke query.
# 6. **Quick eval & build** — a 5-question sanity check, then compile `llama.cpp`.
# 7. **Merge & export** — fuse the adapter, convert to GGUF, quantize to Q4_K_M, import into Ollama.
# 8. **Batch eval & science** — answer the full QA set and score it with BERTScore / ROUGE-L / BLEU / chrF, bootstrap CIs, and a contamination check.
# 9. **Reproducibility** — pin the exact environment to `requirements.txt`.
#
# > Each code cell maps one-to-one to a stage of the original pipeline and contains **no local
# > logic** beyond thin glue — it imports and calls the package.

# %%
# ── Cell 1: Package installations & environment configuration ──
#
# Pinned dependencies. transformers >= 5.5 is the one mandatory upgrade: Gemma 4
# (model_type "gemma4") is unknown to transformers 4.x and will not load. Torch is the
# Kaggle-preinstalled CUDA build and is intentionally NOT reinstalled here.
!pip install -q "transformers==5.5.4" "peft==0.19.0" "trl==1.1.0" datasets accelerate sentencepiece protobuf huggingface-hub requests
!pip install -q evaluate bert-score rouge-score sacrebleu

# ── Rule A2: CUDA_VISIBLE_DEVICES MUST be set BEFORE torch (or any module that pulls in
#    torch) is imported. Setting it after the first `import torch` has no effect — CUDA
#    initializes once, against whatever devices were visible at that moment. ──
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ── Make the local package importable when running from the notebooks/ folder ──
import sys
import pathlib
_SRC = str(pathlib.Path.cwd().parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# ── Import-order rule: finetune.utils.gpu must be imported BEFORE any other module that
#    pulls in torch. (Importing it has no side effects; it just makes the single-GPU view
#    available to the trainer.) ──
import finetune.utils.gpu  # noqa: F401

# ── Public API — every callable below lives in the definition-only src/finetune package ──
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

# %% [markdown]
# ## 🛠️ Phase 1: Environment Diagnostics & Authentication
#
# Before committing GPU hours we confirm the basics. **`report_gpu()`** prints the installed
# library versions and the visible CUDA device(s), and validates that the Gemma 4 model class is
# importable from `transformers` — a fast fail if the environment is on a pre-Gemma-4 stack.
#
# **`authenticate()`** logs in to the Hugging Face Hub (reading `HF_TOKEN` from Kaggle secrets or
# the environment) and exports both `HF_TOKEN` and `HUGGING_FACE_HUB_TOKEN`, so every downstream
# loader sees the credential. Gemma is a **gated** model, so this step is required to obtain the
# weights at all.
#
# **`ensure_directories()`** creates the pipeline's output tree and **`assert_writable()`** proves
# each directory is writable with a tiny write-then-delete probe. Finally, **`model_available()`**
# performs a cheap Hub reachability check confirming the gated repository resolves with our
# token — surfacing any licensing or permission problem *now* rather than mid-download.

# %%
# ── Cell 2: Diagnostics & authentication ──
report_gpu()          # versions + visible GPU(s) + Gemma 4 import validation
authenticate()        # HF Hub login; exports HF_TOKEN + HUGGING_FACE_HUB_TOKEN
ensure_directories()  # create the writable output tree
assert_writable()     # prove every output directory is writable
model_available()     # cheap gated-Hub reachability check with our token

# %% [markdown]
# ## 📦 Phase 2: Base Model & Dataset Preparation
#
# **`ensure_base_model()`** is *offline-first*: if a complete local copy of the model already
# exists (config + tokenizer + weights) it is reused as-is and **no download happens** — this is
# exactly what lets the pipeline survive kernel restarts without re-pulling several gigabytes.
# Otherwise it re-verifies Hub access, performs a one-time `snapshot_download`, validates the
# downloaded files, and asserts the canonical invariant `MODEL_NAME == LOCAL_MODEL_DIR` so every
# later stage loads from the same local path.
#
# **`load_qa_dataset()`** loads the question/answer corpus into the package's shared `ALL_QA` list
# and returns it as `dataset_entries`. We capture that handle explicitly because the bootstrap
# gate in the next phase consumes it directly — there are no hidden globals in play.

# %%
# ── Cell 3: Load model & data ──
ensure_base_model()                   # reuse local copy, else one-time download + validation
dataset_entries = load_qa_dataset()   # populates the shared ALL_QA and returns the records

# %% [markdown]
# ## 🛡️ Phase 3: Preflight Integrity Checks & Bootstrap Gate
#
# Training is the expensive part, so we refuse to start it until everything it depends on is
# proven. **`run_preflight_checks()`** exercises the *actual* target model end-to-end on a tiny
# probe: it loads the tokenizer, confirms the chat template accepts the system/user/assistant
# format, loads the model with the same memory-efficient settings the trainer uses, verifies the
# LoRA target modules exist, tokenizes a sample, and runs a single generation pass — then frees
# the probe so training starts from a clean GPU. Any failure raises and halts the run.
#
# **`bootstrap_gate(dataset_entries=dataset_entries)`** is the centralized go/no-go: it checks
# packages, GPU availability, HF auth, a valid local model install, dataset presence, writable
# output directories, and version/compatibility — aborting with a single consolidated error if
# anything is missing. We pass `dataset_entries` **explicitly** (Rule A5) so the gate evaluates
# the dataset we just loaded rather than reaching into global state.

# %%
# ── Cell 4: Run preflight & gate ──
run_preflight_checks()                            # 6-point check on the real model; frees the probe
bootstrap_gate(dataset_entries=dataset_entries)   # Rule A5: dataset_entries passed explicitly

# %% [markdown]
# ## 🏋️‍♂️ Phase 4: Quantization-Aware LoRA Fine-Tuning
#
# **`train_model()`** runs the full fine-tune through the Hugging Face `Trainer`. A few design
# points worth understanding:
#
# - **LoRA / PEFT** trains only small low-rank adapters on the attention/MLP projections while the
#   base weights stay frozen — this is what makes single-GPU fine-tuning of a multi-billion-
#   parameter model feasible.
# - **Memory-efficient loading** combines `dtype=torch.float16`, `low_cpu_mem_usage=True`, and
#   SDPA attention with **gradient checkpointing** (trading ~20% compute for a large
#   activation-memory saving) to fit on a single T4-class GPU.
# - **Reproducibility** is anchored by `seed=42` in `TrainingArguments`; checkpoints are written
#   per epoch (`save_strategy="epoch"`, `save_total_limit=2`).
# - **Run-control branches** make the cell safe to re-run: a finished adapter is *reused* (skip)
#   unless you opt into **resuming** from the latest checkpoint, request a 20-step **smoke** run,
#   or force a clean **full** retrain.
#
# > Note: the weights here are trained in **fp16** — the aggressive **4-bit (Q4_K_M) quantization
# > happens later, at the GGUF export stage (Phase 7)**, not during training.

# %%
# ── Cell 5: Fine-tuning core (skip / resume / smoke / full is decided inside train_model) ──
train_model()

# %% [markdown]
# ## 🦙 Phase 5: Ollama Daemon & Local Inference Service
#
# To serve the model locally we install and launch **Ollama** as a background daemon.
# **`install_ollama()`** sets `OLLAMA_MODELS` to a persistent directory **before** the server is
# started (Rule F1), so models land on writable storage, then installs the binary.
# **`start_ollama()`** launches `ollama serve` as a background process (logging to
# `/tmp/ollama.log`) and retains the process handle, and **`wait_for_ollama()`** polls the local
# API until it responds (or times out with a clear, actionable error).
#
# We then run a **smoke query**. This call is intentionally placed *here, in this exact position*
# (Rule F2) — it touches the RNG and must not be moved relative to the surrounding steps. At this
# point the fine-tuned model has **not** yet been imported into Ollama (that happens in Phase 7),
# so `chat()` automatically falls back to the **local PEFT model**, giving us an immediate, honest
# read on the freshly trained adapter.

# %%
# ── Cell 6: Ollama server & local chat smoke test ──
install_ollama()    # Rule F1: sets OLLAMA_MODELS before the daemon is started
start_ollama()      # background `ollama serve` (logs to /tmp/ollama.log; handle retained)
wait_for_ollama()   # poll the local API until ready

# Rule F2: smoke query — keep this text and position exactly (RNG-load-bearing).
q = "What is the formula for sample variance?"
print("Q:", q)
print("\nA:", chat(q))

# %% [markdown]
# ## 🧪 Phase 6: Quick Evaluation & llama.cpp Compilation
#
# **`run_quick_eval()`** asks the model five canonical statistics questions and logs the answers
# (with latencies) to `eval_results.json` — a fast qualitative gut-check before the full
# scientific evaluation. It runs **before** the Ollama import, so it too reflects the local model.
#
# **`build_llama_cpp()`** clones and compiles **llama.cpp** with CMake. We need its
# `convert_hf_to_gguf.py` converter and the `llama-quantize` binary to turn the merged Hugging
# Face model into a quantized GGUF in the next phase.

# %%
# ── Cell 7: Quick eval & build ──
run_quick_eval()    # 5-question baseline → eval_results.json
build_llama_cpp()   # clone + CMake build of llama.cpp (converter + quantizer)

# %% [markdown]
# ## 💾 Phase 7: LoRA Merging & GGUF Quantization Export
#
# First, a safety gate: **`assert_llamacpp_supports_gemma4(LLAMA_CPP_DIR)`** confirms — *before* we
# spend time merging — that this particular llama.cpp build can actually convert a Gemma 4
# architecture (it queries the converter's own model registry, with a source-scan fallback) and
# raises early if not. This must run **before the merge** (Rule A4).
#
# Then the export chain:
# - **`merge_adapter()`** fuses the LoRA adapter into the base weights and writes a standalone
#   merged model (reusing an existing merge unless `FORCE_REMERGE` is set).
# - **`export_gguf()`** converts the merged model to an FP16 GGUF and then quantizes it to
#   **Q4_K_M**, cleaning up the intermediate file.
# - **`import_ollama()`** writes a `Modelfile` (carrying the system prompt and sampling
#   parameters) and registers the GGUF with Ollama via `ollama create`.
#
# The blank `print()` calls between stages reproduce the original run-block's spacing for readable
# logs.

# %%
# ── Cell 8: Merge, quantize & Ollama import ──
assert_llamacpp_supports_gemma4(LLAMA_CPP_DIR)   # Rule A4: must run BEFORE the merge
merge_adapter()                                  # fuse LoRA into the base; write merged model
print()
export_gguf()                                    # merged HF model → FP16 GGUF → Q4_K_M
print()
import_ollama()                                  # write Modelfile + `ollama create`

# %% [markdown]
# ## 📊 Phase 8: Automated Batch Inference & Scientific Metrics
#
# Now we evaluate at scale. We resolve the evaluation QA file (searching the writable and
# read-only data roots), then **`process_qa_file(...)`** answers every question — by now the
# fine-tuned model is registered in Ollama, so `chat()` routes through the fast Ollama path — and
# writes `finetuned_answers.json`.
#
# **`run_evaluation()`** then produces a rigorous, reproducible report:
# - **BERTScore F1** (primary) on a multilingual **XLM-RoBERTa-large** backbone (the language
#   defaults to German, `"de"`, and is configurable), plus **ROUGE-L**, **BLEU**, and secondary **chrF**.
# - **Bootstrap confidence intervals** (5,000 resamples, 95%, fixed `seed=42`) around each metric,
#   together with full distribution statistics and a composite ranking.
# - An **exact-overlap contamination check** between the training and evaluation sets.
# - Best/worst-25 example dumps and a sampled **human-review** set for qualitative audit.
#
# All artifacts are written as JSON inside the pipeline's base directory.

# %%
# ── Cell 9: Batch processing & scientific evaluation ──
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

run_evaluation()   # BERTScore/ROUGE-L/BLEU/chrF + bootstrap CIs + contamination + best/worst + human-review

# %% [markdown]
# ## 📜 Phase 9: Reproducibility & Requirements Export
#
# Finally, **`write_requirements()`** snapshots the live environment: it writes a complete
# `requirements.freeze.txt` (everything `pip freeze` reports) plus a curated `requirements.txt`
# pinning the libraries this pipeline actually uses — so the exact run can be reconstructed later.
#
# 🎉 **That's the whole pipeline:** from a gated base model and a QA corpus to a fine-tuned,
# quantized, Ollama-served model with a defensible, reproducible evaluation. Re-running is safe —
# the offline-first model fetch, the adapter/merge reuse, and the run-control branches all avoid
# redoing finished work.

# %%
# ── Cell 10: Export state ──
write_requirements()   # writes requirements.txt + requirements.freeze.txt
