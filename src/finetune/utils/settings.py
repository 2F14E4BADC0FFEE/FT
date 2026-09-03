from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────
# Kaggle:  Path("/kaggle/working/finetune_pipeline_v2")
# Local:   Path("./finetune_pipeline_v2")

# WRITABLE: hier landen Adapter, Checkpoints, Merged Model, GGUF
BASE_DIR = Path("/kaggle/working/finetune_pipeline_v2")

# READ-ONLY: hier liegt dein Dataset (passe an deinen tatsächlichen Pfad an)
DATA_SOURCE_DIR = Path("/kaggle/input/datasets/feyssal/finetuning/FineTuning")

OUTPUT_DIR  = BASE_DIR / "gemma-dfss-peft"      # LoRA adapter
MERGED_DIR  = BASE_DIR / "gemma-dfss-merged"    # Merged full model
GGUF_OUTPUT = BASE_DIR / "gemma-dfss-Q4_K_M.gguf"
MODELFILE   = BASE_DIR / "Modelfile"
LLAMA_CPP_DIR = BASE_DIR / "llama.cpp"          # llama.cpp build directory

# ── Model ────────────────────────────────────────────────────────
# RAG-style local install: the Hub repo is downloaded ONCE into
# LOCAL_MODEL_DIR (see the local-install cell); every load uses the copy.
HF_REPO_ID      = "google/gemma-4-E4B-it"   # source repo on the Hub — used ONLY during installation
                                            # (NB: "google/gemma-4-4b-it" does NOT exist; the ~4B Gemma 4
                                            #  is E4B, the same model the RAG side uses: gemma4:e4b)
LOCAL_MODEL_DIR = BASE_DIR / "gemma_base"   # local Gemma installation directory
MODEL_NAME      = str(LOCAL_MODEL_DIR)      # ALL model loads use the local copy (never the Hub)
MAX_SEQ_LENGTH = 512

# ── Run mode ─────────────────────────────────────────────────────
SMOKE_TEST    = False     # Task 6: True -> 20-step smoke test; full-training config stays untouched
FORCE_RETRAIN = False     # REQ 5: True -> retrain even if a valid adapter already exists
FORCE_REMERGE = False     # REQ 5: True -> re-merge even if a merged model already exists
RESUME_TRAINING = False   # FIX #3: True -> resume from the latest Trainer checkpoint when a valid adapter exists

# ── LoRA ─────────────────────────────────────────────────────────
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ── Training defaults ────────────────────────────────────────────
DEFAULT_EPOCHS = 3
DEFAULT_BATCH = 1
DEFAULT_GRAD_ACCUM = 8
DEFAULT_LR = 2e-4
DEFAULT_WARMUP = 10
DEFAULT_WEIGHT_DECAY = 0.01

# ── System prompt ────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an expert statistics tutor specializing in Design For Six Sigma (DFSS). "
    "You have deep knowledge of probability theory, descriptive and inferential statistics, "
    "hypothesis testing, ANOVA, regression analysis, and statistical process control. "
    "Answer clearly and step-by-step. Use plain text for math."
)

# ── Ollama (for inference / export step) ─────────────────────────
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL_NAME = "gemma-dfss"