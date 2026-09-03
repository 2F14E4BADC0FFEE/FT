import gc
import torch
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    # prepare_model_for_kbit_training  ← NICHT MEHR VERWENDET
)
from datasets import Dataset
from finetune.utils.settings import (
    DEFAULT_EPOCHS, DEFAULT_BATCH, DEFAULT_GRAD_ACCUM, DEFAULT_LR, DEFAULT_WEIGHT_DECAY,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, MAX_SEQ_LENGTH, MODEL_NAME, TARGET_MODULES,
    OUTPUT_DIR, BASE_DIR, SMOKE_TEST, FORCE_RETRAIN, RESUME_TRAINING,
)
from finetune.models.compat import validate_lora_targets, resolve_lora_targets
from finetune.data.dataset_loader import load_training_dataset
from finetune.data.formatting import format_examples
from finetune.training.state import train_state
from finetune.utils.gpu import force_single_gpu_view
from finetune.utils.checks import has_valid_adapter

def _setup_kbit_training(model):
    """
    Ersetzt prepare_model_for_kbit_training durch eine speicherschonende Variante.
    - Norm-Layer in float32 casten (nötig für Stabilität)
    - enable_input_require_grads() (nötig für LoRA + Gradient Checkpointing)
    """
    for name, module in model.named_modules():
        if 'norm' in name or 'Norm' in name:
            module = module.float()
            for param in module.parameters():
                param.data = param.data.float()
    model.train()
    model.enable_input_require_grads()


def run_training(
    epochs=DEFAULT_EPOCHS,
    batch_size=DEFAULT_BATCH,
    grad_accum=DEFAULT_GRAD_ACCUM,
    lr=DEFAULT_LR,
    warmup=3,
    weight_decay=DEFAULT_WEIGHT_DECAY,
    lora_r=LORA_R,
    lora_alpha=LORA_ALPHA,
    max_seq_len=MAX_SEQ_LENGTH,
    max_steps_override=-1,
    resume_from_checkpoint=None,
):

    # ─────────────────────────────────────────────────────────
    # Free memory from any previous run
    # ─────────────────────────────────────────────────────────
    if train_state.get("model") is not None:
        del train_state["model"]
        train_state["model"] = None
    if train_state.get("tokenizer") is not None:
        del train_state["tokenizer"]
        train_state["tokenizer"] = None
    gc.collect()
    torch.cuda.empty_cache()

    print("=" * 60)
    print(f"FINE-TUNING  |  {datetime.now():%H:%M:%S}")
    print("=" * 60)

    print(f"Model:    {MODEL_NAME}")
    print(f"LoRA:     r={lora_r}, alpha={lora_alpha}")
    print(f"Training: {epochs} epochs, batch={batch_size}×{grad_accum}, lr={lr}")
    print(f"Seq len:  {max_seq_len}")
    print(f"Visible GPUs: {torch.cuda.device_count()}")

    splits = load_training_dataset()
    entries = splits["train"]
    # Optional für spätere Evaluierung während des Trainings:
    # eval_dataset = splits["validation"]

    if not entries:
        raise RuntimeError("No dataset to train on.")

    # ─────────────────────────────────────────────────────────
    # Tokenizer
    # ─────────────────────────────────────────────────────────
    print("\nLoading tokenizer & model...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        local_files_only=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    # ─────────────────────────────────────────────────────────
    # Model (4‑Bit QLoRA mit manuellem Setup – KEIN OOM)
    # ─────────────────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map=None,               # verhindert caching_allocator_warmup
        low_cpu_mem_usage=True,        # sparsames Laden auf CPU
        local_files_only=True,
    )
    model = model.to("cuda")           # manuell auf die GPU

    # Manuelles kbit‑Setup (anstelle von prepare_model_for_kbit_training)
    _setup_kbit_training(model)

    model.config.use_cache = False
    model.train()

    # ─────────────────────────────────────────────────────────
    # LoRA
    # ─────────────────────────────────────────────────────────
    print("Adding LoRA adapters...")

    validate_lora_targets(model, TARGET_MODULES)
    _resolved_targets = resolve_lora_targets(model, TARGET_MODULES)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=LORA_DROPOUT,
        target_modules=_resolved_targets,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()  # nach LoRA nochmal sicherstellen
    model.print_trainable_parameters()

    # ─────────────────────────────────────────────────────────
    # Dataset
    # ─────────────────────────────────────────────────────────
    print("\nFormatting & tokenizing dataset...")

    texts = format_examples(entries, tokenizer)

    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_seq_len,
        padding=False,
        return_tensors=None,
    )

    dataset = Dataset.from_dict(tokenized)

    lens = [len(x) for x in tokenized["input_ids"]]

    print(
        f"Dataset: {len(dataset)} examples | "
        f"tokens min={min(lens)} "
        f"max={max(lens)} "
        f"avg={sum(lens)//len(lens)}"
    )

    # ─────────────────────────────────────────────────────────
    # Collator
    # ─────────────────────────────────────────────────────────
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # ─────────────────────────────────────────────────────────
    # Training args
    # ─────────────────────────────────────────────────────────
    args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,

        num_train_epochs=epochs,
        max_steps=max_steps_override if max_steps_override > 0 else -1,

        learning_rate=lr,
        warmup_steps=warmup,
        weight_decay=weight_decay,

        fp16=True,
        bf16=False,

        logging_steps=5,

        save_strategy="epoch",
        save_total_limit=2,

        output_dir=str(OUTPUT_DIR),
        logging_dir=str(BASE_DIR / "logs"),

        optim="adamw_torch",

        seed=42,

        dataloader_num_workers=0,
        dataloader_pin_memory=True,

        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        max_grad_norm=1.0,

        report_to="none",
    )

    # ─────────────────────────────────────────────────────────
    # Trainer
    # ─────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collator,
    )

    # ─────────────────────────────────────────────────────────
    # Info
    # ─────────────────────────────────────────────────────────
    steps_per_epoch = max(
        1,
        len(dataset) // (batch_size * grad_accum)
    )

    total_steps = (
        steps_per_epoch * epochs
        if args.max_steps < 0
        else args.max_steps
    )

    print(f"\nSteps/epoch: ~{steps_per_epoch}")
    print(f"Total steps: ~{total_steps}")

    print(
        f"Rough ETA on T4: "
        f"{total_steps * 4 // 60}–{total_steps * 12 // 60} min "
        f"(slower with gradient checkpointing)"
    )

    # ─────────────────────────────────────────────────────────
    # Train
    # ─────────────────────────────────────────────────────────
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # ─────────────────────────────────────────────────────────
    # Save
    # ─────────────────────────────────────────────────────────
    print(f"\n✅ Training done at {datetime.now():%H:%M:%S}")

    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    print(f"✅ Adapter saved to {OUTPUT_DIR}")

    train_state["model"] = model
    train_state["tokenizer"] = tokenizer

    return model, tokenizer


def train_model():
    force_single_gpu_view()
    gc.collect()
    torch.cuda.empty_cache()

    if (not SMOKE_TEST) and has_valid_adapter(OUTPUT_DIR) and not FORCE_RETRAIN:
        _last_ckpt = None
        if RESUME_TRAINING:
            from transformers.trainer_utils import get_last_checkpoint
            _last_ckpt = get_last_checkpoint(str(OUTPUT_DIR))
        if _last_ckpt is not None:
            print(f"Existing LoRA adapter detected — resuming from {_last_ckpt}.")
            run_training(epochs=3, batch_size=1, grad_accum=8, lr=2e-4,
                         resume_from_checkpoint=_last_ckpt)
        else:
            print("Existing LoRA adapter detected.")
            print(f"   → Reusing {OUTPUT_DIR}. Set FORCE_RETRAIN=True to retrain from scratch.")
    elif SMOKE_TEST:
        run_training(max_steps_override=20)
    else:
        run_training(
            epochs=3,
            batch_size=1,
            grad_accum=8,
            lr=2e-4,
        )