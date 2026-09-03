import json
from pathlib import Path
from typing import List, Dict, Any
from finetune.utils.settings import BASE_DIR, DATA_SOURCE_DIR, SYSTEM_PROMPT


def _find_file(filename: str) -> Path:
    """Sucht eine Datei zuerst im schreibbaren BASE_DIR, dann im Read‑Only‑Input."""
    for root in (BASE_DIR, DATA_SOURCE_DIR):
        candidate = root / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Datei '{filename}' weder in {BASE_DIR} noch in {DATA_SOURCE_DIR} gefunden."
    )


def _load_json(filename: str) -> List[Dict[str, Any]]:
    """Lädt eine JSON‑Datei (Liste von Dicts) aus dem gefundenen Pfad."""
    path = _find_file(filename)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"✅ {len(data)} Einträge aus {path} geladen.")
    return data


def _qa_to_chat_format(qa_list: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Wandelt eine QA‑Liste in das Chat‑Format um, das der Trainer benötigt."""
    return [
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": qa["question"]},
                {"role": "assistant", "content": qa["answer"]},
            ]
        }
        for qa in qa_list
    ]


def load_training_dataset() -> Dict[str, List[Dict[str, Any]]]:
    """
    Lädt Trainings- und Validierungsdaten im Chat‑Format.
    Rückgabe: {'train': [...], 'validation': [...]}
    """
    train_data = _qa_to_chat_format(_load_json("train_qa.json"))
    val_data   = _qa_to_chat_format(_load_json("validation_qa.json"))
    print(f"Training: {len(train_data)} Beispiele, Validation: {len(val_data)} Beispiele")
    return {"train": train_data, "validation": val_data}


def load_evaluation_dataset() -> List[Dict[str, str]]:
    """
    Lädt den Test‑Split (Frage + Referenzantwort) für die wissenschaftliche Evaluation.
    Rückgabe: Liste von {'question': ..., 'answer': ...}
    """
    return _load_json("test_qa.json")