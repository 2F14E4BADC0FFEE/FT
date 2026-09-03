# scientific.py – überarbeitet für stratifizierten Test‑Split
# =============================================================================
# Automatische Evaluierung mit BERTScore, ROUGE-L, BLEU, chrF und Bootstrap-CIs.
# Enthält Kontaminations‑Checker, Human‑Review‑Generator und Cohen's Kappa.
# =============================================================================

import json
import random
from pathlib import Path
import numpy as np
from finetune.utils.settings import BASE_DIR, DATA_SOURCE_DIR

# ── Konfiguration ─────────────────────────────────────────────────
BERTSCORE_LANG  = "de"
BERTSCORE_MODEL = "xlm-roberta-large"

# ── Ausgabedateien (alle innerhalb von BASE_DIR) ─────────────────
EVAL_SUMMARY_PATH    = BASE_DIR / "evaluation_summary.json"
EVAL_DETAILED_PATH   = BASE_DIR / "evaluation_detailed.json"
HUMAN_REVIEW_PATH    = BASE_DIR / "human_review.json"
BEST_EXAMPLES_PATH   = BASE_DIR / "best_25_examples.json"
WORST_EXAMPLES_PATH  = BASE_DIR / "worst_25_examples.json"
BEST_COMPOSITE_PATH  = BASE_DIR / "best_25_composite_examples.json"
WORST_COMPOSITE_PATH = BASE_DIR / "worst_25_composite_examples.json"

# ── Standard‑Eingabedateien ───────────────────────────────────────
GENERATED_ANSWERS_PATH = BASE_DIR / "finetuned_answers.json"

# Nur noch der Test‑Split ist die Referenz
REFERENCE_QA_CANDIDATES = [
    BASE_DIR / "test_qa.json",
    DATA_SOURCE_DIR / "test_qa.json",
]

# Für den Kontaminationscheck: wo liegt der Trainings‑Split?
TRAIN_QA_CANDIDATES = [
    BASE_DIR / "train_qa.json",
    DATA_SOURCE_DIR / "train_qa.json",
]


# ══════════════════════════════════════════════════════════════════
# Hilfsfunktionen
# ══════════════════════════════════════════════════════════════════

def normalize_question(text: str) -> str:
    """Vereinheitlicht Fragen für Matching (Kleinschreibung, Whitespace)."""
    return " ".join((text or "").split()).lower()


def _read_json(path):
    """Liest eine JSON‑Datei und gibt eine Liste zurück (unterstützt top‑level dict)."""
    p = Path(path)
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("questions", data.get("data", []))
    return data if isinstance(data, list) else []


def _resolve_reference_file(reference_path=None):
    """
    Findet die Referenz‑QA‑Datei (nur noch test_qa.json).
    1) Falls `reference_path` übergeben wurde und existiert – verwende sie.
    2) Sonst durchsuche die Kandidatenliste.
    """
    if reference_path is not None:
        p = Path(reference_path)
        if p.exists():
            return str(p)

    for cand in REFERENCE_QA_CANDIDATES:
        if Path(cand).exists():
            return str(cand)
    return None


def _resolve_train_file(train_path=None):
    """Analog: findet train_qa.json für die Kontaminationsanalyse."""
    if train_path is not None:
        p = Path(train_path)
        if p.exists():
            return str(p)

    for cand in TRAIN_QA_CANDIDATES:
        if Path(cand).exists():
            return str(cand)
    return None


def _align(generated_records, reference_records):
    """
    Paart generierte und Referenz‑Antworten anhand der normalisierten Frage.
    Rückgabe: (questions, references, hypotheses)
    """
    ref_lookup = {}
    for r in reference_records:
        q = normalize_question(r.get("question"))
        a = (r.get("answer") or "").strip()
        if q and a and q not in ref_lookup:
            ref_lookup[q] = a

    questions, refs, hyps = [], [], []
    for g in generated_records:
        q_norm = normalize_question(g.get("question"))
        q_disp = (g.get("question") or "").strip()
        h = (g.get("answer") or "").strip()
        if not q_norm or not h:
            continue
        ref = ref_lookup.get(q_norm)
        if ref is None:
            continue
        questions.append(q_disp)
        refs.append(ref)
        hyps.append(h)

    return questions, refs, hyps


def _dist_stats(values):
    """Mittelwert, Std, Median, p25, p75 für eine Liste von Werten."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0}
    return {
        "mean":   float(np.mean(arr)),
        "std":    float(np.std(arr)),
        "median": float(np.median(arr)),
        "p25":    float(np.percentile(arr, 25)),
        "p75":    float(np.percentile(arr, 75)),
    }


# ══════════════════════════════════════════════════════════════════
# 1) Bootstrap‑Konfidenzintervalle
# ══════════════════════════════════════════════════════════════════

def bootstrap_ci(scores, n_iterations=5000, confidence=0.95, seed=42):
    """Perzentil‑Bootstrap für den Mittelwert einer Score‑Liste."""
    vals = np.asarray([s for s in scores if s is not None], dtype=float)
    if vals.size == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}

    rng = np.random.default_rng(seed)
    n = vals.size
    boot_means = np.empty(n_iterations, dtype=float)
    for i in range(n_iterations):
        sample_idx = rng.integers(0, n, size=n)
        boot_means[i] = vals[sample_idx].mean()

    alpha = (1.0 - confidence) / 2.0
    ci_low = float(np.percentile(boot_means, 100.0 * alpha))
    ci_high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha)))
    return {"mean": float(vals.mean()), "ci_low": ci_low, "ci_high": ci_high}


# ══════════════════════════════════════════════════════════════════
# 2) Trainings‑/Test‑Kontamination
# ══════════════════════════════════════════════════════════════════

def detect_contamination(train_path=None, eval_path=None):
    """
    Exakte Überlappung zwischen Trainings‑ und Evaluierungsdaten erkennen.
    Verwendet train_qa.json und test_qa.json.
    """
    train_records = _read_json(train_path) if train_path else []
    eval_records  = _read_json(eval_path) if eval_path else []

    print("=" * 60)
    print("CONTAMINATION CHECK (train_qa.json vs test_qa.json)")
    print("=" * 60)

    if not eval_records:
        msg = ("Contamination analysis cannot be performed because no "
               "evaluation dataset (test_qa.json) was provided.")
        print(msg)
        return {"contamination_check": "not_performed", "reason": msg}

    if not train_records:
        msg = ("No training records loaded – check if train_qa.json is present. "
               "Overlap check will be skipped.")
        print(msg)
        return {"contamination_check": "not_performed", "reason": msg}

    def _q_set(records):
        return {normalize_question(r.get("question"))
                for r in records if normalize_question(r.get("question"))}

    def _a_set(records):
        return {(r.get("answer") or "").strip()
                for r in records if (r.get("answer") or "").strip()}

    train_q = _q_set(train_records)
    eval_q  = _q_set(eval_records)
    train_a = _a_set(train_records)
    eval_a  = _a_set(eval_records)

    q_overlap = len(train_q & eval_q)
    a_overlap = len(train_a & eval_a)
    q_pct = 100.0 * q_overlap / max(1, len(eval_q))
    a_pct = 100.0 * a_overlap / max(1, len(eval_a))

    report = {
        "train_size": len(train_records),
        "eval_size": len(eval_records),
        "question_overlap_count": q_overlap,
        "question_overlap_pct": round(q_pct, 2),
        "answer_overlap_count": a_overlap,
        "answer_overlap_pct": round(a_pct, 2),
    }

    print(f"Train: {report['train_size']}  |  Eval: {report['eval_size']}")
    print(f"Fragen-Überlappung: {q_overlap} ({report['question_overlap_pct']}% des Evals)")
    print(f"Antwort-Überlappung: {a_overlap} ({report['answer_overlap_pct']}% des Evals)")
    if q_pct > 0 or a_pct > 0:
        print("⚠️  WARNUNG: Überlappung zwischen Trainings- und Evaluierungsdaten!")
    else:
        print("✅ Keine exakte Überlappung gefunden.")
    return report


# ══════════════════════════════════════════════════════════════════
# 3) Haupt‑Evaluierung
# ══════════════════════════════════════════════════════════════════

def evaluate_answers(generated_path=None, reference_path=None,
                     n_bootstrap=5000, confidence=0.95, seed=42,
                     bertscore_lang=None):
    """
    Führt BERTScore, ROUGE‑L, BLEU und chrF durch, berechnet CIs
    und schreibt detaillierte Ergebnisse sowie Best‑/Worst‑25‑Listen.
    """
    import evaluate
    from bert_score import score as bertscore_fn

    lang = bertscore_lang or BERTSCORE_LANG

    # Daten laden
    gen_path = generated_path or GENERATED_ANSWERS_PATH
    generated = _read_json(gen_path)
    ref_file = _resolve_reference_file(reference_path)
    if not ref_file:
        raise RuntimeError(
            "Keine Referenz‑QA‑Datei gefunden. "
            "Bitte lege test_qa.json im BASE_DIR oder DATA_SOURCE_DIR ab."
        )
    references = _read_json(ref_file)

    questions, refs, hyps = _align(generated, references)
    n = len(questions)

    # Matching‑Report
    matched_q = {normalize_question(q) for q in questions}
    total_ref = len(references)
    total_gen = len(generated)
    unmatched_ref = sum(1 for r in references if normalize_question(r.get("question")) not in matched_q)
    unmatched_gen = sum(1 for g in generated   if normalize_question(g.get("question")) not in matched_q)
    coverage = (100.0 * n / total_ref) if total_ref else 0.0

    print("=" * 60)
    print(f"EVALUATION  |  lang={lang}  model={BERTSCORE_MODEL}  |  {n} matched examples")
    print("=" * 60)
    print(f"Reference: {total_ref}  Generated: {total_gen}  "
          f"Matched: {n}  Unmatched ref: {unmatched_ref}  Unmatched gen: {unmatched_gen}")
    print(f"Coverage: {coverage:.2f}% of reference examples")

    if n == 0:
        # Leeres Ergebnis
        empty_summary = {
            "bertscore_mean": 0.0, "bertscore_ci_low": 0.0, "bertscore_ci_high": 0.0,
            "bertscore_std": 0.0, "bertscore_median": 0.0, "bertscore_p25": 0.0, "bertscore_p75": 0.0,
            "rougeL_mean": 0.0, "rougeL_ci_low": 0.0, "rougeL_ci_high": 0.0,
            "rougeL_std": 0.0, "rougeL_median": 0.0, "rougeL_p25": 0.0, "rougeL_p75": 0.0,
            "bleu_mean": 0.0, "bleu_ci_low": 0.0, "bleu_ci_high": 0.0,
            "bleu_std": 0.0, "bleu_median": 0.0, "bleu_p25": 0.0, "bleu_p75": 0.0,
            "corpus_bleu": 0.0,
            "chrf_mean": 0.0, "chrf_ci_low": 0.0, "chrf_ci_high": 0.0,
            "chrf_std": 0.0, "chrf_median": 0.0, "chrf_p25": 0.0, "chrf_p75": 0.0,
            "total_reference_examples": total_ref,
            "total_generated_examples": total_gen,
            "matched_examples": 0,
            "unmatched_reference_examples": unmatched_ref,
            "unmatched_generated_examples": unmatched_gen,
            "coverage_pct": coverage,
            "avg_reference_length": 0.0,
            "avg_generated_length": 0.0,
            "avg_length_ratio": 0.0,
            "length_bias_correlation": None,
            "bootstrap_iterations": n_bootstrap,
            "confidence_level": confidence,
            "random_seed": seed,
            "bertscore_lang": lang,
            "bertscore_model": BERTSCORE_MODEL,
            "num_examples": 0,
        }
        for path in [EVAL_DETAILED_PATH, BEST_EXAMPLES_PATH, WORST_EXAMPLES_PATH,
                     BEST_COMPOSITE_PATH, WORST_COMPOSITE_PATH]:
            with open(path, "w") as f:
                json.dump([], f, indent=2, ensure_ascii=False)
        with open(EVAL_SUMMARY_PATH, "w") as f:
            json.dump(empty_summary, f, indent=2, ensure_ascii=False)
        print("⚠️  Keine passenden Paare gefunden.")
        return empty_summary

    # Metriken berechnen
    print("Computing BERTScore F1 ...")
    _, _, bert_f1 = bertscore_fn(hyps, refs, lang=lang, model_type=BERTSCORE_MODEL)
    bert_scores = [float(x) for x in bert_f1]

    print("Computing ROUGE-L ...")
    rouge = evaluate.load("rouge")
    rouge_out = rouge.compute(predictions=hyps, references=refs, use_aggregator=False)
    rouge_scores = [float(x.fmeasure if hasattr(x, "fmeasure") else x) for x in rouge_out["rougeL"]]

    print("Computing BLEU ...")
    bleu = evaluate.load("bleu")
    bleu_scores = []
    for h, r in zip(hyps, refs):
        try:
            bleu_scores.append(float(bleu.compute(predictions=[h], references=[[r]])["bleu"]))
        except Exception:
            bleu_scores.append(0.0)
    try:
        corpus_bleu = float(bleu.compute(predictions=hyps, references=[[r] for r in refs])["bleu"])
    except Exception:
        corpus_bleu = 0.0

    print("Computing chrF ...")
    chrf = evaluate.load("chrf")
    chrf_scores = []
    for h, r in zip(hyps, refs):
        try:
            chrf_scores.append(float(chrf.compute(predictions=[h], references=[[r]])["score"]) / 100.0)
        except Exception:
            chrf_scores.append(0.0)

    # Per‑sample Daten
    per_sample = []
    for i in range(n):
        per_sample.append({
            "question": questions[i],
            "reference_answer": refs[i],
            "generated_answer": hyps[i],
            "bertscore_f1": bert_scores[i],
            "rougeL": rouge_scores[i],
            "bleu": bleu_scores[i],
        })
    detailed = [dict(row, chrf=chrf_scores[i]) for i, row in enumerate(per_sample)]

    with open(EVAL_DETAILED_PATH, "w") as f:
        json.dump(detailed, f, indent=2, ensure_ascii=False)

    best_25  = sorted(per_sample, key=lambda d: d["bertscore_f1"], reverse=True)[:25]
    worst_25 = sorted(per_sample, key=lambda d: d["bertscore_f1"])[:25]
    with open(BEST_EXAMPLES_PATH, "w") as f:
        json.dump(best_25, f, indent=2, ensure_ascii=False)
    with open(WORST_EXAMPLES_PATH, "w") as f:
        json.dump(worst_25, f, indent=2, ensure_ascii=False)

    # Composite‑Ranking (0.5 * BERTScore + 0.3 * ROUGE‑L + 0.2 * chrF)
    composite = [
        dict(row, composite_score=(0.5 * bert_scores[i]
                                   + 0.3 * rouge_scores[i]
                                   + 0.2 * chrf_scores[i]))
        for i, row in enumerate(detailed)
    ]
    best_c = sorted(composite, key=lambda d: d["composite_score"], reverse=True)[:25]
    worst_c = sorted(composite, key=lambda d: d["composite_score"])[:25]
    with open(BEST_COMPOSITE_PATH, "w") as f:
        json.dump(best_c, f, indent=2, ensure_ascii=False)
    with open(WORST_COMPOSITE_PATH, "w") as f:
        json.dump(worst_c, f, indent=2, ensure_ascii=False)

    # Statistiken & Bootstrap
    bert_stats  = _dist_stats(bert_scores)
    rouge_stats = _dist_stats(rouge_scores)
    bleu_stats  = _dist_stats(bleu_scores)
    chrf_stats  = _dist_stats(chrf_scores)

    bert_ci  = bootstrap_ci(bert_scores, n_iterations=n_bootstrap, confidence=confidence, seed=seed)
    rouge_ci = bootstrap_ci(rouge_scores, n_iterations=n_bootstrap, confidence=confidence, seed=seed)
    bleu_ci  = bootstrap_ci(bleu_scores, n_iterations=n_bootstrap, confidence=confidence, seed=seed)
    chrf_ci  = bootstrap_ci(chrf_scores, n_iterations=n_bootstrap, confidence=confidence, seed=seed)

    # Längenanalyse
    ref_len = np.array([len(r) for r in refs], dtype=float)
    gen_len = np.array([len(h) for h in hyps], dtype=float)
    ratios = [len(h) / len(r) for h, r in zip(hyps, refs) if len(r) > 0]
    ratio_arr = np.array(ratios, dtype=float)
    bert_arr = np.array(bert_scores, dtype=float)

    if bert_arr.size >= 2 and ratio_arr.size == bert_arr.size and np.std(bert_arr) > 0 and np.std(ratio_arr) > 0:
        length_bias_corr = float(np.corrcoef(bert_arr, ratio_arr)[0, 1])
    else:
        length_bias_corr = None

    summary = {
        "bertscore_mean": bert_stats["mean"], "bertscore_ci_low": bert_ci["ci_low"],
        "bertscore_ci_high": bert_ci["ci_high"], "bertscore_std": bert_stats["std"],
        "bertscore_median": bert_stats["median"], "bertscore_p25": bert_stats["p25"],
        "bertscore_p75": bert_stats["p75"],
        "rougeL_mean": rouge_stats["mean"], "rougeL_ci_low": rouge_ci["ci_low"],
        "rougeL_ci_high": rouge_ci["ci_high"], "rougeL_std": rouge_stats["std"],
        "rougeL_median": rouge_stats["median"], "rougeL_p25": rouge_stats["p25"],
        "rougeL_p75": rouge_stats["p75"],
        "bleu_mean": bleu_stats["mean"], "bleu_ci_low": bleu_ci["ci_low"],
        "bleu_ci_high": bleu_ci["ci_high"], "bleu_std": bleu_stats["std"],
        "bleu_median": bleu_stats["median"], "bleu_p25": bleu_stats["p25"],
        "bleu_p75": bleu_stats["p75"], "corpus_bleu": corpus_bleu,
        "chrf_mean": chrf_ci["mean"], "chrf_ci_low": chrf_ci["ci_low"],
        "chrf_ci_high": chrf_ci["ci_high"], "chrf_std": chrf_stats["std"],
        "chrf_median": chrf_stats["median"], "chrf_p25": chrf_stats["p25"],
        "chrf_p75": chrf_stats["p75"],
        "total_reference_examples": total_ref,
        "total_generated_examples": total_gen,
        "matched_examples": n,
        "unmatched_reference_examples": unmatched_ref,
        "unmatched_generated_examples": unmatched_gen,
        "coverage_pct": coverage,
        "avg_reference_length": float(np.mean(ref_len)),
        "avg_generated_length": float(np.mean(gen_len)),
        "avg_length_ratio": float(np.mean(ratio_arr)) if ratio_arr.size else 0.0,
        "length_bias_correlation": length_bias_corr,
        "bootstrap_iterations": n_bootstrap,
        "confidence_level": confidence,
        "random_seed": seed,
        "bertscore_lang": lang,
        "bertscore_model": BERTSCORE_MODEL,
        "num_examples": n,
    }

    with open(EVAL_SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Konsolenausgabe
    print(f"BERTScore F1: {summary['bertscore_mean']:.4f}  "
          f"[{summary['bertscore_ci_low']:.4f}, {summary['bertscore_ci_high']:.4f}]  "
          f"(std={summary['bertscore_std']:.4f}, median={summary['bertscore_median']:.4f})")
    print(f"ROUGE-L:      {summary['rougeL_mean']:.4f}  "
          f"[{summary['rougeL_ci_low']:.4f}, {summary['rougeL_ci_high']:.4f}]  "
          f"(std={summary['rougeL_std']:.4f}, median={summary['rougeL_median']:.4f})")
    print(f"BLEU:         {summary['bleu_mean']:.4f}  "
          f"[{summary['bleu_ci_low']:.4f}, {summary['bleu_ci_high']:.4f}]  "
          f"(std={summary['bleu_std']:.4f}, median={summary['bleu_median']:.4f})")
    print(f"Corpus BLEU:  {summary['corpus_bleu']:.4f}")
    print(f"chrF (sec.):  {summary['chrf_mean']:.4f}  "
          f"[{summary['chrf_ci_low']:.4f}, {summary['chrf_ci_high']:.4f}]  "
          f"(std={summary['chrf_std']:.4f}, median={summary['chrf_median']:.4f})")
    print(f"Lengths: ref={summary['avg_reference_length']:.1f}  "
          f"gen={summary['avg_generated_length']:.1f}  ratio={summary['avg_length_ratio']:.3f}")
    lbc = summary["length_bias_correlation"]
    print(f"Length-bias correlation (BERTScore vs length ratio): "
          f"{lbc:.4f}" if lbc is not None else
          "Length-bias correlation (BERTScore vs length ratio): n/a")
    print(f"Coverage: {summary['coverage_pct']:.2f}%")
    print(f"✅ Summary  → {EVAL_SUMMARY_PATH}")
    print(f"✅ Detailed → {EVAL_DETAILED_PATH}")
    print(f"✅ Best 25  → {BEST_EXAMPLES_PATH}")
    print(f"✅ Worst 25 → {WORST_EXAMPLES_PATH}")
    print(f"✅ Best 25 (composite)  → {BEST_COMPOSITE_PATH}")
    print(f"✅ Worst 25 (composite) → {WORST_COMPOSITE_PATH}")

    return summary


# ══════════════════════════════════════════════════════════════════
# 4) Human‑Review‑Set erstellen
# ══════════════════════════════════════════════════════════════════

def create_human_review_set(generated_path=None, reference_path=None,
                            sample_size=100, seed=42, output_path=None):
    """Erzeugt eine Zufallsauswahl für manuelle Inspektion."""
    gen_path = generated_path or GENERATED_ANSWERS_PATH
    generated = _read_json(gen_path)
    ref_file = _resolve_reference_file(reference_path)
    if not ref_file:
        raise RuntimeError(
            "Keine Referenzdatei (test_qa.json) gefunden – Human‑Review‑Set kann nicht erstellt werden."
        )
    references = _read_json(ref_file)
    questions, refs, hyps = _align(generated, references)

    review = [
        {
            "question": q,
            "reference_answer": r,
            "generated_answer": h,
            "correctness": None,
            "relevance": None,
            "fluency": None,
            "reviewer": None,
            "comments": None,
            "error_category": None,
        }
        for q, r, h in zip(questions, refs, hyps)
    ]

    rng = random.Random(seed)
    if len(review) > sample_size:
        review = rng.sample(review, sample_size)
    else:
        rng.shuffle(review)

    out = Path(output_path) if output_path else HUMAN_REVIEW_PATH
    with open(out, "w", encoding="utf-8") as f:
        json.dump(review, f, indent=2, ensure_ascii=False)
    print(f"✅ Human review set ({len(review)} examples) saved → {out}")
    return review


# ══════════════════════════════════════════════════════════════════
# 5) Cohen's Kappa (manuell aufzurufen)
# ══════════════════════════════════════════════════════════════════

def compute_cohens_kappa(review_file=None):
    """Berechnet Cohen's Kappa für zwei Reviewer."""
    path = Path(review_file) if review_file else HUMAN_REVIEW_PATH
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)

    reviewers = sorted({it.get("reviewer") for it in items if it.get("reviewer") is not None})
    if len(reviewers) < 2:
        return {"error": "Benötigt zwei Reviewer.", "reviewers_found": reviewers}

    r1, r2 = reviewers[0], reviewers[1]
    by_q1 = {normalize_question(it.get("question")): it for it in items if it.get("reviewer") == r1}
    by_q2 = {normalize_question(it.get("question")): it for it in items if it.get("reviewer") == r2}
    common_q = [q for q in by_q1 if q in by_q2]

    def _kappa(a, b):
        if not a:
            return None
        labels = sorted({*a, *b})
        idx = {lab: i for i, lab in enumerate(labels)}
        m = np.zeros((len(labels), len(labels)), dtype=float)
        for x, y in zip(a, b):
            m[idx[x], idx[y]] += 1.0
        total = m.sum()
        po = np.trace(m) / total
        pe = float(np.sum((m.sum(axis=1) / total) * (m.sum(axis=0) / total)))
        if abs(1.0 - pe) < 1e-12:
            return 1.0 if po >= 1.0 else None
        return float((po - pe) / (1.0 - pe))

    result = {"reviewers": [r1, r2]}
    for dim in ("correctness", "relevance", "fluency"):
        a_vals, b_vals = [], []
        for q in common_q:
            va = by_q1[q].get(dim)
            vb = by_q2[q].get(dim)
            if va is not None and vb is not None:
                a_vals.append(va)
                b_vals.append(vb)
        result[dim] = {"kappa": _kappa(a_vals, b_vals), "n": len(a_vals)}
    return result


# ══════════════════════════════════════════════════════════════════
# 6) Einstiegsfunktion (aufrufbar ohne Argumente)
# ══════════════════════════════════════════════════════════════════

def run_evaluation(generated_path=None, reference_path=None):
    """
    Führt die gesamte Evaluierungspipeline aus:
    1. Kontaminationscheck (train_qa.json vs test_qa.json)
    2. Human‑Review‑Set erstellen
    3. Automatische Metriken berechnen
    """
    ref_file = _resolve_reference_file(reference_path)
    if ref_file is None:
        print("❌ Keine Referenzdatei (test_qa.json) gefunden – Evaluierung wird übersprungen.")
        return

    # Trainingsdatei für Kontaminationscheck automatisch suchen
    train_file = _resolve_train_file()
    detect_contamination(train_path=train_file, eval_path=ref_file)

    create_human_review_set(generated_path=generated_path, reference_path=ref_file)
    evaluate_answers(generated_path=generated_path, reference_path=ref_file)