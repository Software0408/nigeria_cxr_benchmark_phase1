#!/usr/bin/env python3
# scripts/nlp/run_extraction.py
"""
Run ensemble label extraction on the full FMC EB report dataset.

Week 8 — March 2026.

Key features:
    - Multi-fold BERT: loads all 5 CV fold models and averages their probabilities
    - Checkpoint/resume: saves progress every N reports, auto-resumes on restart
    - Single-pass extraction: one ensemble call per report
    - Full audit trail: per-label raw predictions from both methods
    - Dynamic report discovery: no hardcoded counts

Usage:
    python scripts/nlp/run_extraction.py                       # start or auto-resume
    python scripts/nlp/run_extraction.py --fresh                # ignore checkpoints, start fresh
    python scripts/nlp/run_extraction.py --checkpoint-every 50  # save every 50 reports
    python scripts/nlp/run_extraction.py --reports-dir /path    # custom reports directory

Output:
    results/nlp/ensemble_predictions.csv     — full predictions with per-method outputs
    results/nlp/disagreement_log.csv         — all disagreements with full report text
    results/nlp/label_distribution.json      — prevalence and co-occurrence statistics
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np
from tqdm import tqdm

# scripts/nlp/run_extraction.py -> nlp/ -> scripts/ -> repo root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nlp.labels import LabelHierarchy  # noqa: E402
from nlp.extractors.bioclinicalbert_extractor import (  # noqa: E402
    BioClinicalBERTExtractor,
    PATHOLOGY_LABELS,
)
from nlp.extractors.ensemble import EnsembleExtractor  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CHECKPOINT_DIR = PROJECT_ROOT / "results" / "nlp" / ".extraction_checkpoints"


def _short(label: str) -> str:
    return label.replace(" / ", "_").replace(" ", "_")


# ---- Report discovery ----

def discover_reports(reports_dir: Path) -> pd.DataFrame:
    report_files = sorted(reports_dir.glob("*.txt"))
    if not report_files:
        raise FileNotFoundError(f"No .txt report files found in {reports_dir}")

    records = []
    for fpath in tqdm(report_files, desc="Reading reports", unit="file"):
        study_id = fpath.stem
        text = fpath.read_text(encoding="utf-8", errors="replace").strip()
        if text:
            records.append({"study_id": study_id, "report_text": text})

    df = pd.DataFrame(records)
    logger.info(f"Discovered {len(df)} non-empty reports out of {len(report_files)} files")
    return df


# ---- Multi-fold BERT loading ----

def load_bert_folds(models_dir: Path) -> tuple:
    """
    Discover and load all available BERT fold models.
    Returns (list of BioClinicalBERTExtractor, averaged thresholds dict).
    """
    fold_dirs = sorted(models_dir.glob("bert_fold*"))
    if not fold_dirs:
        logger.error(f"No BERT fold models found in {models_dir}. Run train_bert_extractor.py first.")
        sys.exit(1)

    extractors = []
    all_thresholds = []

    for fold_dir in fold_dirs:
        if not (fold_dir / "model.pt").exists():
            logger.warning(f"Skipping {fold_dir.name}: model.pt not found")
            continue

        logger.info(f"Loading {fold_dir.name}...")
        extractor = BioClinicalBERTExtractor.load(str(fold_dir))
        extractors.append(extractor)

        # Load per-fold thresholds
        thresh_path = fold_dir / "thresholds.json"
        if thresh_path.exists():
            with open(thresh_path) as f:
                all_thresholds.append(json.load(f))

    if not extractors:
        logger.error("No valid BERT fold models found.")
        sys.exit(1)

    # Average thresholds across folds
    avg_thresholds = {}
    if all_thresholds:
        for label in PATHOLOGY_LABELS:
            vals = [t.get(label, 0.5) for t in all_thresholds]
            avg_thresholds[label] = round(float(np.mean(vals)), 2)

    logger.info(f"Loaded {len(extractors)} BERT fold models")
    return extractors, avg_thresholds


# ---- Checkpointing ----

def save_checkpoint(
    pred_records: List[Dict],
    dis_records: List[Dict],
    processed_ids: set,
):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(pred_records).to_csv(CHECKPOINT_DIR / "pred_partial.csv", index=False)
    pd.DataFrame(dis_records).to_csv(CHECKPOINT_DIR / "dis_partial.csv", index=False)
    with open(CHECKPOINT_DIR / "processed_ids.json", "w") as f:
        json.dump(sorted(processed_ids), f)

    logger.info(f"  Checkpoint saved: {len(processed_ids)} reports processed")


def load_checkpoint() -> tuple:
    pred_path = CHECKPOINT_DIR / "pred_partial.csv"
    dis_path = CHECKPOINT_DIR / "dis_partial.csv"
    ids_path = CHECKPOINT_DIR / "processed_ids.json"

    if not all(p.exists() for p in [pred_path, ids_path]):
        return [], [], set()

    pred_df = pd.read_csv(pred_path)
    pred_records = pred_df.to_dict("records")

    dis_records = []
    if dis_path.exists() and dis_path.stat().st_size > 0:
        dis_df = pd.read_csv(dis_path)
        dis_records = dis_df.to_dict("records")

    with open(ids_path) as f:
        processed_ids = set(json.load(f))

    logger.info(f"Checkpoint loaded: {len(processed_ids)} reports already processed")
    return pred_records, dis_records, processed_ids


def clear_checkpoints():
    if CHECKPOINT_DIR.exists():
        for f in CHECKPOINT_DIR.iterdir():
            f.unlink()
        CHECKPOINT_DIR.rmdir()
        logger.info("Extraction checkpoints cleared")


# ---- Single-pass extraction with checkpointing ----

def run_extraction(
    df: pd.DataFrame,
    ensemble: EnsembleExtractor,
    checkpoint_every: int = 100,
    resume: bool = True,
) -> tuple:
    """
    Run ensemble on every report with periodic checkpointing.
    Returns (pred_records, dis_records).
    """
    # Try to resume
    if resume:
        pred_records, dis_records, processed_ids = load_checkpoint()
        if processed_ids:
            remaining = len(df) - len(processed_ids)
            logger.info(f"Resuming: {len(processed_ids)} done, {remaining} remaining")
    else:
        pred_records, dis_records, processed_ids = [], [], set()

    since_checkpoint = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting labels",
                       unit="report", initial=len(processed_ids)):
        study_id = row["study_id"]

        # Skip already-processed reports
        if study_id in processed_ids:
            continue

        report_text = row["report_text"]
        result = ensemble.extract(report_text)

        # ---- Prediction record ----
        record = {
            "study_id": study_id,
            "report_text": report_text,
            "report_length": len(report_text),
            "final_labels": ",".join(result["labels"]) if result["labels"] else "Normal",
            "num_labels": len(result["labels"]),
            "is_normal": int(result["meta"]["is_normal"]),
            "rb_labels_str": ",".join(result["rb_labels"]) if result["rb_labels"] else "Normal",
            "bert_labels_str": ",".join(result["bert_labels"]) if result["bert_labels"] else "Normal",
            "rb_count": result["meta"]["rb_count"],
            "bert_count": result["meta"]["bert_count"],
            "disagreement_count": result["meta"]["disagreement_count"],
            "has_normal_phrase": int(result["meta"]["has_normal_phrase"]),
            "n_bert_folds": result["meta"]["n_bert_folds"],
        }

        for label in PATHOLOGY_LABELS:
            s = _short(label)
            record[f"rb_pred_{s}"] = int(label in result["rb_labels"])
            record[f"bert_pred_{s}"] = int(label in result["bert_labels"])
            record[f"pred_{s}"] = int(label in result["labels"])
            record[f"bert_prob_{s}"] = round(result["bert_scores"].get(label, 0.0), 4)
            record[f"conf_{s}"] = result["confidence"].get(label, "high")

        pred_records.append(record)

        # ---- Disagreement records ----
        for dis in result["disagreements"]:
            dis_records.append({
                "study_id": study_id,
                "label": dis["label"],
                "policy": dis["policy"],
                "rb_detected": int(dis["rb_detected"]),
                "bert_detected": int(dis["bert_detected"]),
                "bert_prob": dis["bert_prob"],
                "rb_score": dis["rb_score"],
                "decision": dis["decision"],
                "reason": dis["reason"],
                "final_labels": record["final_labels"],
                "rb_labels": record["rb_labels_str"],
                "bert_labels": record["bert_labels_str"],
                "report_text": report_text,
                "report_length": len(report_text),
            })

        processed_ids.add(study_id)
        since_checkpoint += 1

        # Periodic checkpoint
        if since_checkpoint >= checkpoint_every:
            save_checkpoint(pred_records, dis_records, processed_ids)
            since_checkpoint = 0

    # Final checkpoint
    if since_checkpoint > 0:
        save_checkpoint(pred_records, dis_records, processed_ids)

    return pred_records, dis_records


# ---- Label distribution ----

def compute_label_distribution(predictions_df: pd.DataFrame) -> Dict:
    total = len(predictions_df)
    normal_count = int(predictions_df["is_normal"].sum())

    label_counts = {}
    for label in PATHOLOGY_LABELS:
        s = _short(label)
        label_counts[label] = {
            "ensemble_count": int(predictions_df[f"pred_{s}"].sum()),
            "rb_count": int(predictions_df[f"rb_pred_{s}"].sum()),
            "bert_count": int(predictions_df[f"bert_pred_{s}"].sum()),
        }
        label_counts[label]["ensemble_pct"] = round(
            label_counts[label]["ensemble_count"] / total * 100, 2
        )

    num_labels = predictions_df["num_labels"]

    conf_counts = {"high": 0, "medium": 0, "low": 0}
    for label in PATHOLOGY_LABELS:
        s = _short(label)
        col = f"conf_{s}"
        if col in predictions_df.columns:
            for val in predictions_df[col]:
                if val in conf_counts:
                    conf_counts[val] += 1

    dis_total = int(predictions_df["disagreement_count"].sum())
    reports_with_dis = int((predictions_df["disagreement_count"] > 0).sum())

    return {
        "total_reports": total,
        "label_counts": label_counts,
        "multi_label_stats": {
            "normal_count": normal_count,
            "normal_pct": round(normal_count / total * 100, 2),
            "abnormal_count": total - normal_count,
            "abnormal_pct": round((total - normal_count) / total * 100, 2),
            "mean_labels_per_report": round(float(num_labels.mean()), 2),
            "median_labels_per_report": round(float(num_labels.median()), 2),
            "max_labels_per_report": int(num_labels.max()),
            "reports_with_1_label": int((num_labels == 1).sum()),
            "reports_with_2_labels": int((num_labels == 2).sum()),
            "reports_with_3plus_labels": int((num_labels >= 3).sum()),
        },
        "confidence_distribution": conf_counts,
        "disagreement_stats": {
            "total_disagreements": dis_total,
            "reports_with_disagreements": reports_with_dis,
            "pct_reports_with_disagreements": round(
                reports_with_dis / total * 100, 2
            ) if total > 0 else 0,
        },
    }


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(description="Run ensemble extraction on full dataset")
    parser.add_argument(
        "--reports-dir", type=str,
        default=str(PROJECT_ROOT / "workspace" / "data" / "preprocessed_data" / "reports"),
        help="Directory containing report .txt files",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=100,
        help="Save checkpoint every N reports (default: 100)",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Ignore existing checkpoints, start from scratch",
    )
    args = parser.parse_args()

    # ---- Load extractors ----

    logger.info("Loading rule-based extractor...")
    rb = LabelHierarchy(
        config_path=str(PROJECT_ROOT / "configs" / "labels" / "hierarchy.yaml"),
        dict_path=str(PROJECT_ROOT / "src" / "nlp" / "dictionary" / "bilingual_dict.json"),
    )

    logger.info("Loading BioClinicalBERT fold models...")
    models_dir = PROJECT_ROOT / "models"
    bert_extractors, avg_thresholds = load_bert_folds(models_dir)

    # Build ensemble with all folds
    ensemble = EnsembleExtractor(
        rule_based_extractor=rb,
        bert_extractors=bert_extractors,
        bert_thresholds=avg_thresholds,
    )

    # ---- Handle fresh start ----

    if args.fresh:
        clear_checkpoints()

    # ---- Discover reports ----

    reports_dir = Path(args.reports_dir)
    df = discover_reports(reports_dir)

    # ---- Run extraction with checkpointing ----

    logger.info(f"\nRunning ensemble extraction on {len(df)} reports "
                f"({len(bert_extractors)} BERT folds, checkpoint every {args.checkpoint_every})...")

    pred_records, dis_records = run_extraction(
        df, ensemble,
        checkpoint_every=args.checkpoint_every,
        resume=not args.fresh,
    )

    predictions_df = pd.DataFrame(pred_records)
    disagreement_df = pd.DataFrame(dis_records)

    # ---- Label distribution ----

    logger.info("\nComputing label distribution...")
    distribution = compute_label_distribution(predictions_df)

    # ---- Save final outputs ----

    results_dir = PROJECT_ROOT / "results" / "nlp"
    results_dir.mkdir(parents=True, exist_ok=True)

    pred_path = results_dir / "ensemble_predictions.csv"
    predictions_df.to_csv(pred_path, index=False)
    logger.info(f"Predictions saved: {pred_path} ({len(predictions_df)} reports)")

    dis_path = results_dir / "disagreement_log.csv"
    disagreement_df.to_csv(dis_path, index=False)
    dis_reports = disagreement_df["study_id"].nunique() if len(disagreement_df) > 0 else 0
    logger.info(f"Disagreement log saved: {dis_path} ({len(disagreement_df)} records across {dis_reports} reports)")

    dist_path = results_dir / "label_distribution.json"
    with open(dist_path, "w") as f:
        json.dump(distribution, f, indent=2)
    logger.info(f"Label distribution saved: {dist_path}")

    # Clean up checkpoints on successful completion
    clear_checkpoints()

    # ---- Print summary ----

    stats = distribution["multi_label_stats"]
    dis_stats = distribution["disagreement_stats"]
    conf = distribution["confidence_distribution"]

    logger.info(f"\n{'=' * 70}")
    logger.info(f"EXTRACTION COMPLETE — {distribution['total_reports']} reports ({len(bert_extractors)} BERT folds)")
    logger.info(f"{'=' * 70}")
    logger.info(
        f"Normal: {stats['normal_count']} ({stats['normal_pct']}%) | "
        f"Abnormal: {stats['abnormal_count']} ({stats['abnormal_pct']}%)"
    )
    logger.info(
        f"Labels/report: mean={stats['mean_labels_per_report']}, "
        f"median={stats['median_labels_per_report']}, max={stats['max_labels_per_report']}"
    )
    logger.info(
        f"Multi-label: 1={stats['reports_with_1_label']}, "
        f"2={stats['reports_with_2_labels']}, 3+={stats['reports_with_3plus_labels']}"
    )

    logger.info(f"\n{'Label':<42} {'Ensemble':>8} {'RB':>6} {'BERT':>6} {'Ens%':>6}")
    logger.info(f"{'-' * 70}")
    for label in PATHOLOGY_LABELS:
        info = distribution["label_counts"].get(label, {})
        logger.info(
            f"{label:<42} {info.get('ensemble_count', 0):>8} "
            f"{info.get('rb_count', 0):>6} {info.get('bert_count', 0):>6} "
            f"{info.get('ensemble_pct', 0):>5.1f}%"
        )

    logger.info(
        f"\nDisagreements: {dis_stats['total_disagreements']} across "
        f"{dis_stats['reports_with_disagreements']} reports "
        f"({dis_stats['pct_reports_with_disagreements']:.1f}%)"
    )
    logger.info(f"Confidence: high={conf['high']}, medium={conf['medium']}, low={conf['low']}")
    logger.info("\nDone!")


if __name__ == "__main__":
    main()
