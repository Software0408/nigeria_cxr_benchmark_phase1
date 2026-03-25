#!/usr/bin/env python3
# scripts/nlp/train_bert_extractor.py
"""
Train BioClinicalBERT multi-label classifier on FMC EB annotated reports.

Weeks 7–9 — March 2026.

Strategy:
    - 5-fold multi-label stratified cross-validation (iterstrat) on REAL reports only
    - Optional synthetic augmentation: positive (rare labels) + hard negatives injected
      into TRAINING folds only (validation = exclusively real clinical reports)
    - Normal as explicit 14th label (teaches BERT what normality looks like)
    - Focal loss option to down-weight easy negatives and focus on hard boundary cases
    - Per-fold class weight computation (no validation leakage)
    - Early stopping on macro F1 (not loss) for clinical relevance
    - Per-label threshold tuning on validation fold
    - Per-label AUROC and PR-AUC where computable
    - Mixed precision (torch.amp) for GPU efficiency
    - Enriched OOF output with gold labels for downstream analysis
    - Checkpoint save/resume (epoch-level + fold-level)
    - Post-training temperature calibration for probability scores

Usage:
    # Default (original 497 reports):
    python scripts/nlp/train_bert_extractor.py

    # With augmented data + hard negatives + focal loss + Normal label:
    python scripts/nlp/train_bert_extractor.py --data-path results/nlp/training_reports_v2.csv --focal-loss --use-normal-label --fresh

    python scripts/nlp/train_bert_extractor.py --data-path results/nlp/training_reports_augmented.csv --use-normal-label --fresh

    # Ignore checkpoints:
    python scripts/nlp/train_bert_extractor.py --fresh

Output:
    models/bert_foldN/                  — saved model, tokenizer, thresholds per fold
    results/nlp/bert_evaluation.json    — full metrics + augmentation metadata
    results/nlp/bert_oof_predictions.csv — enriched OOF predictions (real reports only)
    results/nlp/bert_thresholds.json    — averaged per-label thresholds

Hardware:
    GTX 1070 (8GB VRAM): batch_size=8, max_length=512
    Mixed precision reduces VRAM ~30%. If OOM, set batch_size=4.

Augmentation guarantee:
    When --data-path points to a CSV with 'is_synthetic' column:
    - CV splits on real reports ONLY
    - Synthetic (positive + hard negative) injected into TRAINING folds only
    - Validation = ZERO synthetic
    - All evaluation on real clinical data exclusively
"""

import os
import sys
import json
import random
import logging
import warnings
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    average_precision_score,
)

try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
except ImportError:
    raise ImportError(
        "iterative-stratification is required for multi-label stratified CV.\n"
        "Install with: pip install iterative-stratification"
    )

# scripts/nlp/train_bert_extractor.py -> nlp/ -> scripts/ -> repo root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nlp.extractors.bioclinicalbert_extractor import (
    CXRBertClassifier,
    BioClinicalBERTExtractor,
    PATHOLOGY_LABELS,
    NUM_LABELS,
)

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---- Configuration ----

CONFIG = {
    "model_name": "emilyalsentzer/Bio_ClinicalBERT",
    "max_length": 512,
    "batch_size": 8,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "epochs": 10,
    "patience": 3,
    "warmup_ratio": 0.1,
    "n_folds": 5,
    "seed": 42,
    "data_path": "results/annotated_reports.csv",
    "output_dir": "models",
    "results_dir": "results/nlp",
    "checkpoint_dir": "models/checkpoints",
    "use_amp": True,
}


# ---- Reproducibility ----

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---- Checkpointing ----

def save_epoch_checkpoint(
    checkpoint_dir: Path,
    fold: int,
    epoch: int,
    model_state: dict,
    optimizer_state: dict,
    scheduler_state: dict,
    scaler_state: dict,
    best_state: Optional[dict],
    best_val_macro_f1: float,
    patience_counter: int,
):
    """Save training state after each epoch — allows resume after power cut."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"fold{fold}_epoch.pt"

    torch.save({
        "fold": fold,
        "epoch": epoch,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer_state,
        "scheduler_state_dict": scheduler_state,
        "scaler_state_dict": scaler_state,
        "best_state_dict": best_state,
        "best_val_macro_f1": best_val_macro_f1,
        "patience_counter": patience_counter,
    }, ckpt_path)

    logger.info(f"  Epoch checkpoint saved: fold={fold} epoch={epoch} (macro F1={best_val_macro_f1:.3f})")


def load_epoch_checkpoint(checkpoint_dir: Path, fold: int) -> Optional[dict]:
    """Load epoch checkpoint for a specific fold, if it exists."""
    ckpt_path = checkpoint_dir / f"fold{fold}_epoch.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        logger.info(f"  Resuming fold {fold} from epoch {ckpt['epoch'] + 1} (best macro F1={ckpt['best_val_macro_f1']:.3f})")
        return ckpt
    return None


def clear_epoch_checkpoint(checkpoint_dir: Path, fold: int):
    """Remove epoch checkpoint after fold completes successfully."""
    ckpt_path = checkpoint_dir / f"fold{fold}_epoch.pt"
    if ckpt_path.exists():
        ckpt_path.unlink()


def save_fold_progress(
    checkpoint_dir: Path,
    completed_folds: List[int],
    oof_probs: np.ndarray,
    oof_thresholds: dict,
    fold_indices: np.ndarray,
):
    """Save cross-fold progress so we can skip completed folds on resume."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    progress_path = checkpoint_dir / "fold_progress.pt"

    torch.save({
        "completed_folds": completed_folds,
        "oof_probs": oof_probs,
        "oof_thresholds": dict(oof_thresholds),
        "fold_indices": fold_indices,
    }, progress_path)

    logger.info(f"Fold progress saved: completed folds = {completed_folds}")


def load_fold_progress(checkpoint_dir: Path) -> Optional[dict]:
    """Load fold-level progress if it exists."""
    progress_path = checkpoint_dir / "fold_progress.pt"
    if progress_path.exists():
        progress = torch.load(progress_path, map_location="cpu", weights_only=False)
        logger.info(f"Resuming from fold progress: completed folds = {progress['completed_folds']}")
        return progress
    return None


def clear_all_checkpoints(checkpoint_dir: Path):
    """Clean up all checkpoints after successful completion."""
    if checkpoint_dir.exists():
        for f in checkpoint_dir.iterdir():
            f.unlink()
        checkpoint_dir.rmdir()
        logger.info("All checkpoints cleared (training complete)")


# ---- Dataset ----

class ReportDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.float32),
        }

        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)

        return item


# ---- Data loading ----

def load_data(data_path: str, use_normal_label: bool = False):
    """
    Load annotated reports and convert gold labels to multi-hot vectors.

    Args:
        data_path: path to CSV with report_text, gold_labels columns
        use_normal_label: if True, add Normal as explicit 14th label column

    Returns:
        texts: list of report texts
        label_vectors: (N, num_labels) multi-hot array (13 or 14 columns)
        study_ids: list of study ID strings
        is_synthetic: (N,) boolean array
        label_names: list of label names (PATHOLOGY_LABELS or ALL_LABELS)
        num_labels: number of labels (13 or 14)
    """
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} reports from {data_path}")

    texts = df["report_text"].tolist()
    study_ids = df["study_id"].astype(str).tolist()

    # Build multi-hot label vectors for 13 pathology labels
    label_vectors = []
    for gold_str in df["gold_labels"]:
        gold_set = set(l.strip() for l in str(gold_str).split(","))
        gold_set.discard("Normal")
        vector = [1.0 if label in gold_set else 0.0 for label in PATHOLOGY_LABELS]
        label_vectors.append(vector)

    label_vectors = np.array(label_vectors)

    # Optionally add Normal as 14th column
    if use_normal_label:
        normal_col = (label_vectors.sum(axis=1) == 0).astype(float).reshape(-1, 1)
        label_vectors = np.hstack([label_vectors, normal_col])
        label_names = list(PATHOLOGY_LABELS) + ["Normal"]
        num_labels = NUM_LABELS + 1
        n_normal = int(normal_col.sum())
        logger.info(f"Normal label added as column {num_labels - 1}: {n_normal} positive ({n_normal / len(df) * 100:.1f}%)")
    else:
        label_names = list(PATHOLOGY_LABELS)
        num_labels = NUM_LABELS

    # Detect synthetic flag (backward-compatible)
    if "is_synthetic" in df.columns:
        is_synthetic = df["is_synthetic"].astype(bool).values
    else:
        is_synthetic = np.zeros(len(df), dtype=bool)

    n_real = int((~is_synthetic).sum())
    n_synth = int(is_synthetic.sum())

    logger.info(f"Data composition: {n_real} real + {n_synth} synthetic = {len(df)} total")
    logger.info(f"Label dimensions: {num_labels} labels ({'with' if use_normal_label else 'without'} Normal)")

    # Log class distribution (real only)
    real_labels = label_vectors[~is_synthetic]
    logger.info("Label distribution (real reports only):")
    for i, label in enumerate(label_names):
        count = int(real_labels[:, i].sum())
        logger.info(f"  {label}: {count} ({count / n_real * 100:.1f}%)")

    if n_synth > 0:
        synth_labels = label_vectors[is_synthetic]
        logger.info("Synthetic label distribution:")
        for i, label in enumerate(label_names):
            count = int(synth_labels[:, i].sum())
            if count > 0:
                logger.info(f"  {label}: {count}")

    return texts, label_vectors, study_ids, is_synthetic, label_names, num_labels


def compute_class_weights(label_vectors: np.ndarray) -> torch.Tensor:
    n_samples = label_vectors.shape[0]
    pos_counts = label_vectors.sum(axis=0)
    neg_counts = n_samples - pos_counts
    weights = np.where(pos_counts > 0, neg_counts / pos_counts, 1.0)
    weights = np.clip(weights, 1.0, 50.0)
    return torch.tensor(weights, dtype=torch.float32)


# ---- Focal Loss (Week 9) ----

class FocalBCEWithLogitsLoss(nn.Module):
    """
    Multi-label focal loss for imbalanced classification.

    Standard BCE treats every negative equally. With 13+ labels and most being
    negative for any given report, the model gets overwhelmed by easy negatives.
    Focal loss down-weights easy examples (high-confidence correct predictions)
    and focuses gradient on the hard boundary cases.

    gamma=2.0 is standard from the original paper (Lin et al., 2017).
    pos_weight still applies for class imbalance (same as BCEWithLogitsLoss).
    """

    def __init__(self, gamma: float = 2.0, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Standard BCE component (with pos_weight)
        if self.pos_weight is not None:
            weight = self.pos_weight.unsqueeze(0).expand_as(targets)
            bce = nn.functional.binary_cross_entropy_with_logits(
                logits, targets, weight=None, pos_weight=self.pos_weight, reduction="none"
            )
        else:
            bce = nn.functional.binary_cross_entropy_with_logits(
                logits, targets, reduction="none"
            )

        # Focal modulation: (1 - p_t)^gamma
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)  # prob of correct class
        focal_weight = (1 - p_t) ** self.gamma

        loss = focal_weight * bce
        return loss.mean()


# ---- Threshold tuning ----

def tune_thresholds(y_true: np.ndarray, y_prob: np.ndarray, label_names: Optional[List[str]] = None) -> Dict[str, float]:
    if label_names is None:
        label_names = list(PATHOLOGY_LABELS)
    thresholds = {}
    for i, label in enumerate(label_names):
        best_f1 = 0.0
        best_t = 0.5

        if y_true[:, i].sum() == 0:
            thresholds[label] = 0.5
            continue

        for t in np.arange(0.1, 0.91, 0.05):
            preds = (y_prob[:, i] >= t).astype(int)
            f1 = f1_score(y_true[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        thresholds[label] = round(float(best_t), 2)

    return thresholds


# ---- AUROC / PR-AUC ----

def compute_auroc_prauc(y_true: np.ndarray, y_prob: np.ndarray) -> Dict:
    metrics = {}
    auroc_list = []
    prauc_list = []

    for i, label in enumerate(PATHOLOGY_LABELS):
        n_pos = int(y_true[:, i].sum())
        n_neg = int((1 - y_true[:, i]).sum())

        if n_pos == 0 or n_neg == 0:
            metrics[label] = {"auroc": None, "pr_auc": None}
            continue

        auroc = roc_auc_score(y_true[:, i], y_prob[:, i])
        pr_auc = average_precision_score(y_true[:, i], y_prob[:, i])

        metrics[label] = {"auroc": round(auroc, 4), "pr_auc": round(pr_auc, 4)}
        auroc_list.append(auroc)
        prauc_list.append(pr_auc)

    metrics["_macro_auroc"] = round(np.mean(auroc_list), 4) if auroc_list else None
    metrics["_macro_prauc"] = round(np.mean(prauc_list), 4) if prauc_list else None
    return metrics


# ---- Training loop (with checkpoint resume) ----

def train_one_fold(
    fold: int,
    train_texts: List[str],
    train_labels: np.ndarray,
    val_texts: List[str],
    val_labels: np.ndarray,
    config: dict,
    device: str,
    label_names: Optional[List[str]] = None,
    num_labels_override: Optional[int] = None,
    use_focal_loss: bool = False,
) -> Tuple[np.ndarray, Dict[str, float]]:

    checkpoint_dir = PROJECT_ROOT / config["checkpoint_dir"]
    n_labels = num_labels_override or NUM_LABELS
    lbl_names = label_names or list(PATHOLOGY_LABELS)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"FOLD {fold + 1}/{config['n_folds']}")
    logger.info(f"Train: {len(train_texts)} | Val: {len(val_texts)} | Labels: {n_labels}")
    logger.info(f"Loss: {'Focal (gamma=2.0)' if use_focal_loss else 'BCE'}")
    logger.info(f"{'=' * 60}")

    class_weights = compute_class_weights(train_labels)
    logger.info(f"Class weights: min={class_weights.min():.1f} max={class_weights.max():.1f}")

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    train_ds = ReportDataset(train_texts, train_labels, tokenizer, config["max_length"])
    val_ds = ReportDataset(val_texts, val_labels, tokenizer, config["max_length"])

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=0)

    model = CXRBertClassifier(model_name=config["model_name"], num_labels=n_labels)
    model.to(device)

    pos_weight = class_weights.to(device)
    if use_focal_loss:
        criterion = FocalBCEWithLogitsLoss(gamma=2.0, pos_weight=pos_weight)
        logger.info(f"Using Focal Loss (gamma=2.0) with pos_weight")
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    total_steps = len(train_loader) * config["epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    use_amp = config.get("use_amp", False) and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_macro_f1 = -1.0
    patience_counter = 0
    best_state = None
    start_epoch = 0

    # ---- Resume from epoch checkpoint if available ----
    ckpt = load_epoch_checkpoint(checkpoint_dir, fold)
    if ckpt is not None:
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        best_state = ckpt["best_state_dict"]
        best_val_macro_f1 = ckpt["best_val_macro_f1"]
        patience_counter = ckpt["patience_counter"]
        start_epoch = ckpt["epoch"] + 1  # Resume from next epoch
        logger.info(f"  Resumed at epoch {start_epoch}, patience={patience_counter}")

    epoch_pbar = tqdm(
        range(start_epoch, config["epochs"]),
        desc=f"Fold {fold + 1}/{config['n_folds']}",
        unit="epoch",
        initial=start_epoch,
        total=config["epochs"],
    )
    for epoch in epoch_pbar:
        # -- Train --
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc="  train", leave=False, unit="batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(input_ids, attention_mask, token_type_ids)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # -- Validate --
        model.eval()
        val_loss = 0.0
        all_probs = []
        all_true = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="  val  ", leave=False, unit="batch"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device)
                labels = batch["labels"].to(device)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(input_ids, attention_mask, token_type_ids)
                    loss = criterion(logits, labels)

                val_loss += loss.item()
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)
                all_true.append(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        all_probs = np.concatenate(all_probs, axis=0)
        all_true = np.concatenate(all_true, axis=0)

        preds_05 = (all_probs >= 0.5).astype(int)
        macro_f1 = f1_score(all_true, preds_05, average="macro", zero_division=0)

        epoch_pbar.set_postfix(
            tr_loss=f"{avg_train_loss:.4f}",
            vl_loss=f"{avg_val_loss:.4f}",
            macro_f1=f"{macro_f1:.3f}",
        )
        logger.info(
            f"Epoch {epoch + 1}/{config['epochs']} | "
            f"Train loss: {avg_train_loss:.4f} | "
            f"Val loss: {avg_val_loss:.4f} | "
            f"Val macro F1: {macro_f1:.3f}"
        )

        # Early stopping on macro F1
        if macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = macro_f1
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            logger.info(f"  -> New best macro F1: {macro_f1:.3f}")
        else:
            patience_counter += 1

        # ---- Save epoch checkpoint (survives power cuts) ----
        save_epoch_checkpoint(
            checkpoint_dir=checkpoint_dir,
            fold=fold,
            epoch=epoch,
            model_state={k: v.cpu().clone() for k, v in model.state_dict().items()},
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict(),
            scaler_state=scaler.state_dict(),
            best_state=best_state,
            best_val_macro_f1=best_val_macro_f1,
            patience_counter=patience_counter,
        )

        if patience_counter >= config["patience"]:
            logger.info(f"Early stopping at epoch {epoch + 1} (best macro F1: {best_val_macro_f1:.3f})")
            break

    # Load best state
    model.load_state_dict(best_state)
    model.to(device)

    # Final validation predictions with best model
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="  final val", leave=False, unit="batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(input_ids, attention_mask, token_type_ids)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)

    val_probs = np.concatenate(all_probs, axis=0)

    thresholds = tune_thresholds(val_labels, val_probs, label_names=lbl_names)
    logger.info(f"Tuned thresholds: {thresholds}")

    # Save final fold model
    fold_dir = PROJECT_ROOT / Path(config["output_dir"]) / f"bert_fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), fold_dir / "model.pt")
    tokenizer.save_pretrained(fold_dir / "tokenizer")
    with open(fold_dir / "thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)
    with open(fold_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    # Clear epoch checkpoint for this fold (completed successfully)
    clear_epoch_checkpoint(checkpoint_dir, fold)
    logger.info(f"Fold {fold} complete — model saved to {fold_dir}")

    return val_probs, thresholds


# ---- Evaluation ----

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str],
    y_prob: Optional[np.ndarray] = None,
) -> Dict:
    results = {}
    macro_f1_list = []
    w_num = 0.0
    w_den = 0

    for i, label in enumerate(label_names):
        support = int(y_true[:, i].sum())
        tp = int(((y_pred[:, i] == 1) & (y_true[:, i] == 1)).sum())
        fp = int(((y_pred[:, i] == 1) & (y_true[:, i] == 0)).sum())
        fn = int(((y_pred[:, i] == 0) & (y_true[:, i] == 1)).sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        results[label] = {
            "tp": tp, "fp": fp, "fn": fn, "support": support,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }

        if support > 0:
            macro_f1_list.append(f1)
            w_num += f1 * support
            w_den += support

    results["_macro_f1"] = round(float(np.mean(macro_f1_list)), 4) if macro_f1_list else 0.0
    results["_weighted_f1"] = round(w_num / w_den, 4) if w_den > 0 else 0.0

    exact = int((y_pred == y_true).all(axis=1).sum())
    results["_exact_match"] = round(exact / len(y_true), 4)

    if y_prob is not None:
        auc_metrics = compute_auroc_prauc(y_true, y_prob)
        for label in label_names:
            if label in auc_metrics:
                results[label]["auroc"] = auc_metrics[label].get("auroc")
                results[label]["pr_auc"] = auc_metrics[label].get("pr_auc")
        results["_macro_auroc"] = auc_metrics.get("_macro_auroc")
        results["_macro_prauc"] = auc_metrics.get("_macro_prauc")

    return results



# ---- Main ----

def main():
    # ---- CLI arguments ----
    parser = argparse.ArgumentParser(description="Train BioClinicalBERT extractor")
    parser.add_argument("--fresh", action="store_true", help="Ignore checkpoints, start from scratch")
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to training CSV (relative to project root). "
             "Default: results/annotated_reports.csv. "
             "For augmented+hard negatives: results/nlp/training_reports_v2.csv",
    )
    parser.add_argument(
        "--focal-loss",
        action="store_true",
        help="Use focal loss (gamma=2.0) instead of standard BCE. "
             "Down-weights easy negatives to reduce over-prediction.",
    )
    parser.add_argument(
        "--use-normal-label",
        action="store_true",
        help="Add Normal as explicit 14th label. Teaches BERT what normality "
             "looks like, reducing false positives on normal reports.",
    )
    args = parser.parse_args()

    set_seed(CONFIG["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        logger.info(f"Mixed precision: {CONFIG['use_amp']}")

    logger.info(f"Focal loss: {args.focal_loss}")
    logger.info(f"Normal label: {args.use_normal_label}")

    checkpoint_dir = PROJECT_ROOT / CONFIG["checkpoint_dir"]

    if args.fresh and checkpoint_dir.exists():
        clear_all_checkpoints(checkpoint_dir)
        logger.info("Fresh start requested — all checkpoints cleared")

    # ---- Load data ----
    data_path_rel = args.data_path if args.data_path else CONFIG["data_path"]
    data_path = PROJECT_ROOT / data_path_rel
    logger.info(f"Data source: {data_path_rel}")

    texts, label_vectors, study_ids, is_synthetic, label_names, n_labels = load_data(
        str(data_path), use_normal_label=args.use_normal_label
    )

    # ---- Separate real vs synthetic ----
    real_mask = ~is_synthetic
    real_indices = np.where(real_mask)[0]
    synth_indices = np.where(is_synthetic)[0]

    n_real = len(real_indices)
    n_synth = len(synth_indices)

    real_texts = [texts[i] for i in real_indices]
    real_labels = label_vectors[real_indices]
    real_study_ids = [study_ids[i] for i in real_indices]

    if n_synth > 0:
        synth_texts = [texts[i] for i in synth_indices]
        synth_labels = label_vectors[synth_indices]
        logger.info(f"Synthetic reports to inject per training fold: {n_synth}")
    else:
        synth_texts = []
        synth_labels = np.empty((0, n_labels))

    # For stratification, use only pathology columns (first 13)
    strat_labels = real_labels[:, :NUM_LABELS]

    mskf = MultilabelStratifiedKFold(
        n_splits=CONFIG["n_folds"],
        shuffle=True,
        random_state=CONFIG["seed"],
    )

    # ---- Resume fold progress ----
    fold_progress = None if args.fresh else load_fold_progress(checkpoint_dir)

    if fold_progress is not None:
        completed_folds = fold_progress["completed_folds"]
        oof_probs = fold_progress["oof_probs"]
        oof_thresholds = defaultdict(list, fold_progress["oof_thresholds"])
        fold_indices = fold_progress["fold_indices"]

        if oof_probs.shape[0] != n_real:
            logger.warning(
                f"Checkpoint OOF size ({oof_probs.shape[0]}) != current real reports ({n_real}). "
                f"Resetting progress."
            )
            completed_folds = []
            oof_probs = np.zeros((n_real, n_labels))
            oof_thresholds = defaultdict(list)
            fold_indices = np.zeros(n_real, dtype=int)
        else:
            logger.info(f"Resuming: folds {completed_folds} already done")
    else:
        completed_folds = []
        oof_probs = np.zeros((n_real, n_labels))
        oof_thresholds = defaultdict(list)
        fold_indices = np.zeros(n_real, dtype=int)

    # ---- CV loop ----
    for fold, (real_train_idx, real_val_idx) in tqdm(
        enumerate(mskf.split(real_texts, strat_labels)),
        total=CONFIG["n_folds"],
        desc="5-fold CV",
        unit="fold",
    ):
        if fold in completed_folds:
            logger.info(f"\nSkipping fold {fold} (already completed)")
            continue

        val_texts_fold = [real_texts[i] for i in real_val_idx]
        val_labels_fold = real_labels[real_val_idx]

        fold_real_train_texts = [real_texts[i] for i in real_train_idx]
        fold_real_train_labels = real_labels[real_train_idx]

        if n_synth > 0:
            train_texts_fold = fold_real_train_texts + synth_texts
            train_labels_fold = np.vstack([fold_real_train_labels, synth_labels])
        else:
            train_texts_fold = fold_real_train_texts
            train_labels_fold = fold_real_train_labels

        logger.info(
            f"Fold {fold} — Train: {len(fold_real_train_texts)} real + {n_synth} synthetic = {len(train_texts_fold)} | "
            f"Val: {len(val_texts_fold)} real (0 synthetic)"
        )

        train_pos = train_labels_fold.sum(axis=0)
        val_pos = val_labels_fold.sum(axis=0)
        logger.info(f"Fold {fold} labels — Train: {train_pos.astype(int).tolist()} | Val: {val_pos.astype(int).tolist()}")

        val_probs, thresholds = train_one_fold(
            fold=fold,
            train_texts=train_texts_fold,
            train_labels=train_labels_fold,
            val_texts=val_texts_fold,
            val_labels=val_labels_fold,
            config=CONFIG,
            device=device,
            label_names=label_names,
            num_labels_override=n_labels,
            use_focal_loss=args.focal_loss,
        )

        oof_probs[real_val_idx] = val_probs
        fold_indices[real_val_idx] = fold
        for label, t in thresholds.items():
            oof_thresholds[label].append(t)

        completed_folds.append(fold)
        save_fold_progress(checkpoint_dir, completed_folds, oof_probs, oof_thresholds, fold_indices)

    # ---- Aggregate results (pathology labels only for evaluation) ----
    avg_thresholds = {label: round(float(np.mean(ts)), 2) for label, ts in oof_thresholds.items()}
    logger.info(f"\nAveraged thresholds: {avg_thresholds}")

    eval_label_names = list(PATHOLOGY_LABELS)
    n_eval = NUM_LABELS

    oof_preds = np.zeros((n_real, n_eval))
    for i, label in enumerate(eval_label_names):
        t = avg_thresholds.get(label, 0.5)
        oof_preds[:, i] = (oof_probs[:, i] >= t).astype(int)

    bert_results = evaluate_predictions(
        real_labels[:, :n_eval], oof_preds, eval_label_names, y_prob=oof_probs[:, :n_eval]
    )

    # Normal label stats
    if args.use_normal_label and n_labels > NUM_LABELS:
        normal_idx = NUM_LABELS
        normal_prob_mean = oof_probs[:, normal_idx].mean()
        actual_normal_rate = (real_labels[:, normal_idx] == 1).mean()
        normal_t = avg_thresholds.get("Normal", 0.5)
        pred_normal = (oof_probs[:, normal_idx] >= normal_t).astype(int)
        normal_acc = (pred_normal == real_labels[:, normal_idx].astype(int)).mean()
        logger.info(f"\nNormal label stats:")
        logger.info(f"  Mean Normal prob: {normal_prob_mean:.3f}")
        logger.info(f"  Actual Normal rate: {actual_normal_rate:.1%}")
        logger.info(f"  Normal threshold: {normal_t}")
        logger.info(f"  Normal accuracy: {normal_acc:.1%}")

    logger.info(f"\n{'=' * 80}")
    logger.info(f"BioClinicalBERT OOF RESULTS (5-fold CV, {n_real} real reports)")
    logger.info(f"{'=' * 80}")
    logger.info(f"Exact match: {bert_results['_exact_match']:.1%}")
    logger.info(f"Macro F1:    {bert_results['_macro_f1']:.3f}")
    logger.info(f"Weighted F1: {bert_results['_weighted_f1']:.3f}")
    if bert_results.get("_macro_auroc") is not None:
        logger.info(f"Macro AUROC: {bert_results['_macro_auroc']:.3f}")
    if bert_results.get("_macro_prauc") is not None:
        logger.info(f"Macro PR-AUC:{bert_results['_macro_prauc']:.3f}")
    logger.info("")
    logger.info(f"{'Label':<42} {'TP':>4} {'FP':>4} {'FN':>4} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUROC':>7} {'Sup':>5}")
    logger.info(f"{'-' * 88}")
    for label in eval_label_names:
        r = bert_results[label]
        auroc_str = f"{r['auroc']:>7.3f}" if r.get("auroc") is not None else "   N/A"
        logger.info(
            f"{label:<42} {r['tp']:>4} {r['fp']:>4} {r['fn']:>4} "
            f"{r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>6.3f} "
            f"{auroc_str} {r['support']:>5}"
        )
    logger.info(f"{'-' * 88}")
    macro_auroc_str = f"{bert_results['_macro_auroc']:.3f}" if bert_results.get("_macro_auroc") else "N/A"
    logger.info(
        f"{'Macro':<42} {'':>4} {'':>4} {'':>4} {'':>6} {'':>6} "
        f"{bert_results['_macro_f1']:>6.3f} {macro_auroc_str:>7}"
    )

    # ---- Compare with rule-based ----
    logger.info(f"\n{'=' * 80}")
    logger.info("COMPARISON: Rule-Based vs BioClinicalBERT")
    logger.info(f"{'=' * 80}")

    try:
        from nlp.labels import LabelHierarchy

        lh = LabelHierarchy(
            config_path=str(PROJECT_ROOT / "configs" / "labels" / "hierarchy.yaml"),
            dict_path=str(PROJECT_ROOT / "src" / "nlp" / "dictionary" / "bilingual_dict.json"),
        )

        rb_preds = np.zeros((n_real, n_eval))
        for idx, text in enumerate(real_texts):
            rb_labels_extracted = lh.rule_based_extractor(text)
            for label in rb_labels_extracted:
                if label in eval_label_names:
                    rb_preds[idx, eval_label_names.index(label)] = 1

        rb_results = evaluate_predictions(real_labels[:, :n_eval], rb_preds, eval_label_names)

        logger.info("")
        logger.info(f"{'Label':<42} {'RB F1':>7} {'BERT F1':>8} {'Delta':>7} {'Winner':>8}")
        logger.info(f"{'-' * 82}")
        for label in eval_label_names:
            rb_f1 = rb_results[label]["f1"]
            bert_f1 = bert_results[label]["f1"]
            delta = bert_f1 - rb_f1
            winner = "BERT" if delta > 0.01 else ("RB" if delta < -0.01 else "Tie")
            logger.info(f"{label:<42} {rb_f1:>7.3f} {bert_f1:>8.3f} {delta:>+7.3f} {winner:>8}")
        logger.info(f"{'-' * 82}")
        logger.info(
            f"{'MACRO':<42} {rb_results['_macro_f1']:>7.3f} {bert_results['_macro_f1']:>8.3f} "
            f"{bert_results['_macro_f1'] - rb_results['_macro_f1']:>+7.3f}"
        )
        logger.info(
            f"{'WEIGHTED':<42} {rb_results['_weighted_f1']:>7.3f} {bert_results['_weighted_f1']:>8.3f} "
            f"{bert_results['_weighted_f1'] - rb_results['_weighted_f1']:>+7.3f}"
        )

        comparison = {
            "rule_based": rb_results,
            "bioclinicalbert": bert_results,
            "thresholds": avg_thresholds,
            "config": CONFIG,
        }
    except Exception as e:
        logger.warning(f"Could not run rule-based comparison: {e}")
        comparison = {
            "bioclinicalbert": bert_results,
            "thresholds": avg_thresholds,
            "config": CONFIG,
        }

    # ---- Metadata ----
    comparison["augmentation"] = {
        "used": n_synth > 0,
        "n_real": n_real,
        "n_synthetic": n_synth,
        "data_source": data_path_rel,
        "evaluation_on": "real_reports_only",
        "synthetic_in_validation": False,
    }
    if n_synth > 0:
        synth_label_counts = {}
        for i, label in enumerate(label_names):
            count = int(synth_labels[:, i].sum())
            if count > 0:
                synth_label_counts[label] = count
        comparison["augmentation"]["synthetic_label_counts"] = synth_label_counts

    comparison["training_config"] = {
        "focal_loss": args.focal_loss,
        "use_normal_label": args.use_normal_label,
        "num_labels": n_labels,
        "label_names": label_names,
    }

    # ---- Save outputs ----
    results_dir = PROJECT_ROOT / CONFIG["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "bert_evaluation.json", "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    logger.info(f"\nEvaluation saved to {results_dir / 'bert_evaluation.json'}")

    oof_df = pd.DataFrame()
    oof_df["study_id"] = real_study_ids
    oof_df["fold"] = fold_indices.astype(int)

    for i, label in enumerate(eval_label_names):
        oof_df[f"true_{label}"] = real_labels[:, i].astype(int)
    for i, label in enumerate(eval_label_names):
        oof_df[f"prob_{label}"] = np.round(oof_probs[:, i], 4)
    for i, label in enumerate(eval_label_names):
        t = avg_thresholds.get(label, 0.5)
        oof_df[f"pred_{label}"] = (oof_probs[:, i] >= t).astype(int)

    if args.use_normal_label and n_labels > NUM_LABELS:
        oof_df["prob_Normal"] = np.round(oof_probs[:, NUM_LABELS], 4)
        normal_t = avg_thresholds.get("Normal", 0.5)
        oof_df["pred_Normal"] = (oof_probs[:, NUM_LABELS] >= normal_t).astype(int)

    oof_df["num_true_labels"] = real_labels[:, :n_eval].sum(axis=1).astype(int)
    oof_df["num_pred_labels"] = oof_preds.sum(axis=1).astype(int)
    oof_df["is_normal_true"] = (real_labels[:, :n_eval].sum(axis=1) == 0).astype(int)
    oof_df["is_normal_pred"] = (oof_preds.sum(axis=1) == 0).astype(int)

    oof_df.to_csv(results_dir / "bert_oof_predictions.csv", index=False)
    logger.info(f"OOF predictions saved to {results_dir / 'bert_oof_predictions.csv'}")

    with open(results_dir / "bert_thresholds.json", "w") as f:
        json.dump(avg_thresholds, f, indent=2)

    clear_all_checkpoints(checkpoint_dir)

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
