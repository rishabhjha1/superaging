"""Training, augmentation, inference and metrics (paper Sec. 3.4).

Optimisation: AdamW, cosine annealing with a 3-epoch warmup (min-LR ratio 0.05),
class-balanced cross-entropy with label smoothing (eps = 0.05), early stopping
(patience 8) on 0.5 * AUC + 0.5 * balanced accuracy.

Learning rates differ by regime for a stated reason. From-scratch models use
8e-4 because randomly initialised weights need larger updates to converge;
pretrained-backbone heads use 1e-4 since only a lightweight head is optimised on
top of stable features.
"""

from __future__ import annotations

import logging
import math
import random
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight

from .config import Config
from .models import MODELS, ViTNet, build_model, epochs_for

log = logging.getLogger(__name__)

METRIC_KEYS = ("auc", "auprc", "balanced_accuracy", "f1", "accuracy")


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================ AUGMENTATION
def affine_zoom_shift(x: torch.Tensor, scale: float, tx: float = 0.0,
                      ty: float = 0.0) -> torch.Tensor:
    theta = torch.tensor([[1.0 / scale, 0.0, tx], [0.0, 1.0 / scale, ty]],
                         dtype=x.dtype, device=x.device)
    theta = theta.unsqueeze(0).repeat(x.shape[0], 1, 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="border")


def augment(x: torch.Tensor, strong: bool) -> torch.Tensor:
    """Training augmentation.

    `strong` adds affine zoom/shift (p=0.4) and gamma jitter (p=0.25). It is
    disabled for the ViT: pretrained transformer features are sensitive to
    aggressive geometric distortion, so only flipping and mild intensity
    perturbation are kept.
    """
    if random.random() < 0.5:
        x = torch.flip(x, [-1])
    if strong and random.random() < 0.4:
        x = affine_zoom_shift(x, 0.95 + 0.10 * random.random(),
                              (random.random() * 2 - 1) * 0.03,
                              (random.random() * 2 - 1) * 0.03)
    if strong and random.random() < 0.25:
        x = x.clamp(0, 1) ** (0.9 + 0.2 * random.random())
    if random.random() < 0.25:
        x = x * (0.95 + 0.10 * torch.rand(1, device=x.device))
    return (x + 0.01 * torch.randn_like(x)).clamp(0, 1)


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, idx: np.ndarray,
            device: torch.device, tta: bool = False,
            batch_size: int = 16) -> np.ndarray:
    """P(SuperAger). With `tta`, averages original, h-flip, x1.05 and x0.95."""
    model.eval()
    out: list[np.ndarray] = []
    amp = device.type == "cuda"
    for b in range(0, len(idx), batch_size):
        x = torch.from_numpy(X[idx[b:b + batch_size]]).float().to(device)
        with torch.autocast(device.type, enabled=amp):
            views = ([x, torch.flip(x, [-1]), affine_zoom_shift(x, 1.05),
                      affine_zoom_shift(x, 0.95)] if tta else [x])
            p = torch.stack([torch.softmax(model(v), 1)[:, 1] for v in views]).mean(0)
        out.append(p.float().cpu().numpy())
    return np.concatenate(out)


# ============================================================ METRICS
def compute_metrics(y_true: np.ndarray, prob: np.ndarray,
                    threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true)
    pred = (np.asarray(prob) >= threshold).astype(int)
    two_class = len(set(y_true.tolist())) > 1
    return dict(
        auc=float(roc_auc_score(y_true, prob)) if two_class else float("nan"),
        auprc=float(average_precision_score(y_true, prob)) if two_class else float("nan"),
        balanced_accuracy=float(balanced_accuracy_score(y_true, pred)),
        f1=float(f1_score(y_true, pred, pos_label=1, zero_division=0)),
        accuracy=float(accuracy_score(y_true, pred)),
    )


def pick_threshold(y_true: np.ndarray, prob: np.ndarray,
                   margin: float = 0.02) -> float:
    """Balanced-accuracy threshold chosen on VALIDATION predictions, never test.

    A tuned threshold is adopted only if it beats the 0.5 default by more than
    two points, so a marginal, noise-driven cut does not get locked in.
    """
    if len(set(np.asarray(y_true).tolist())) < 2:
        return 0.5
    base = balanced_accuracy_score(y_true, (prob >= 0.5).astype(int))
    best_score, best_t = base, 0.5
    for t in np.linspace(0.2, 0.8, 31):
        score = balanced_accuracy_score(y_true, (prob >= t).astype(int))
        if score > best_score + 1e-9:
            best_score, best_t = score, float(t)
    return best_t if (best_score - base) > margin else 0.5


# ============================================================ OPTIMISATION
def build_optimizer(model: nn.Module, cfg: Config) -> tuple[torch.optim.Optimizer, float]:
    params = [p for p in model.parameters() if p.requires_grad]
    lr = cfg.lr_head if model.pretrained_backbone else cfg.lr_scratch
    wd = cfg.vit_weight_decay if isinstance(model, ViTNet) else cfg.weight_decay
    return torch.optim.AdamW(params, lr=lr, weight_decay=wd), lr


def lr_factor(epoch: int, warmup: int, max_epochs: int, min_ratio: float) -> float:
    """Linear warmup then cosine decay to min_ratio."""
    if epoch < warmup:
        return (epoch + 1) / max(1, warmup)
    progress = (epoch - warmup) / max(1, max_epochs - warmup)
    return min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * progress))


# ============================================================ FOLD TRAINING
def train_fold(model_name: str, X: np.ndarray, y: np.ndarray,
               train_idx: np.ndarray, val_idx: np.ndarray,
               cfg: Config, device: torch.device) -> dict:
    """Train one inner fold and return its best checkpoint.

    `best` excludes epochs whose validation predictions are single-class: at
    n = 200 a fold can produce a good AUC alongside a degenerate prediction, and
    selecting that checkpoint would poison the fold ensemble. `best_any` is kept
    as a fallback so a fold never returns an empty state dict.
    """
    model = build_model(model_name, cfg).to(device)
    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y[train_idx])
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32).to(device),
        label_smoothing=cfg.label_smoothing)
    optimizer, base_lr = build_optimizer(model, cfg)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    max_epochs = epochs_for(model_name, cfg)
    strong_aug = not isinstance(model, ViTNet)

    blank = dict(score=-1.0, state=None, auc=float("nan"),
                 balanced_accuracy=float("nan"), val_prob=None, epoch=-1)
    best, best_any = dict(blank), dict(blank)
    stale = 0

    for epoch in range(max_epochs):
        factor = lr_factor(epoch, cfg.warmup, max_epochs, cfg.min_lr_ratio)
        for group in optimizer.param_groups:
            group["lr"] = base_lr * factor

        model.train()
        order = np.random.permutation(train_idx)
        for b in range(0, len(order), cfg.batch_size):
            batch = order[b:b + cfg.batch_size]
            xb = augment(torch.from_numpy(X[batch]).float().to(device), strong_aug)
            yb = torch.from_numpy(y[batch]).long().to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = criterion(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        prob = predict(model, X, val_idx, device, tta=False, batch_size=cfg.batch_size)
        metrics = compute_metrics(y[val_idx], prob)
        auc = 0.0 if np.isnan(metrics["auc"]) else metrics["auc"]
        score = 0.5 * auc + 0.5 * metrics["balanced_accuracy"]
        degenerate = len(set((prob >= 0.5).astype(int).tolist())) < 2
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if score > best_any["score"]:
            best_any.update(score=score, state=state, auc=auc, epoch=epoch,
                            balanced_accuracy=metrics["balanced_accuracy"],
                            val_prob=prob.copy())
        if not degenerate and score > best["score"]:
            best.update(score=score, state=state, auc=auc, epoch=epoch,
                        balanced_accuracy=metrics["balanced_accuracy"],
                        val_prob=prob.copy())
            stale = 0
        else:
            stale += 1
        if stale >= cfg.patience:
            break

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best if best["state"] is not None else best_any


# ============================================================ MODEL RUN
def run_model(model_name: str, X: np.ndarray, y: np.ndarray,
              test_idx: np.ndarray,
              cv_folds: Sequence[tuple[np.ndarray, np.ndarray]],
              cfg: Config, device: torch.device, seed: int,
              save_checkpoint: bool = False) -> dict:
    """Fold-ensembled evaluation of one model for one seed.

    Test predictions are the mean over the inner-fold checkpoints; the decision
    threshold comes from the pooled inner-validation predictions.
    """
    is_torch = MODELS[model_name].is_torch
    use_tta = cfg.vit_tta if model_name == "ViT" else True

    test_probs, val_true, val_probs, fold_aucs = [], [], [], []
    best_state, best_score = None, -np.inf

    for i, (train_idx, val_idx) in enumerate(cv_folds):
        if is_torch:
            fold = train_fold(model_name, X, y, train_idx, val_idx, cfg, device)
            model = build_model(model_name, cfg).to(device)
            model.load_state_dict(fold["state"])
            test_probs.append(predict(model, X, test_idx, device, tta=use_tta,
                                      batch_size=cfg.batch_size))
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if fold["score"] > best_score:
                best_score, best_state = fold["score"], fold["state"]
        else:
            estimator = build_model(model_name, cfg).fit(X[train_idx], y[train_idx])
            fold = dict(val_prob=estimator.predict_proba(X[val_idx]),
                        auc=float("nan"), score=0.0)
            two_class = len(set(y[val_idx].tolist())) > 1
            fold["auc"] = (float(roc_auc_score(y[val_idx], fold["val_prob"]))
                           if two_class else float("nan"))
            test_probs.append(estimator.predict_proba(X[test_idx]))

        fold_aucs.append(fold["auc"])
        val_true.append(y[val_idx])
        val_probs.append(fold["val_prob"])
        log.info("    fold %d/%d: val AUC %.3f", i + 1, len(cv_folds), fold["auc"])

    ensemble = np.mean(test_probs, axis=0)
    threshold = pick_threshold(np.concatenate(val_true), np.concatenate(val_probs),
                               cfg.threshold_margin)
    metrics = compute_metrics(y[test_idx], ensemble, threshold)

    if save_checkpoint and best_state is not None:
        cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = cfg.checkpoint_dir / f"{model_name.lower()}_seed{seed}.pt"
        torch.save({"model_name": model_name, "state_dict": best_state,
                    "threshold": float(threshold), "seed": int(seed),
                    "val_score": float(best_score),
                    "config": dict(vars(cfg).items())}, path)
        log.info("    saved checkpoint -> %s", path)

    return dict(**metrics, threshold=float(threshold),
                cv_auc=float(np.nanmean(fold_aucs)), test_prob=ensemble)


def age_only_baseline(ages: np.ndarray, y: np.ndarray, groups: np.ndarray,
                      sites: np.ndarray, cfg: Config) -> dict | None:
    """Sanity check from Sec. 3.1: age alone should classify at chance.

    Because age strongly shapes cortical structure, this confirms the imaging
    result is not age-driven. It reuses the SAME splits as the imaging models --
    a fresh split would not be a valid comparison.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from .splits import make_splits

    if ages is None or np.isnan(ages).any():
        log.warning("No complete age column; skipping the age-only baseline.")
        return None

    ages = np.asarray(ages, dtype=float).reshape(-1, 1)
    aucs = []
    for seed in cfg.seeds:
        set_seed(seed)
        test_idx, _, cv_folds = make_splits(y, groups, cfg.n_splits, sites, seed)
        probs = []
        for train_idx, _ in cv_folds:
            scaler = StandardScaler().fit(ages[train_idx])
            clf = LogisticRegression(class_weight="balanced", max_iter=2000)
            clf.fit(scaler.transform(ages[train_idx]), y[train_idx])
            probs.append(clf.predict_proba(scaler.transform(ages[test_idx]))[:, 1])
        aucs.append(float(roc_auc_score(y[test_idx], np.mean(probs, axis=0))))

    result = dict(auc_mean=float(np.mean(aucs)), auc_std=float(np.std(aucs)),
                  per_seed=aucs)
    log.info("Age-only baseline: test AUC %.3f +/- %.3f over %d seeds",
             result["auc_mean"], result["auc_std"], len(aucs))
    if result["auc_mean"] > 0.60:
        log.warning("Age alone exceeds 0.60 AUC -- the imaging result may be "
                    "partly age-driven. Re-check the SuperAger age criteria "
                    "before interpreting the regional findings.")
    return result
