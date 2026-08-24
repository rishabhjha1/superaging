#!/usr/bin/env python
"""Evaluate released checkpoints on the held-out test split (paper Table 1).

This is the entry point for reproducing the reported numbers WITHOUT retraining.
It rebuilds the exact split each checkpoint was trained under (the seed is stored
in the checkpoint), runs the fold ensemble, and prints the metrics.

Examples
--------
Evaluate every released ViT checkpoint and report mean ± std across seeds:

    python eval.py --checkpoints checkpoints/vit_seed*.pt \
                   --labels data/labels.csv --roots data/ADNI data/OASIS

Evaluate a single checkpoint:

    python eval.py --checkpoints results/checkpoints/vit_seed42.pt

Note on split reconstruction: a checkpoint is only meaningful against the split
it was trained under. `eval.py` recreates that split from the stored seed, so
evaluating a released checkpoint against a *different* cohort will report
optimistic numbers — subjects the model trained on will land in the test fold.
Pass --external to score every subject instead, which is the correct mode for a
genuinely held-out external cohort.
"""

from __future__ import annotations

import argparse
import glob
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from superager.config import load_config
from superager.data import collect_records, preprocess
from superager.engine import compute_metrics, get_device, predict, set_seed
from superager.models import build_model
from superager.splits import make_splits

log = logging.getLogger("eval")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="Checkpoint files or globs, e.g. 'checkpoints/vit_seed*.pt'")
    p.add_argument("--config", help="YAML config; CLI flags override it")
    p.add_argument("--labels", dest="labels_csv")
    p.add_argument("--roots", dest="image_roots", nargs="+")
    p.add_argument("--outdir", default="results/eval")
    p.add_argument("--external", action="store_true",
                   help="Score every subject rather than reconstructing the "
                        "training split. Use for a genuinely external cohort.")
    p.add_argument("--no-tta", dest="tta", action="store_false",
                   help="Disable four-view test-time augmentation")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def expand(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matched = sorted(glob.glob(pattern))
        if not matched:
            log.warning("No checkpoint matched %r", pattern)
        paths.extend(Path(m) for m in matched)
    if not paths:
        raise FileNotFoundError(f"No checkpoints found for {patterns}")
    return paths


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")

    overrides = {k: v for k, v in vars(args).items() if k in ("labels_csv", "image_roots")}
    cfg = load_config(args.config, **overrides)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = get_device()

    df = collect_records(cfg.labels_csv, cfg.image_roots)
    X = preprocess(df, cfg)
    y = df.label.to_numpy()
    groups = df.subject.to_numpy()
    sites = df.site.to_numpy()

    rows = []
    for path in expand(args.checkpoints):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model_name = ckpt.get("model_name", "ViT")
        seed = int(ckpt.get("seed", cfg.seed))
        threshold = float(ckpt.get("threshold", 0.5))

        model = build_model(model_name, cfg).to(device)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        model.eval()

        if args.external:
            eval_idx = np.arange(len(y))
            split_note = "all subjects (external cohort mode)"
        else:
            set_seed(seed)
            eval_idx, _, _ = make_splits(y, groups, cfg.n_splits, sites, seed)
            split_note = f"held-out test fold reconstructed from seed {seed}"

        prob = predict(model, X, eval_idx, device, tta=args.tta,
                       batch_size=cfg.batch_size)
        metrics = compute_metrics(y[eval_idx], prob, threshold)

        log.info("%s | %s | n=%d | AUC %.3f | AUPRC %.3f | BalAcc %.3f",
                 path.name, split_note, len(eval_idx),
                 metrics["auc"], metrics["auprc"], metrics["balanced_accuracy"])
        rows.append(dict(checkpoint=path.name, model=model_name, seed=seed,
                         n_eval=len(eval_idx), threshold=threshold, **metrics))

        pd.DataFrame(dict(subject=df.subject.to_numpy()[eval_idx],
                          label=y[eval_idx], prob_superager=prob)).to_csv(
            outdir / f"predictions_{path.stem}.csv", index=False)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    table = pd.DataFrame(rows)
    table.to_csv(outdir / "eval_metrics.csv", index=False)

    print("\n" + "=" * 72)
    print(f"EVALUATION - {len(table)} checkpoint(s), mean ± std across seeds")
    print("=" * 72)
    for metric in ("auc", "balanced_accuracy", "auprc", "accuracy"):
        print(f"  {metric:<20} {table[metric].mean():.3f} ± {table[metric].std():.3f}")
    print(f"\nWrote {outdir}/eval_metrics.csv and per-checkpoint predictions.")

    if not args.external and len(table) > 1:
        print("\nEach row is a different seed, so each is a different test fold. "
              "The spread across rows is seed-to-seed variance, not a confidence "
              "interval on a single estimate.")


if __name__ == "__main__":
    main()
