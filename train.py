#!/usr/bin/env python


from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from superager.config import load_config
from superager.data import audit_site_label, collect_records, preprocess
from superager.engine import METRIC_KEYS, age_only_baseline, get_device, run_model, set_seed
from superager.models import MODEL_ORDER, build_model, count_parameters
from superager.plotting import seed_summary
from superager.splits import describe_split, make_splits
from superager.stats import format_table, paired_significance, summarise

log = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="YAML config; CLI flags override it")
    p.add_argument("--labels", dest="labels_csv", help="CSV with subject,label[,age]")
    p.add_argument("--roots", dest="image_roots", nargs="+",
                   help="One or more directories searched recursively for NIfTI files")
    p.add_argument("--outdir", help="Where results and checkpoints are written")
    p.add_argument("--seeds", type=int, nargs="+", help="Random seeds (default 42 1 2 3 4)")
    p.add_argument("--models", nargs="+", default=MODEL_ORDER, choices=MODEL_ORDER,
                   help="Subset of models to run")
    p.add_argument("--reference", default="ViT",
                   help="Model compared against the others in the significance tests")
    p.add_argument("--quick", action="store_true", help="Few epochs and seeds; a smoke test")
    p.add_argument("--no-cache", dest="cache", action="store_false", default=None,
                   help="Recompute preprocessing instead of reading the cache")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")

    overrides = {k: v for k, v in vars(args).items()
                 if k in ("labels_csv", "image_roots", "outdir", "seeds", "cache")}
    if args.quick:
        overrides["quick"] = True
    cfg = load_config(args.config, **overrides)

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    log.info("Device: %s | models: %s | seeds: %s", device, args.models, cfg.seeds)

    # ---- data ----
    df = collect_records(cfg.labels_csv, cfg.image_roots)
    audit_site_label(df).to_csv(outdir / "site_audit.csv")
    X = preprocess(df, cfg)
    y = df.label.to_numpy()
    groups = df.subject.to_numpy()
    sites = df.site.to_numpy()

    for name in args.models:
        model = build_model(name, cfg)
        if getattr(model, "is_torch", False):
            total, trainable = count_parameters(model)
            log.info("  %-11s %6.2fM params, %6.3fM trainable", name, total, trainable)
        else:
            log.info("  %-11s classical (StandardScaler + PCA + LogReg)", name)
        del model

    # ---- age-only sanity baseline (Sec. 3.1) ----
    ages = df["age"].to_numpy() if "age" in df.columns else None
    age_result = age_only_baseline(ages, y, groups, sites, cfg) if ages is not None else None

    # ---- seeds x models ----
    results = {name: {k: [] for k in METRIC_KEYS} for name in args.models}
    for seed in cfg.seeds:
        print(f"\n{'#' * 64}\n# SEED {seed}\n{'#' * 64}")
        set_seed(seed)
        test_idx, _, cv_folds = make_splits(y, groups, cfg.n_splits, sites, seed)
        log.info("\n%s", describe_split(y, test_idx, cv_folds))

        for name in args.models:
            log.info("  --- %s ---", name)
            out = run_model(name, X, y, test_idx, cv_folds, cfg, device, seed,
                            save_checkpoint=(name == "ViT"))
            log.info("  => AUC %.3f | AUPRC %.3f | BalAcc %.3f | thr %.2f",
                     out["auc"], out["auprc"], out["balanced_accuracy"], out["threshold"])
            for k in METRIC_KEYS:
                results[name][k].append(out[k])

    # ---- Table 1 ----
    summary = summarise(results, args.models)
    print("\n" + "=" * 78)
    print(f"RESULTS - held-out test, mean ± std over {len(cfg.seeds)} seeds")
    print("=" * 78)
    print(format_table(summary))

    summary.to_csv(outdir / "results_summary.csv", index=False)
    pd.DataFrame([
        dict(seed=seed, model=name, **{k: results[name][k][i] for k in METRIC_KEYS})
        for name in args.models for i, seed in enumerate(cfg.seeds)
    ]).to_csv(outdir / "results_per_seed.csv", index=False)

    # ---- significance ----
    if args.reference in args.models and len(args.models) > 1:
        significance = paired_significance(results, args.models, reference=args.reference)
        significance.to_csv(outdir / "significance.csv", index=False)

    seed_summary(results, args.models, outdir / "seed_summary.png")

    with open(outdir / "run_manifest.json", "w") as f:
        json.dump(dict(config=asdict(cfg), models=args.models,
                       n_scans=int(len(df)), n_superagers=int((y == 1).sum()),
                       cohorts=df.cohort.value_counts().to_dict(),
                       age_only_baseline=age_result), f, indent=2, default=str)

    print(f"\nWrote results to {outdir}/ "
          f"(results_summary.csv, results_per_seed.csv, significance.csv, "
          f"seed_summary.png, run_manifest.json, checkpoints/)")


if __name__ == "__main__":
    main()
