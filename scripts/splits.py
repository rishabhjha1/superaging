"""Subject-grouped, site-aware stratified splitting (paper Sec. 3.4).

Two invariants, both asserted rather than assumed:

  * **Grouping.** Splits are grouped by subject, so no subject ever appears in
    both a training and an evaluation partition.
  * **Site-aware stratification.** The key is the pair (label, site) rather than
    the label alone. This spreads each acquisition site across folds instead of
    letting a fold become site-pure, which would turn a scanner signature into
    a free label.

`make_splits` returns a nested design: one outer held-out test fold and an inner
K-fold CV over the remaining development pool. Test predictions are the mean
over the K inner-fold checkpoints (a fold ensemble); the decision threshold is
chosen on the pooled inner validation predictions and never on test.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

log = logging.getLogger(__name__)

SplitBundle = tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]


def stratification_key(y: np.ndarray, sites: np.ndarray | None,
                       min_per_stratum: int = 0) -> np.ndarray:
    """Combine label and site into a single categorical stratification key.

    ADNI encodes acquisition site in the subject ID, so a pooled cohort can have
    dozens of sites holding a handful of subjects each. A naive (label, site)
    key then produces strata smaller than the fold count and
    `StratifiedGroupKFold` refuses to split at all.

    `min_per_stratum` guards against that: sites contributing fewer than that
    many subjects are pooled into a shared "small sites" bucket. The result
    still spreads the large sites across folds while staying feasible. Pass 0
    to disable merging.
    """
    y = np.asarray(y).astype(np.int64)
    if sites is None:
        return y.copy()

    sites = np.asarray(sites)
    if min_per_stratum > 0:
        merged = sites.astype(object).copy()
        labels, counts = np.unique(sites, return_counts=True)
        rare = {s for s, c in zip(labels, counts) if c < min_per_stratum}
        if rare:
            merged[np.isin(sites, list(rare))] = "__SMALL_SITES__"
        sites = merged

    site_ids = pd.factorize(sites)[0]
    # Multiplier 1000, not 100: ADNI can exceed 100 distinct sites, and a
    # smaller multiplier would collide (label 0, site 100) with (label 1, site 0).
    return y * 1000 + site_ids.astype(np.int64)


def _feasible(strat: np.ndarray, groups: np.ndarray, n_splits: int) -> bool:
    """Can StratifiedGroupKFold actually produce n_splits folds from this key?"""
    _, counts = np.unique(strat, return_counts=True)
    return bool(counts.min() >= n_splits and len(np.unique(groups)) >= n_splits)


def resolve_stratification(y: np.ndarray, groups: np.ndarray,
                           sites: np.ndarray | None,
                           n_splits: int) -> tuple[np.ndarray, str]:
    """Pick the finest stratification the cohort can actually support.

    Tried in order, stopping at the first feasible option:
      1. (label, site) with every site kept separate
      2. (label, site) with sites below n_splits subjects pooled
      3. label only

    Returns the key and a short description, so the run record always says which
    level was used rather than silently degrading.
    """
    if sites is not None:
        fine = stratification_key(y, sites, min_per_stratum=0)
        if _feasible(fine, groups, n_splits):
            return fine, "label x site"

        merged = stratification_key(y, sites, min_per_stratum=n_splits)
        if _feasible(merged, groups, n_splits):
            log.warning("Some sites are too small to stratify on individually; "
                        "pooling rare sites into one bucket.")
            return merged, "label x site (rare sites pooled)"

    if sites is not None:
        log.warning("Site-aware stratification is infeasible at n_splits=%d; "
                    "falling back to label-only. Folds may end up site-imbalanced "
                    "-- check the site audit before reading cross-site "
                    "generalisation into the results.", n_splits)
    return np.asarray(y).astype(np.int64).copy(), "label only"


def assert_disjoint(a: Sequence[int], b: Sequence[int], groups: np.ndarray) -> None:
    """Raise if any subject appears on both sides of a split."""
    overlap = set(groups[np.asarray(a)]) & set(groups[np.asarray(b)])
    if overlap:
        raise RuntimeError(
            f"Subject leakage across partitions: {sorted(overlap)[:5]} "
            f"({len(overlap)} subjects appear on both sides).")


def make_splits(y: np.ndarray, groups: np.ndarray, n_splits: int = 5,
                sites: np.ndarray | None = None, seed: int = 42) -> SplitBundle:
    """Return (test_idx, dev_idx, cv_folds) with cv_folds over the dev pool."""
    y, groups = np.asarray(y), np.asarray(groups)

    strat, level = resolve_stratification(y, groups, sites, n_splits)
    log.debug("Outer split stratified by: %s", level)
    outer = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    dev_idx, test_idx = next(iter(outer.split(np.zeros(len(y)), strat, groups)))

    # The development pool is smaller than the cohort, so a key that was
    # feasible overall can become infeasible here; re-resolve on the subset.
    dev_sites = None if sites is None else np.asarray(sites)[dev_idx]
    dev_strat, dev_level = resolve_stratification(y[dev_idx], groups[dev_idx],
                                                  dev_sites, n_splits)
    log.debug("Inner split stratified by: %s", dev_level)
    inner = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + 1)
    cv_folds = [(dev_idx[tr], dev_idx[va]) for tr, va in
                inner.split(np.zeros(len(dev_idx)), dev_strat, groups[dev_idx])]

    assert_disjoint(test_idx, dev_idx, groups)
    for tr, va in cv_folds:
        assert_disjoint(tr, va, groups)
    return test_idx, dev_idx, cv_folds


def describe_split(y: np.ndarray, test_idx: np.ndarray,
                   cv_folds: Sequence[tuple[np.ndarray, np.ndarray]]) -> str:
    y_test = np.asarray(y)[test_idx]
    lines = [f"held-out test: {len(test_idx)} scans "
             f"({int((y_test == 0).sum())} typical / {int((y_test == 1).sum())} SuperAger)"]
    for i, (tr, va) in enumerate(cv_folds):
        lines.append(f"  fold {i + 1}: train {len(tr)} | val {len(va)} "
                     f"({int((np.asarray(y)[va] == 1).sum())} SuperAger)")
    return "\n".join(lines)
