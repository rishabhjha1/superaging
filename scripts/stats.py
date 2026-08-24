"""Paired significance testing on per-seed held-out metrics (paper Sec. 4.1).

Every model is evaluated on the same seed list, so `results[model][metric][i]`
pairs with `results[ref][metric][i]` by seed index. That pairing is what
licenses a paired test.

  * Primary: paired t-test (two-sided), with a t-based 95% CI and Cohen's dz.
  * Robustness: Wilcoxon signed-rank.
  * Multiplicity: Holm-Bonferroni across the baseline comparisons per metric.

POWER CAVEAT. At n = 5 seeds the exact signed-rank test cannot fall below
p = 2 / 2^5 = 0.0625. It is a directional check, not a significance test at this
n; the paired t-test is primary. This is stated in the paper and should be
stated in any write-up that quotes these numbers.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
from scipy.stats import ttest_rel, wilcoxon

log = logging.getLogger(__name__)


def holm_bonferroni(pvalues: Sequence[float]) -> list[float]:
    """Step-down Holm correction, preserving input order."""
    p = np.asarray(pvalues, dtype=float)
    corrected = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(np.argsort(p)):
        running = max(running, min(1.0, p[idx] * (len(p) - rank)))
        corrected[idx] = running
    return [float(v) for v in corrected]


def paired_comparison(a: np.ndarray, b: np.ndarray,
                      alpha: float = 0.05) -> dict | None:
    """Paired t-test and Wilcoxon between two per-seed metric vectors."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n_pair = min(a.size, b.size)
    a, b = a[:n_pair], b[:n_pair]
    keep = ~(np.isnan(a) | np.isnan(b))     # drop degenerate-fold NaNs
    a, b = a[keep], b[keep]
    if a.size < 2:
        return None

    diff = a - b
    n = a.size
    mean_diff = float(diff.mean())
    sd = float(diff.std(ddof=1))
    se = sd / math.sqrt(n)
    t_crit = float(t_dist.ppf(1 - alpha / 2, df=n - 1))

    t_p = float(ttest_rel(a, b).pvalue)
    if np.allclose(diff, 0):
        w_p = 1.0
    else:
        try:
            w_p = float(wilcoxon(a, b, zero_method="wilcox",
                                 alternative="two-sided", method="exact").pvalue)
        except (TypeError, ValueError):
            w_p = float(wilcoxon(a, b, alternative="two-sided").pvalue)

    return dict(n=n, mean_diff=mean_diff,
                ci_low=mean_diff - t_crit * se, ci_high=mean_diff + t_crit * se,
                t_p=t_p, wilcoxon_p=w_p,
                cohen_dz=float(mean_diff / sd) if sd > 0 else float("nan"))


def paired_significance(results: dict[str, dict[str, list[float]]],
                        model_names: Sequence[str], reference: str = "ViT",
                        metrics: Sequence[str] = ("auc", "auprc"),
                        alpha: float = 0.05, verbose: bool = True) -> pd.DataFrame:
    """Compare `reference` against every other model on each metric."""
    if reference not in results:
        raise KeyError(f"Reference model {reference!r} not in {list(results)}")

    rows: list[dict] = []
    for metric in metrics:
        ref_values = np.asarray(results[reference].get(metric, []), dtype=float)
        if ref_values.size < 2:
            log.warning("Not enough '%s' values for %s; skipping.", metric, reference)
            continue

        if verbose:
            print("\n" + "=" * 88)
            print(f"PAIRED SIGNIFICANCE - {reference} vs baselines on '{metric}' "
                  f"(n={ref_values.size} seeds, paired by seed)")
            print("=" * 88)
            print(f"{reference} per-seed: {np.round(ref_values, 3).tolist()}  "
                  f"(mean {np.nanmean(ref_values):.3f} "
                  f"+/- {np.nanstd(ref_values, ddof=1):.3f})")
            print(f"\n{'vs baseline':<13}{'mean d':>9}{'95% CI':>21}"
                  f"{'t p':>9}{'Wilcox p':>11}{'dz':>8}")
            print("-" * 88)

        metric_rows: list[dict] = []
        for other in [m for m in model_names if m != reference]:
            comparison = paired_comparison(
                ref_values, np.asarray(results[other].get(metric, []), dtype=float), alpha)
            if comparison is None:
                log.warning("Insufficient paired points for %s vs %s", reference, other)
                continue
            metric_rows.append(dict(metric=metric, reference=reference,
                                    baseline=other, **comparison))
            if verbose:
                c = comparison
                print(f"{other:<13}{c['mean_diff']:>+9.3f}  "
                      f"[{c['ci_low']:>+6.3f},{c['ci_high']:>+6.3f}]"
                      f"{c['t_p']:>9.3f}{c['wilcoxon_p']:>11.3f}{c['cohen_dz']:>8.2f}")

        if metric_rows:
            for row, corrected in zip(metric_rows,
                                      holm_bonferroni([r["t_p"] for r in metric_rows])):
                row["t_p_holm"] = corrected
                row["significant"] = bool(corrected < alpha)
            if verbose:
                print("\nHolm-corrected paired-t p-values:")
                for row in metric_rows:
                    print(f"  {reference} vs {row['baseline']:<12} "
                          f"raw {row['t_p']:.3f} -> Holm {row['t_p_holm']:.3f} "
                          f"({'sig' if row['significant'] else 'ns'})")
        rows.extend(metric_rows)

    if verbose and rows:
        print("\nNote: at n=5 seeds the signed-rank test cannot fall below "
              "p=0.0625, so it is a directional check only.")
    return pd.DataFrame(rows)


def summarise(results: dict[str, dict[str, list[float]]],
              model_names: Sequence[str],
              metrics: Sequence[str] = ("auc", "balanced_accuracy",
                                        "auprc", "accuracy")) -> pd.DataFrame:
    """Mean +/- std per model per metric, ranked by AUC."""
    rows = []
    for name in model_names:
        row = {"model": name}
        for metric in metrics:
            values = np.asarray(results[name].get(metric, []), dtype=float)
            row[f"{metric}_mean"] = float(np.nanmean(values)) if values.size else float("nan")
            row[f"{metric}_std"] = float(np.nanstd(values)) if values.size else float("nan")
        row["n_seeds"] = len(results[name].get("auc", []))
        rows.append(row)
    return (pd.DataFrame(rows).sort_values("auc_mean", ascending=False)
            .reset_index(drop=True))


def format_table(summary: pd.DataFrame) -> str:
    """Render the summary as the paper's Table 1."""
    lines = [f"{'Model':<12}{'AUC':>18}{'Bal Acc':>18}{'AUPRC':>18}{'Accuracy':>12}",
             "-" * 78]
    for _, r in summary.iterrows():
        cells = "".join(f"{r[f'{m}_mean']:>11.3f} ±{r[f'{m}_std']:.3f}"
                        for m in ("auc", "balanced_accuracy", "auprc"))
        lines.append(f"{r['model']:<12}{cells}{r['accuracy_mean']:>12.3f}")
    return "\n".join(lines)
