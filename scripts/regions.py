"""Grid-based regional analysis (paper Sec. 3.5, Table 2).

The axial slice is partitioned into a 6x6 grid and each cell assigned to one of
five coarse regions. Per region we report:

    I_R      mean attention weight over the region's cells
    S_R      I_R * P(SuperAger), i.e. importance discounted by model confidence
    sigma_R  standard deviation of I_R across subjects and slice offsets

Consistency is checked by repeating the analysis at +/- 5% depth increments
around the mid-slice (five slices at 40/45/50/55/60% depth by default), so a
region's score is not an artefact of one arbitrary slice.

TWO CAVEATS, stated here because they change how the numbers should be read.

1. **The grid is a coarse proxy, not an atlas.** A 6x6 partition of a mid-axial
   slice approximates lobar location; it is not registered to a parcellation and
   cell-to-region assignment near boundaries is approximate. For atlas-grade
   claims, register to MNI space and pool over an atlas such as AAL or
   Desikan-Killiany. `region_masks_from_atlas` is the hook for that.

2. **I_R is reported relative to peak.** With `scaling="peak"`, the five region
   means are divided by the largest within each slice, so the top region sits at
   ~1.0 and every other value is a fraction of it. This is a disclosed display
   convention that makes values comparable across slices; it does not change the
   model, the ranking, or any statistical comparison. Use `scaling="raw"` for
   unnormalised mean rollout weights.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

REGIONS: tuple[str, ...] = ("Frontal", "Parietal", "Temporal", "Central", "Subcortical")

_CODE_TO_REGION = {"F": "Frontal", "P": "Parietal", "T": "Temporal",
                   "C": "Central", "S": "Subcortical"}

DISPLAY_NAMES = {"Frontal": "Frontal Cortex", "Parietal": "Parietal Cortex",
                 "Temporal": "Temporal Cortex", "Central": "Central Regions",
                 "Subcortical": "Subcortical"}

# Default 6x6 layout for a canonical (RAS) axial mid-slice.
# Rows run posterior -> anterior along array axis 0; columns run left -> right
# along axis 1 after the canonical reorientation used in preprocessing.
#
# VERIFY THIS AGAINST YOUR OWN DATA before quoting anatomy: run
# `python explain.py --check-orientation`, which overlays the grid on a real
# slice. A flipped or transposed volume will swap frontal and parietal labels
# while leaving every number in the table unchanged and plausible-looking.
DEFAULT_GRID = [
    list("FFFFFF"),
    list("FFFFFF"),
    list("TCCCCT"),
    list("TSSSST"),
    list("PPPPPP"),
    list("PPPPPP"),
]


@dataclass
class RegionalScores:
    """Per-region scores for one subject at one slice offset."""

    importance: dict[str, float]          # I_R
    superager_score: dict[str, float]     # S_R = I_R * P(SuperAger)
    probability: float
    slice_fraction: float


def validate_grid(grid: Sequence[Sequence[str]]) -> np.ndarray:
    """Check the grid is square and only uses known region codes."""
    arr = np.array([list(row) for row in grid], dtype="<U1")
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Grid must be square, got shape {arr.shape}")
    unknown = sorted(set(arr.ravel()) - set(_CODE_TO_REGION))
    if unknown:
        raise ValueError(f"Unknown region codes {unknown}; "
                         f"valid codes are {sorted(_CODE_TO_REGION)}")
    return arr


def grid_region_names(grid: Sequence[Sequence[str]] = DEFAULT_GRID) -> np.ndarray:
    """Return the grid with codes expanded to region names."""
    return np.vectorize(_CODE_TO_REGION.get)(validate_grid(grid))


def pool_to_grid(saliency, grid_size: int = 6) -> np.ndarray:
    """Average-pool an arbitrary-resolution saliency map onto a grid_size grid.

    Implemented in numpy (adaptive-average-pool semantics: cell boundaries are
    floor/ceil of the exact split) so this module stays importable without
    torch — regional analysis is pure array arithmetic.
    """
    arr = np.asarray(saliency, dtype=np.float32)
    if hasattr(saliency, "detach"):                      # a torch tensor
        arr = saliency.detach().cpu().numpy().astype(np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Saliency must be 2D, got shape {arr.shape}")

    h, w = arr.shape
    out = np.empty((grid_size, grid_size), dtype=np.float32)
    for i in range(grid_size):
        r0, r1 = int(np.floor(i * h / grid_size)), int(np.ceil((i + 1) * h / grid_size))
        for j in range(grid_size):
            c0, c1 = int(np.floor(j * w / grid_size)), int(np.ceil((j + 1) * w / grid_size))
            out[i, j] = arr[r0:r1, c0:c1].mean()
    return out


def region_scores(saliency, grid: Sequence[Sequence[str]] = DEFAULT_GRID,
                  scaling: str = "peak") -> dict[str, float]:
    """Pool a saliency map into per-region means.

    Args:
        saliency: any 2D map (rollout, Grad-CAM, LIME saliency).
        grid: cell -> region-code layout.
        scaling: "peak" divides by the largest region mean (top region ~1.0);
                 "raw" returns unnormalised means; "sum" normalises to sum 1.
    """
    names = grid_region_names(grid)
    pooled = pool_to_grid(saliency, names.shape[0])
    values = {r: float(pooled[names == r].mean()) if (names == r).any() else float("nan")
              for r in REGIONS}

    if scaling == "raw":
        return values
    if scaling == "sum":
        total = sum(v for v in values.values() if np.isfinite(v)) + 1e-8
        return {r: v / total for r, v in values.items()}
    if scaling == "peak":
        peak = max(v for v in values.values() if np.isfinite(v)) + 1e-8
        return {r: v / peak for r, v in values.items()}
    raise ValueError(f"Unknown scaling {scaling!r}; use peak | raw | sum")


def slice_fractions(centre: float = 0.5, increment: float = 0.05,
                    steps: int = 2) -> list[float]:
    """Depth fractions for the consistency sweep, e.g. 0.40 ... 0.60."""
    return [round(centre + k * increment, 4) for k in range(-steps, steps + 1)]


def aggregate_regional(samples: Sequence[RegionalScores]) -> pd.DataFrame:
    """Aggregate per-subject, per-slice scores into the paper's Table 2.

    Returns a dataframe sorted by descending I_R, with columns
    [region, I_R, sigma_R, S_R, S_R_sd, n, rank].
    """
    if not samples:
        return pd.DataFrame()

    rows = []
    for region in REGIONS:
        importance = np.array([s.importance[region] for s in samples], dtype=float)
        sa_score = np.array([s.superager_score[region] for s in samples], dtype=float)
        rows.append(dict(
            region=region,
            I_R=float(np.nanmean(importance)),
            sigma_R=float(np.nanstd(importance, ddof=1)) if importance.size > 1 else 0.0,
            S_R=float(np.nanmean(sa_score)),
            S_R_sd=float(np.nanstd(sa_score, ddof=1)) if sa_score.size > 1 else 0.0,
            n=int(importance.size),
        ))

    df = pd.DataFrame(rows).sort_values("I_R", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def regional_agreement(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    """Pearson and Spearman agreement between two regional score dicts.

    With only five regions these coefficients carry roughly two effective
    degrees of freedom: they indicate whether two methods rank regions the same
    way, and should not be read as precise effect sizes. The paper reports
    r = 0.84 between Grad-CAM and attention on exactly this basis.
    """
    from scipy.stats import pearsonr, spearmanr

    common = [r for r in REGIONS if r in a and r in b]
    va = np.array([a[r] for r in common], dtype=float)
    vb = np.array([b[r] for r in common], dtype=float)
    if len(common) < 3 or np.allclose(va, va[0]) or np.allclose(vb, vb[0]):
        return dict(pearson_r=float("nan"), pearson_p=float("nan"),
                    spearman_rho=float("nan"), n_regions=len(common))

    r, rp = pearsonr(va, vb)
    rho, rhop = spearmanr(va, vb)
    return dict(pearson_r=float(r), pearson_p=float(rp),
                spearman_rho=float(rho), spearman_p=float(rhop),
                n_regions=len(common))


def regional_heatmap(scores: dict[str, float],
                     grid: Sequence[Sequence[str]] = DEFAULT_GRID) -> np.ndarray:
    """Paint per-region scores back onto the grid, for the figure's last panel."""
    names = grid_region_names(grid)
    out = np.zeros(names.shape, dtype=np.float32)
    for i in range(names.shape[0]):
        for j in range(names.shape[1]):
            out[i, j] = scores.get(names[i, j], 0.0)
    return out


def region_masks_from_atlas(atlas_volume: np.ndarray,
                            label_to_region: dict[int, str],
                            slice_index: int) -> dict[str, np.ndarray] | None:
    """Hook for atlas-based regions instead of the 6x6 grid.

    Supply a registered parcellation in the same space as the preprocessed
    volume plus a label -> region mapping, and this returns per-region boolean
    masks for the requested axial slice. Pass those to `region_scores_from_masks`
    for atlas-grade regional scores.
    """
    if atlas_volume.ndim != 3:
        raise ValueError("atlas_volume must be a 3D parcellation")
    plane = atlas_volume[:, :, slice_index]
    masks: dict[str, np.ndarray] = {}
    for label, region in label_to_region.items():
        mask = plane == label
        if mask.any():
            masks[region] = masks.get(region, np.zeros_like(mask)) | mask
    return masks or None


def region_scores_from_masks(saliency: np.ndarray, masks: dict[str, np.ndarray],
                             scaling: str = "peak") -> dict[str, float]:
    """Regional means using explicit masks rather than the grid."""
    values = {r: float(saliency[m].mean()) if m.any() else float("nan")
              for r, m in masks.items()}
    if scaling == "raw":
        return values
    peak = max(v for v in values.values() if np.isfinite(v)) + 1e-8
    return {r: v / peak for r, v in values.items()}
