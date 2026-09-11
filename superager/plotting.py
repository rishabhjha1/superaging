"""Figures: the explainability stack (Fig. 2) and the seed-summary bar chart."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from skimage.segmentation import mark_boundaries

from .regions import DISPLAY_NAMES, grid_region_names

PALETTE = ["#34495e", "#8e44ad", "#27ae60", "#c0392b", "#2980b9"]


def _bare(ax, title: str) -> None:
    ax.set_title(title, fontsize=8, fontweight="bold", pad=6)
    ax.set_xticks([])
    ax.set_yticks([])


def explainability_stack(exemplar: dict, path: str | Path, grid_size: int = 6,
                         dpi: int = 150) -> Path:
    """Three-block figure: grid analysis, Grad-CAM/attention, LIME."""
    fig = plt.figure(figsize=(11, 11.5))
    gs = gridspec.GridSpec(3, 1, hspace=0.42)

    # ---------------- Block 1: grid-based regional analysis ----------------
    g1 = gridspec.GridSpecFromSubplotSpec(1, 3, gs[0], wspace=0.30,
                                          width_ratios=[1, 1.15, 1.25])
    ax = fig.add_subplot(g1[0])
    ax.imshow(exemplar["slice"], cmap="gray")
    _bare(ax, "Original Brain Slice\n(Middle Axial)")

    ax = fig.add_subplot(g1[1])
    cells = exemplar["cell_probs"]
    im = ax.imshow(cells, cmap="RdYlBu_r", vmin=0, vmax=1)
    for i in range(grid_size):
        for j in range(grid_size):
            ax.text(j, i, int(cells[i, j] >= 0.5), ha="center", va="center",
                    color="white", fontsize=8, fontweight="bold")
    ax.set_title("SuperAger Predictions\n(1=SuperAger, 0=Normal)",
                 fontsize=8, fontweight="bold", pad=6)
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    ax.tick_params(labelsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046).ax.tick_params(labelsize=6)

    ax = fig.add_subplot(g1[2])
    order = ["Subcortical", "Central", "Temporal", "Parietal", "Frontal"]
    values = [exemplar["region_probs"][r] for r in order]
    colors = ["#1f4fd8" if r in ("Subcortical", "Central") else "#a01028" for r in order]
    ax.barh(range(len(order)), values, color=colors, height=0.6)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([DISPLAY_NAMES[r] for r in order], fontsize=6)
    ax.set_xlabel("Super Ager Probability", fontsize=6)
    ax.set_xlim(0, 1.0)
    ax.tick_params(axis="x", labelsize=6)
    ax.grid(axis="x", alpha=0.2)
    fig.text(0.5, 0.665, "Grid Based Regional Analysis", ha="center",
             fontsize=11, fontweight="bold")

    # ---------------- Block 2: Grad-CAM and attention ----------------
    g2 = gridspec.GridSpecFromSubplotSpec(1, 4, gs[1], wspace=0.32)
    ax = fig.add_subplot(g2[0])
    ax.imshow(exemplar["slice"], cmap="gray")
    _bare(ax, "Original Brain MRI\n(Middle Axial Slice)")

    ax = fig.add_subplot(g2[1])
    ax.imshow(exemplar["slice"], cmap="gray")
    im = ax.imshow(exemplar["gradcam"], cmap="jet", alpha=0.55)
    _bare(ax, "Attention Heatmap\n(Model Focus Areas)")
    fig.colorbar(im, ax=ax, fraction=0.046).ax.tick_params(labelsize=6)

    ax = fig.add_subplot(g2[2])
    im = ax.imshow(exemplar["rollout"], cmap="hot")
    _bare(ax, "Attention Weights\n(Higher = More Important)")
    fig.colorbar(im, ax=ax, fraction=0.046).ax.tick_params(labelsize=6)

    ax = fig.add_subplot(g2[3])
    probs = exemplar["probs"]
    ax.barh([0, 1], [probs[0], probs[1]], color=["#8080e0", "#f08080"], height=0.55)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Normal", "SuperAger"], fontsize=7)
    ax.set_xlim(0, 1.0)
    ax.tick_params(axis="x", labelsize=6)
    for i, p in enumerate(probs[:2]):
        ax.text(p + 0.03, i, f"{p:.3f}", va="center", fontsize=7, fontweight="bold")
    ax.set_title("Classification Probabilities", fontsize=8, fontweight="bold", pad=6)
    ax.grid(axis="x", alpha=0.2)
    fig.text(0.5, 0.345, "Grad-CAM based Analysis", ha="center",
             fontsize=11, fontweight="bold")

    # ---------------- Block 3: LIME ----------------
    lime = exemplar["lime"]
    g3 = gridspec.GridSpecFromSubplotSpec(1, 3, gs[2], wspace=0.32)
    ax = fig.add_subplot(g3[0])
    top = sorted(lime["top"], key=lambda kv: kv[1])
    values = [w for _, w in top]
    ax.barh(range(len(top)), values, height=0.6,
            color=["#3f9d5a" if w > 0 else "#d94a6a" for w in values])
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([f"Region {sid}" for sid, _ in top], fontsize=6)
    ax.set_xlabel("Feature Importance", fontsize=6)
    ax.axvline(0, color="black", lw=0.7)
    ax.tick_params(axis="x", labelsize=6)
    ax.set_title("Brain Region Importance\nPro-SuperAger | Pro-Normal",
                 fontsize=8, fontweight="bold", pad=6)

    ax = fig.add_subplot(g3[1])
    ax.imshow(mark_boundaries(np.stack([lime["image"]] * 3, axis=-1),
                              lime["segments"], color=(1, 1, 0), mode="thick"))
    _bare(ax, "Brain Segmentation\n(Analysis Regions)")

    ax = fig.add_subplot(g3[2])
    ax.pie([lime["n_pos"], lime["n_neg"]], colors=["#2e8b57", "#d61f43"],
           labels=[f"SuperAger\nSupport\n({lime['n_pos']})",
                   f"Normal\nSupport\n({lime['n_neg']})"],
           autopct="%1.1f%%", startangle=90, textprops={"fontsize": 6})
    ax.set_title("Region Analysis\nDistribution", fontsize=8, fontweight="bold", pad=6)
    fig.text(0.5, 0.028, "LIME based Analysis", ha="center",
             fontsize=11, fontweight="bold")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def orientation_check(slice_2d: np.ndarray, volume_shape, path: str | Path,
                      dpi: int = 150) -> Path:
    """Overlay the labelled grid on a real slice so orientation is visible.

    A flipped or transposed volume swaps frontal and parietal labels while
    leaving every number in Table 2 unchanged. This is the only way to catch it.
    """
    names = grid_region_names()
    G = names.shape[0]
    edges = np.linspace(0, slice_2d.shape[0], G + 1).astype(int)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.imshow(slice_2d, cmap="gray")
    for e in edges:
        ax.axhline(e, color="yellow", lw=0.7)
        ax.axvline(e, color="yellow", lw=0.7)
    for i in range(G):
        for j in range(G):
            ax.text((edges[j] + edges[j + 1]) / 2, (edges[i] + edges[i + 1]) / 2,
                    names[i, j][:4], ha="center", va="center",
                    color="cyan", fontsize=9, fontweight="bold")
    ax.set_title(f"Orientation check | volume shape {tuple(volume_shape)}\n"
                 "Frontal must be at the TOP. If not, region labels are wrong.",
                 fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def seed_summary(results: dict[str, dict[str, list]], model_names: Sequence[str],
                 path: str | Path, dpi: int = 150) -> Path:
    """Grouped bars with per-seed dots, so the spread is visible, not just the mean."""
    metrics = ["auc", "auprc", "balanced_accuracy", "f1"]
    labels = ["Test AUC", "Test AUPRC", "Test BalAcc", "Test F1"]
    colors = {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(model_names)}
    x = np.arange(len(metrics))
    width = 0.8 / len(model_names)
    offsets = {n: -0.4 + width / 2 + i * width for i, n in enumerate(model_names)}

    fig, ax = plt.subplots(figsize=(12, 5.5))
    for name in model_names:
        means = [np.nanmean(results[name][m]) for m in metrics]
        stds = [np.nanstd(results[name][m]) for m in metrics]
        ax.bar(x + offsets[name], means, width, yerr=stds, capsize=4,
               label=name, color=colors[name], alpha=0.85)
        for mi, m in enumerate(metrics):
            points = results[name][m]
            ax.scatter(np.full(len(points), x[mi] + offsets[name]), points,
                       s=16, color="black", alpha=0.5, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.axhline(0.5, ls="--", c="gray", lw=0.8)
    n_seeds = len(results[model_names[0]]["auc"])
    ax.set_title(f"Five-model comparison - held-out test, mean ± std over "
                 f"{n_seeds} seeds (dots = seeds)", fontsize=12)
    ax.legend(fontsize=9, ncol=len(model_names))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
