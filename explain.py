"""
Explainability stack for the trained ViT-B/16 (paper Sec. 3.5 / 4.2).

Four methods: attention rollout, Grad-CAM, LIME, and grid-based regional
analysis. Produces the three-block figure and the regional importance table.

Usage:
    python explain.py --ckpt results/vit_seed42.pt --labels data/labels.csv \
                      --roots data/ADNI data/OASIS
    python explain.py --ckpt ... --check-orientation   # run this FIRST

WARNING -- read before quoting anatomy:
  * The 6x6 grid assumes a canonical-RAS AXIAL slice with frontal at the top.
    A flipped or transposed volume swaps the frontal and parietal labels while
    leaving every number in the table unchanged and plausible-looking.
    --check-orientation is the only way to see this. Run it once per dataset.
  * Importance is PEAK-SCALED: the top region sits at ~1.0 by construction and
    the rest are fractions of it. Report values as relative attention.
  * The grid is a coarse lobar proxy, not an atlas-registered parcellation.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import pearsonr
from skimage.segmentation import mark_boundaries, slic

from train import (SIZE, DEV, ViTNet, collect_records, load_volume, normalise,
                   resize2d, to_imagenet, set_seed)

# 6x6 grid -> five coarse regions, for a canonical axial mid-slice.
# Rows run posterior -> anterior; F must land on the anterior half of the image.
GRID = np.array([list("FFFFFF"), list("FFFFFF"), list("TCCCCT"),
                 list("TSSSST"), list("PPPPPP"), list("PPPPPP")])
CODES = {"F": "Frontal", "P": "Parietal", "T": "Temporal",
         "C": "Central", "S": "Subcortical"}
NAMES = np.vectorize(CODES.get)(GRID)
REGIONS = ("Frontal", "Parietal", "Temporal", "Central", "Subcortical")
AXIAL = 0  # channel order from train.triplanar is (axial, coronal, sagittal)


# ============================================================ INPUTS
def build_input(path, frac=0.5):
    """Preprocess one scan; the axial slice is taken at `frac` depth."""
    v = normalise(load_volume(path))
    k = int(round(frac * (v.shape[2] - 1)))
    ax = v[:, :, k]
    if ax.shape != (v.shape[0], v.shape[1]):
        raise AssertionError("Slice is not axial in canonical RAS -- region "
                             "labels would be wrong. Check nib.as_closest_canonical.")
    return np.stack([resize2d(ax),
                     resize2d(v[:, v.shape[1] // 2, :]),
                     resize2d(v[v.shape[0] // 2, :, :])]).astype(np.float32)


def load_model(ckpt):
    m = ViTNet().to(DEV)
    m.load_state_dict(torch.load(ckpt, map_location=DEV)["state_dict"], strict=False)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(True)   # Grad-CAM needs backward; nothing is updated
    return m


def upsample(m, size=SIZE):
    t = torch.from_numpy(m.astype(np.float32))[None, None]
    return F.interpolate(t, (size, size), mode="bilinear", align_corners=False)[0, 0].numpy()


def norm01(a):
    return (a - a.min()) / (a.max() - a.min() + 1e-8)


# ============================================================ ATTENTION ROLLOUT
class Catcher:
    """torchvision's EncoderBlock calls attention with need_weights=False, so a
    plain hook returns nothing. Patch each forward to request weights, restore on exit.
    """

    def __init__(self, vit):
        self.layers = list(vit.encoder.layers)

    def __enter__(self):
        self.attn, self.orig = [], []
        for lyr in self.layers:
            mha = lyr.self_attention
            o = mha.forward
            self.orig.append((mha, o))

            def patched(q, k, v, _o=o, **kw):
                kw.update(need_weights=True, average_attn_weights=True)
                out, w = _o(q, k, v, **kw)
                self.attn.append(w.detach())
                return out, w
            mha.forward = patched
        return self

    def __exit__(self, *e):
        for mha, o in self.orig:
            mha.forward = o


@torch.no_grad()
def rollout(model, x):
    """Residual-augmented attention rollout over the axial plane -> 224x224 map."""
    with Catcher(model.backbone) as c:
        model.backbone(to_imagenet(x[:, AXIAL:AXIAL + 1]))
        attns = list(c.attn)

    L = attns[0].shape[-1]
    eye = torch.eye(L, device=attns[0].device).unsqueeze(0)
    r = eye.clone()
    for a in attns:
        a = 0.5 * a.float() + 0.5 * eye          # identity = the residual path
        r = torch.bmm(a / (a.sum(-1, keepdim=True) + 1e-8), r)

    cls = r[0, 0, 1:]                             # CLS row, patch columns
    s = int(np.sqrt(cls.numel()))
    return upsample(norm01(cls.reshape(s, s).cpu().numpy()))


# ============================================================ GRAD-CAM
def gradcam(model, x, target=1):
    """Grad-CAM on the final encoder block's token embeddings (the ViT analogue
    of a conv feature map). The CLS token has no location, so it is dropped."""
    store = {}
    h1 = model.backbone.encoder.ln.register_forward_hook(
        lambda m, i, o: store.__setitem__("a", o))
    h2 = model.backbone.encoder.ln.register_full_backward_hook(
        lambda m, gi, go: store.__setitem__("g", go[0]))
    try:
        model.zero_grad(set_to_none=True)
        logits = model(x.clone().requires_grad_(True))
        probs = torch.softmax(logits, 1)[0].detach().cpu().numpy()
        logits[0, target].backward()
        a, g = store["a"][AXIAL:AXIAL + 1], store["g"][AXIAL:AXIAL + 1]
        cam = F.relu((g.mean(1, keepdim=True) * a).sum(-1))[0, 1:]
        s = int(np.sqrt(cam.numel()))
        return upsample(norm01(cam.reshape(s, s).detach().cpu().numpy())), probs
    finally:
        h1.remove(); h2.remove()


# ============================================================ LIME
def run_lime(model, x_np, n_samples=1000, n_segments=50):
    """LIME over SLIC superpixels of the axial plane.

    Only the axial channel is perturbed; coronal and sagittal keep their real
    values, so coefficients read as "holding the other two views fixed".
    """
    from lime import lime_image

    plane = x_np[AXIAL]
    rgb = np.stack([plane] * 3, -1).astype(np.double)

    def predict(images):
        out = []
        for b in range(0, len(images), 32):
            batch = images[b:b + 32]
            x = np.repeat(x_np[None], len(batch), 0).copy()
            x[:, AXIAL] = batch[..., 0].astype(np.float32)
            with torch.no_grad():
                out.append(torch.softmax(
                    model(torch.from_numpy(x).float().to(DEV)), 1).cpu().numpy())
        return np.concatenate(out)

    exp = lime_image.LimeImageExplainer(random_state=42).explain_instance(
        rgb, predict, labels=(1,), top_labels=None, hide_color=0,
        num_samples=n_samples, random_seed=42,
        segmentation_fn=lambda im: slic(im, n_segments=n_segments,
                                        compactness=10, start_label=0,
                                        channel_axis=-1))

    weights = dict(exp.local_exp[1])
    saliency = np.zeros_like(exp.segments, np.float32)
    for sid, w in weights.items():
        saliency[exp.segments == sid] = w
    top = sorted(weights.items(), key=lambda kv: abs(kv[1]), reverse=True)[:10]
    return dict(segments=exp.segments, saliency=saliency, image=plane, top=top,
                n_pos=sum(w > 0 for w in weights.values()),
                n_neg=sum(w <= 0 for w in weights.values()))


# ============================================================ REGIONAL ANALYSIS
def region_scores(saliency, scaling="peak"):
    """Pool any saliency map onto the 6x6 grid, then average per region."""
    t = torch.from_numpy(np.asarray(saliency, np.float32))[None, None]
    pooled = F.adaptive_avg_pool2d(t, (6, 6))[0, 0].numpy()
    vals = {r: float(pooled[NAMES == r].mean()) for r in REGIONS}
    if scaling == "raw":
        return vals
    peak = max(vals.values()) + 1e-8   # top region -> ~1.0: a display convention
    return {r: v / peak for r, v in vals.items()}


@torch.no_grad()
def cell_predictions(model, x_np):
    """P(SuperAger) when only one 6x6 cell of the axial plane is retained.

    Cell retention, not occlusion: asks how far each cell alone moves the
    decision, which is what a 0/1 grid of predictions means.
    """
    e = np.linspace(0, x_np.shape[-1], 7).astype(int)
    variants = []
    for i in range(6):
        for j in range(6):
            v = x_np.copy()
            keep = np.zeros_like(v[AXIAL])
            keep[e[i]:e[i+1], e[j]:e[j+1]] = v[AXIAL][e[i]:e[i+1], e[j]:e[j+1]]
            v[AXIAL] = keep
            variants.append(v)
    b = torch.from_numpy(np.stack(variants)).float().to(DEV)
    out = [torch.softmax(model(b[i:i + 16]), 1)[:, 1].cpu().numpy()
           for i in range(0, len(b), 16)]
    return np.concatenate(out).reshape(6, 6)


def regional_table(model, df, n_subjects=30):
    """Table 2: I_R and S_R = I_R * P(SuperAger), pooled over subjects and a
    +/-5% depth sweep so a score is not an artefact of one arbitrary slice."""
    rows = []
    for i, (_, r) in enumerate(df.head(n_subjects).iterrows()):
        for frac in (0.40, 0.45, 0.50, 0.55, 0.60):
            x_np = build_input(r.path, frac)
            x = torch.from_numpy(x_np[None]).float().to(DEV)
            imp = region_scores(rollout(model, x))
            with torch.no_grad():
                p = float(torch.softmax(model(x), 1)[0, 1])
            for reg, v in imp.items():
                rows.append(dict(region=reg, I_R=v, S_R=v * p))
        if (i + 1) % 5 == 0:
            print(f"  regional analysis: {i + 1}/{min(n_subjects, len(df))}")

    t = (pd.DataFrame(rows).groupby("region")
         .agg(I_R=("I_R", "mean"), I_R_sd=("I_R", "std"),
              S_R=("S_R", "mean"), S_R_sd=("S_R", "std"))
         .sort_values("I_R", ascending=False).reset_index())
    return t


def agreement(rollout_map, cam_map, lime_map):
    """Regional agreement between methods. With n=5 regions these coefficients
    indicate whether two methods RANK regions alike -- not an effect size."""
    s = {"attention": region_scores(rollout_map), "gradcam": region_scores(cam_map),
         "lime": region_scores(np.abs(lime_map))}
    rows = []
    keys = list(s)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            va = [s[a][r] for r in REGIONS]
            vb = [s[b][r] for r in REGIONS]
            rows.append(dict(method_a=a, method_b=b, pearson_r=pearsonr(va, vb)[0],
                             n_regions=5))
    return pd.DataFrame(rows)


# ============================================================ FIGURE
def make_figure(ex, out):
    fig = plt.figure(figsize=(11, 11.5))
    gs = gridspec.GridSpec(3, 1, hspace=0.42)
    bare = lambda ax, t: (ax.set_title(t, fontsize=8, fontweight="bold", pad=6),
                          ax.set_xticks([]), ax.set_yticks([]))

    # ---- Block 1: grid-based regional analysis ----
    g1 = gridspec.GridSpecFromSubplotSpec(1, 3, gs[0], wspace=0.30,
                                          width_ratios=[1, 1.15, 1.25])
    ax = fig.add_subplot(g1[0]); ax.imshow(ex["slice"], cmap="gray")
    bare(ax, "Original Brain Slice\n(Middle Axial)")

    ax = fig.add_subplot(g1[1])
    im = ax.imshow(ex["cells"], cmap="RdYlBu_r", vmin=0, vmax=1)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, int(ex["cells"][i, j] >= 0.5), ha="center", va="center",
                    color="white", fontsize=8, fontweight="bold")
    ax.set_title("SuperAger Predictions\n(1=SuperAger, 0=Normal)",
                 fontsize=8, fontweight="bold", pad=6)
    ax.set_xticks(range(6)); ax.set_yticks(range(6)); ax.tick_params(labelsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046).ax.tick_params(labelsize=6)

    ax = fig.add_subplot(g1[2])
    order = ["Subcortical", "Central", "Temporal", "Parietal", "Frontal"]
    ax.barh(range(5), [ex["region_p"][r] for r in order], height=0.6,
            color=["#1f4fd8" if r in ("Subcortical", "Central") else "#a01028"
                   for r in order])
    ax.set_yticks(range(5)); ax.set_yticklabels(order, fontsize=6)
    ax.set_xlabel("Super Ager Probability", fontsize=6); ax.set_xlim(0, 1)
    ax.tick_params(axis="x", labelsize=6); ax.grid(axis="x", alpha=0.2)
    fig.text(0.5, 0.665, "Grid Based Regional Analysis", ha="center",
             fontsize=11, fontweight="bold")

    # ---- Block 2: Grad-CAM and attention ----
    g2 = gridspec.GridSpecFromSubplotSpec(1, 4, gs[1], wspace=0.32)
    ax = fig.add_subplot(g2[0]); ax.imshow(ex["slice"], cmap="gray")
    bare(ax, "Original Brain MRI\n(Middle Axial Slice)")

    ax = fig.add_subplot(g2[1]); ax.imshow(ex["slice"], cmap="gray")
    im = ax.imshow(ex["cam"], cmap="jet", alpha=0.55)
    bare(ax, "Attention Heatmap\n(Model Focus Areas)")
    fig.colorbar(im, ax=ax, fraction=0.046).ax.tick_params(labelsize=6)

    ax = fig.add_subplot(g2[2]); im = ax.imshow(ex["rollout"], cmap="hot")
    bare(ax, "Attention Weights\n(Higher = More Important)")
    fig.colorbar(im, ax=ax, fraction=0.046).ax.tick_params(labelsize=6)

    ax = fig.add_subplot(g2[3])
    ax.barh([0, 1], ex["probs"][:2], color=["#8080e0", "#f08080"], height=0.55)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Normal", "SuperAger"], fontsize=7)
    ax.set_xlim(0, 1); ax.tick_params(axis="x", labelsize=6)
    for i, p in enumerate(ex["probs"][:2]):
        ax.text(p + 0.03, i, f"{p:.3f}", va="center", fontsize=7, fontweight="bold")
    ax.set_title("Classification Probabilities", fontsize=8, fontweight="bold", pad=6)
    fig.text(0.5, 0.345, "Grad-CAM based Analysis", ha="center",
             fontsize=11, fontweight="bold")

    # ---- Block 3: LIME ----
    lime = ex["lime"]
    g3 = gridspec.GridSpecFromSubplotSpec(1, 3, gs[2], wspace=0.32)
    ax = fig.add_subplot(g3[0])
    top = sorted(lime["top"], key=lambda kv: kv[1])
    vals = [w for _, w in top]
    ax.barh(range(len(top)), vals, height=0.6,
            color=["#3f9d5a" if w > 0 else "#d94a6a" for w in vals])
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([f"Region {s}" for s, _ in top], fontsize=6)
    ax.set_xlabel("Feature Importance", fontsize=6); ax.axvline(0, color="k", lw=0.7)
    ax.tick_params(axis="x", labelsize=6)
    ax.set_title("Brain Region Importance\nPro-SuperAger | Pro-Normal",
                 fontsize=8, fontweight="bold", pad=6)

    ax = fig.add_subplot(g3[1])
    ax.imshow(mark_boundaries(np.stack([lime["image"]] * 3, -1), lime["segments"],
                              color=(1, 1, 0), mode="thick"))
    bare(ax, "Brain Segmentation\n(Analysis Regions)")

    ax = fig.add_subplot(g3[2])
    ax.pie([lime["n_pos"], lime["n_neg"]], colors=["#2e8b57", "#d61f43"],
           labels=[f"SuperAger\nSupport\n({lime['n_pos']})",
                   f"Normal\nSupport\n({lime['n_neg']})"],
           autopct="%1.1f%%", startangle=90, textprops={"fontsize": 6})
    ax.set_title("Region Analysis\nDistribution", fontsize=8, fontweight="bold", pad=6)
    fig.text(0.5, 0.028, "LIME based Analysis", ha="center",
             fontsize=11, fontweight="bold")

    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def check_orientation(path, out):
    """Overlay the labelled grid on a real slice. Frontal must be at the TOP;
    if it is not, every region name in the table is wrong."""
    v = load_volume(path)
    sl = resize2d(v[:, :, v.shape[2] // 2])
    e = np.linspace(0, SIZE, 7).astype(int)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.imshow(sl, cmap="gray")
    for k in e:
        ax.axhline(k, color="yellow", lw=0.7); ax.axvline(k, color="yellow", lw=0.7)
    for i in range(6):
        for j in range(6):
            ax.text((e[j] + e[j+1]) / 2, (e[i] + e[i+1]) / 2, NAMES[i, j][:4],
                    ha="center", va="center", color="cyan", fontsize=9,
                    fontweight="bold")
    ax.set_title(f"Orientation check | volume shape {v.shape}\n"
                 "Frontal must be at the TOP -- if not, region labels are wrong.",
                 fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Wrote {out} -- INSPECT IT before quoting anatomy.")


# ============================================================ MAIN
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/vit_seed42.pt")
    ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--roots", nargs="+", default=["data/ADNI", "data/OASIS"])
    ap.add_argument("--outdir", default="results/explain")
    ap.add_argument("--n-subjects", type=int, default=30)
    ap.add_argument("--check-orientation", action="store_true")
    args = ap.parse_args()

    set_seed(42)
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    df = collect_records(args.labels, args.roots)

    if args.check_orientation:
        check_orientation(df.path.iloc[0], out / "orientation_check.png")
        return

    model = load_model(args.ckpt)
    x_np = build_input(df.path.iloc[0])
    x = torch.from_numpy(x_np[None]).float().to(DEV)

    print("attention rollout ..."); roll = rollout(model, x)
    print("grad-cam ...");          cam, probs = gradcam(model, x)
    print("lime ...");              lime = run_lime(model, x_np)
    print("grid analysis ...");     cells = cell_predictions(model, x_np)
    region_p = {r: float(cells[NAMES == r].mean()) for r in REGIONS}

    print("regional table ...")
    table = regional_table(model, df, args.n_subjects)
    table.to_csv(out / "regional_table.csv", index=False)
    print(table.to_string(index=False))

    agreement(roll, cam, lime["saliency"]).to_csv(out / "agreement.csv", index=False)

    make_figure(dict(slice=x_np[AXIAL], cells=cells, region_p=region_p,
                     rollout=roll, cam=cam, probs=probs, lime=lime),
                out / "explainability_stack.png")
    print(f"\nSaved {out}/explainability_stack.png and regional_table.csv")
    print("Reminder: I_R is peak-scaled (relative), the grid is a lobar proxy, "
          "and this is agreement among METHODS on one trained model.")


if __name__ == "__main__":
    main()
