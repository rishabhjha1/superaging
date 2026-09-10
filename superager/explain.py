"""Explainability methods for the trained ViT-B/16 (paper Sec. 3.5).

Four independent views of the same model:

  1. Attention rollout -- recursively multiplies residual-augmented attention
     matrices across encoder layers, upsampled to slice resolution.
  2. Grad-CAM -- weights final-block token embeddings by the gradient of the
     target-class score.
  3. LIME -- fits an interpretable linear model to 1000 masking perturbations
     over SLIC superpixels.
  4. Grid-based regional analysis -- see `regions.py`.

All operate on the axial mid-slice. Per-plane maps are reported separately
because the three planes are mean-fused *after* the backbone: there is no single
fused attention map, and averaging the three rollouts would blend anatomy from
orthogonal planes into a map corresponding to no real slice.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.segmentation import slic

from .config import Config
from .data import AXIAL, extract_slice, load_volume, normalise_volume, resize_2d
from .models import ViTNet, to_imagenet
from .regions import (
    REGIONS,
    RegionalScores,
    aggregate_regional,
    grid_region_names,
    region_scores,
    regional_agreement,
)

log = logging.getLogger(__name__)


def normalise01(a: np.ndarray) -> np.ndarray:
    return (a - a.min()) / (a.max() - a.min() + 1e-8)


def upsample(m: np.ndarray, size: int = 224) -> np.ndarray:
    """Bilinearly upsample a coarse patch map to slice resolution."""
    t = torch.from_numpy(m.astype(np.float32))[None, None]
    return F.interpolate(t, (size, size), mode="bilinear", align_corners=False)[0, 0].numpy()


# ============================================================ INPUT BUILDING
def assert_axial(volume: np.ndarray, slice_2d: np.ndarray) -> None:
    """Verify the extracted plane really is axial in canonical RAS.

    After reorientation the axes are (L->R, P->A, I->S), so an axial slice is
    volume[:, :, k] and its in-plane shape must be (axis0, axis1). A transposed
    volume silently swaps frontal and parietal grid labels while leaving every
    number in Table 2 unchanged and plausible-looking.
    """
    expected = (volume.shape[0], volume.shape[1])
    if slice_2d.shape != expected:
        raise AssertionError(
            f"Extracted slice {slice_2d.shape} does not match the axial in-plane "
            f"shape {expected}. The volume is not in canonical RAS, so region "
            f"labels would be wrong.")


def build_input(path, cfg: Config, fraction: float = 0.5,
                reference: np.ndarray | None = None) -> np.ndarray:
    """Preprocess one scan into a (3, size, size) model input.

    The axial plane is taken at `fraction` depth; coronal and sagittal stay at
    their mid-slice, so only the plane under analysis moves during the sweep.
    """
    volume = normalise_volume(load_volume(path), reference, cfg.mask_threshold)
    axial = extract_slice(volume, "axial", fraction)
    assert_axial(volume, axial)
    return np.stack([
        resize_2d(axial, cfg.size_2d),
        resize_2d(extract_slice(volume, "coronal", 0.5), cfg.size_2d),
        resize_2d(extract_slice(volume, "sagittal", 0.5), cfg.size_2d),
    ]).astype(np.float32)


def load_vit(checkpoint: str, cfg: Config, device: torch.device):
    """Rebuild the ViT as trained and load its weights."""
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = ViTNet(cfg).to(device)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    if missing or unexpected:
        log.warning("State-dict mismatch | missing=%s | unexpected=%s",
                    list(missing)[:3], list(unexpected)[:3])
    model.eval()
    # Grad-CAM requires a backward pass through the frozen backbone. This enables
    # gradient flow only; no optimiser is attached, so no weight is ever updated.
    for p in model.parameters():
        p.requires_grad_(True)
    log.info("Loaded ViT from %s (seed %s, threshold %.2f)",
             checkpoint, ckpt.get("seed", "?"), ckpt.get("threshold", 0.5))
    return model, float(ckpt.get("threshold", 0.5))


# ============================================================ ATTENTION ROLLOUT
class AttentionCatcher:
    """Capture per-layer attention from torchvision's ViT encoder.

    torchvision's EncoderBlock calls `self_attention(x, x, x, need_weights=False)`,
    so the attention matrices are discarded before any forward hook can see them.
    This patches each MultiheadAttention forward to request head-averaged
    weights, then restores the original forward on exit.
    """

    def __init__(self, vit: nn.Module):
        self.layers = list(vit.encoder.layers)
        self.attentions: list[torch.Tensor] = []
        self._originals: list = []

    def __enter__(self) -> AttentionCatcher:
        self.attentions, self._originals = [], []
        for layer in self.layers:
            mha = layer.self_attention
            original = mha.forward
            self._originals.append((mha, original))

            def patched(query, key, value, _orig=original, **kwargs):
                kwargs["need_weights"] = True
                kwargs["average_attn_weights"] = True
                out, weights = _orig(query, key, value, **kwargs)
                self.attentions.append(weights.detach())
                return out, weights

            mha.forward = patched
        return self

    def __exit__(self, *exc) -> None:
        for mha, original in self._originals:
            mha.forward = original


def rollout_from_attentions(attentions: Sequence[torch.Tensor],
                            residual: float = 0.5) -> np.ndarray:
    """Residual-augmented attention rollout -> square patch map in [0, 1].

    Each layer's attention is augmented with the identity to account for the
    residual path (A_hat = 0.5 A + 0.5 I), row-normalised, and multiplied across
    layers. The CLS row of the product gives each patch's contribution.
    """
    if not attentions:
        raise ValueError("No attention matrices captured -- use AttentionCatcher.")

    device = attentions[0].device
    n_tokens = attentions[0].shape[-1]
    eye = torch.eye(n_tokens, device=device).unsqueeze(0)
    result = eye.clone()

    for attn in attentions:
        a = attn.mean(dim=0, keepdim=True) if attn.shape[0] > 1 else attn
        a = residual * a.float() + (1.0 - residual) * eye
        result = torch.bmm(a / (a.sum(-1, keepdim=True) + 1e-8), result)

    cls_to_patches = result[0, 0, 1:]      # row 0 = CLS, drop the CLS column
    side = int(np.sqrt(cls_to_patches.numel()))
    if side * side != cls_to_patches.numel():
        raise ValueError(f"{cls_to_patches.numel()} patch tokens is not a square grid.")
    return normalise01(cls_to_patches.reshape(side, side).cpu().numpy())


@torch.no_grad()
def attention_rollout(model: ViTNet, x: torch.Tensor,
                      plane: int = AXIAL) -> np.ndarray:
    """Attention rollout for one plane of a (1, 3, H, W) input."""
    with AttentionCatcher(model.backbone) as catcher:
        model.backbone(to_imagenet(x[:, plane:plane + 1]))
        attentions = list(catcher.attentions)
    return upsample(rollout_from_attentions(attentions), x.shape[-1])


# ============================================================ GRAD-CAM
def grad_cam(model: ViTNet, x: torch.Tensor, target_class: int = 1,
             plane: int = AXIAL):
    """Grad-CAM on the final encoder block's token embeddings.

    On a transformer the analogue of a conv feature map is the final block's
    token sequence, shape (B*3, 197, 768). Channel weights are the gradient of
    the target logit averaged over tokens; the CAM is the ReLU'd weighted sum
    over the embedding dimension. The CLS token has no spatial location, so it
    is dropped before reshaping.

    Returns (cam at slice resolution in [0, 1], class probabilities).
    """
    store: dict[str, torch.Tensor] = {}
    target = model.backbone.encoder.ln
    handles = [
        target.register_forward_hook(lambda m, i, o: store.__setitem__("act", o)),
        target.register_full_backward_hook(
            lambda m, gi, go: store.__setitem__("grad", go[0])),
    ]
    try:
        model.zero_grad(set_to_none=True)
        logits = model(x.clone().requires_grad_(True))
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        logits[0, target_class].backward()

        if "act" not in store or "grad" not in store:
            raise RuntimeError("No activations captured; check the hook target.")
        act = store["act"][plane:plane + 1]
        grad = store["grad"][plane:plane + 1]

        cam = F.relu((grad.mean(dim=1, keepdim=True) * act).sum(dim=-1))[0, 1:]
        side = int(np.sqrt(cam.numel()))
        cam = normalise01(cam.reshape(side, side).detach().cpu().numpy())
        return upsample(cam, x.shape[-1]), probs
    finally:
        for h in handles:
            h.remove()


# ============================================================ LIME
def run_lime(model: ViTNet, x_np: np.ndarray, cfg: Config,
             device: torch.device, plane: int = AXIAL) -> dict:
    """LIME over SLIC superpixels of the analysed plane.

    Only that plane is perturbed; the other two channels keep their real values.
    Coefficients therefore read as "holding the other views fixed, this is how
    the axial superpixels move the decision" -- a conditional explanation, which
    is the honest reading given mean fusion. Perturbing all three planes at once
    would confound which view carried the signal.
    """
    from lime import lime_image

    plane_img = x_np[plane]
    rgb = np.stack([plane_img] * 3, axis=-1).astype(np.double)

    def predict_fn(images: np.ndarray) -> np.ndarray:
        out = []
        for b in range(0, len(images), 32):
            batch = images[b:b + 32]
            x = np.repeat(x_np[None], len(batch), axis=0).copy()
            x[:, plane] = batch[..., 0].astype(np.float32)
            with torch.no_grad():
                logits = model(torch.from_numpy(x).float().to(device))
                out.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(out)

    def segmentation_fn(image):
        return slic(image, n_segments=cfg.lime_segments,
                    compactness=cfg.lime_compactness, start_label=0, channel_axis=-1)

    explainer = lime_image.LimeImageExplainer(random_state=cfg.seed)
    explanation = explainer.explain_instance(
        rgb, predict_fn, labels=(1,), top_labels=None, hide_color=0,
        num_samples=cfg.lime_samples, segmentation_fn=segmentation_fn,
        random_seed=cfg.seed)

    weights = dict(explanation.local_exp[1])
    saliency = np.zeros_like(explanation.segments, dtype=np.float32)
    for seg_id, w in weights.items():
        saliency[explanation.segments == seg_id] = w   # signed, not clipped

    fidelity = float(getattr(explanation, "score", float("nan")))
    n_pos = int(sum(1 for w in weights.values() if w > 0))
    n_neg = int(sum(1 for w in weights.values() if w <= 0))
    log.info("LIME: %d superpixels | %d pro-SuperAger / %d pro-Normal | fidelity %.2f",
             len(weights), n_pos, n_neg, fidelity)

    return dict(segments=explanation.segments, saliency=saliency, image=plane_img,
                weights=weights, fidelity=fidelity, n_pos=n_pos, n_neg=n_neg,
                top=sorted(weights.items(), key=lambda kv: abs(kv[1]), reverse=True)[:10])


# ============================================================ GRID ANALYSIS
@torch.no_grad()
def cell_predictions(model: ViTNet, x_np: np.ndarray, cfg: Config,
                     device: torch.device, plane: int = AXIAL) -> np.ndarray:
    """P(SuperAger) when only one grid cell of the analysed plane is retained.

    Cell retention rather than occlusion: this asks how far each cell alone
    moves the decision, which is what a 0/1 grid of per-cell predictions means.
    """
    G, size = cfg.grid_size, x_np.shape[-1]
    edges = np.linspace(0, size, G + 1).astype(int)

    variants = []
    for i in range(G):
        for j in range(G):
            v = x_np.copy()
            kept = np.zeros_like(v[plane])
            kept[edges[i]:edges[i + 1], edges[j]:edges[j + 1]] = \
                v[plane][edges[i]:edges[i + 1], edges[j]:edges[j + 1]]
            v[plane] = kept
            variants.append(v)

    batch = torch.from_numpy(np.stack(variants)).float().to(device)
    probs = [torch.softmax(model(batch[b:b + 16]), dim=1)[:, 1].cpu().numpy()
             for b in range(0, len(batch), 16)]
    return np.concatenate(probs).reshape(G, G)


def regional_probabilities(cell_probs: np.ndarray) -> dict[str, float]:
    """Mean per-cell P(SuperAger) per region -- the figure's bar chart."""
    names = grid_region_names()
    return {r: float(cell_probs[names == r].mean()) for r in REGIONS}


def analyse_subject(model: ViTNet, path, cfg: Config, device: torch.device,
                    reference: np.ndarray | None = None) -> list[RegionalScores]:
    """Rollout-derived I_R and S_R for one subject across the depth sweep."""
    samples: list[RegionalScores] = []
    for fraction in cfg.slice_fractions:
        x_np = build_input(path, cfg, fraction, reference)
        x = torch.from_numpy(x_np[None]).float().to(device)
        importance = region_scores(attention_rollout(model, x),
                                   scaling=cfg.region_scaling)
        with torch.no_grad():
            p_super = float(torch.softmax(model(x), dim=1)[0, 1].cpu())
        samples.append(RegionalScores(
            importance=importance,
            superager_score={r: v * p_super for r, v in importance.items()},
            probability=p_super, slice_fraction=fraction))
    return samples


def build_regional_table(model: ViTNet, paths: Sequence, cfg: Config,
                         device: torch.device,
                         reference: np.ndarray | None = None) -> pd.DataFrame:
    """Pool subjects x depth offsets into the paper's Table 2."""
    samples: list[RegionalScores] = []
    for i, path in enumerate(paths):
        samples.extend(analyse_subject(model, path, cfg, device, reference))
        if (i + 1) % 5 == 0:
            log.info("  regional analysis: %d/%d subjects", i + 1, len(paths))
    return aggregate_regional(samples)


def method_agreement(rollout_map: np.ndarray, cam_map: np.ndarray,
                     lime_saliency: np.ndarray, scaling: str = "peak") -> pd.DataFrame:
    """Pairwise regional agreement between the three post-hoc methods.

    With n = 5 regions these coefficients indicate whether two methods RANK
    regions alike -- they are not precise effect sizes.
    """
    scores = {"attention": region_scores(rollout_map, scaling=scaling),
              "gradcam": region_scores(cam_map, scaling=scaling),
              "lime": region_scores(np.abs(lime_saliency), scaling=scaling)}
    names = list(scores)
    rows = [dict(method_a=a, method_b=b, **regional_agreement(scores[a], scores[b]))
            for i, a in enumerate(names) for b in names[i + 1:]]
    return pd.DataFrame(rows)
