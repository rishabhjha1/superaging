"""Streamlit demo for the SuperAger ViT-B/16 explainability pipeline.

Local:   streamlit run streamlit_app.py
Deploy:  share.streamlit.io -> repo, branch main, main file streamlit_app.py

Point SUPERAGER_CKPT at a checkpoint written by `train.py` (results/checkpoints/
vit_seed42.pt). Without one the app still runs end to end, but on an untrained
head, so the numbers mean nothing.

This demo reuses the same functions as `explain.py`, so what it shows is the
pipeline described in the paper and not a reimplementation of it. One difference
is unavoidable: `preprocess` in `superager.data` applies a per-site z-score
across a whole cohort (Sec. 3.2, stage 6), which a single uploaded scan cannot
support. The app stops after the per-scan stages, exactly as `explain.py` does
for its exemplar.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from superager.config import load_config
from superager.explain import attention_rollout, build_input, load_vit
from superager.models import build_model
from superager.regions import DISPLAY_NAMES, REGIONS, region_scores

CKPT = Path(os.environ.get("SUPERAGER_CKPT", "results/checkpoints/vit_seed42.pt"))
PLANES = ("axial", "coronal", "sagittal")

st.set_page_config(page_title="SuperAger explainability demo", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    """Build the ViT as trained. Returns (model, threshold, is_trained)."""
    cfg = load_config(None)
    device = torch.device("cpu")
    if CKPT.exists():
        model, threshold = load_vit(str(CKPT), cfg, device)
        return model, threshold, True
    # Smoke-test path: the plumbing runs, the outputs are meaningless.
    model = build_model("ViT", cfg).to(device)
    model.eval()
    return model, 0.5, False


@st.cache_data(show_spinner="Preprocessing volume...")
def preprocess_upload(raw: bytes, name: str) -> np.ndarray:
    """Bytes from the uploader -> (3, 224, 224) float32 (axial, coronal, sagittal)."""
    cfg = load_config(None)
    suffix = ".nii.gz" if name.endswith(".gz") else ".nii"
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir) / f"upload{suffix}"
        tmp.write_bytes(raw)
        # reference=None: histogram matching needs a cohort to pool a reference from.
        return build_input(tmp, cfg, fraction=0.5, reference=None)


def show_map(base: np.ndarray, overlay: np.ndarray | None, title: str):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(base, cmap="gray")
    if overlay is not None:
        ax.imshow(overlay, cmap="jet", alpha=0.45)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


st.title("Explainable classification of cognitive SuperAgers")
st.caption(
    "Upload a T1-weighted NIfTI volume. The scan is sent to the server for "
    "processing and is not stored after the session ends."
)

cfg = load_config(None)
model, threshold, trained = load_model()
if not trained:
    st.warning(
        f"No checkpoint at {CKPT} — running with an untrained head. "
        "The pipeline executes end to end but the outputs are not meaningful. "
        "Train one with `python train.py --models ViT`, or set SUPERAGER_CKPT."
    )

upload = st.file_uploader("T1-weighted MRI", type=["nii", "gz"])
if upload is None:
    st.info("Waiting for a .nii or .nii.gz file.")
    st.stop()

x_np = preprocess_upload(upload.getvalue(), upload.name)
x = torch.from_numpy(x_np[None]).float()  # (1, 3 planes, 224, 224)

with torch.no_grad():
    prob = float(torch.softmax(model(x), dim=1)[0, 1])

left, right = st.columns(2)
left.metric("P(SuperAger)", f"{prob:.3f}")
right.metric(
    f"Decision at threshold {threshold:.2f}",
    "SuperAger" if prob >= threshold else "Typical ager",
)
st.progress(prob)

st.subheader("Attention rollout")
st.caption(
    "The three planes are mean-fused *after* the backbone, so each plane has its "
    "own rollout and there is no single fused map."
)
saliency = {}
for i, (col, plane) in enumerate(zip(st.columns(len(PLANES)), PLANES)):
    with col:
        saliency[plane] = attention_rollout(model, x, plane=i)
        show_map(x_np[i], saliency[plane], plane)

st.subheader("Regional analysis (axial)")
scores = region_scores(saliency["axial"], scaling=cfg.region_scaling)
table = pd.DataFrame(
    [{"Region": DISPLAY_NAMES[r], "I_R": scores[r], "S_R": scores[r] * prob}
     for r in REGIONS]
).sort_values("I_R", ascending=False)
st.dataframe(table.style.format({"I_R": "{:.3f}", "S_R": "{:.3f}"}),
             hide_index=True, use_container_width=True)

st.caption(
    f"I_R uses '{cfg.region_scaling}' scaling, so values are RELATIVE attention "
    "(top region ~1.0); S_R = I_R x P(SuperAger). The 6x6 grid is a coarse "
    "positional proxy, not an atlas registration, and region labels assume a "
    "canonical RAS orientation."
)
