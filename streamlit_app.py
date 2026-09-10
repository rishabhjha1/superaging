"""Streamlit demo for the SuperAger ViT-B/16 explainability pipeline.

Local:   streamlit run streamlit_app.py
Deploy:  share.streamlit.io -> repo, branch main, main file streamlit_app.py

Set SUPERAGER_CKPT to a checkpoint produced by train.py (e.g.
results/checkpoints/vit_seed42.pt) to run the trained model; without one the
pipeline still executes end to end on a randomly initialised head so the app
can be smoke-tested, but the predictions are meaningless.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import streamlit as st
import torch

from superager.config import Config
from superager.data import AXIAL, CORONAL, SAGITTAL, normalise_volume, triplanar
from superager.explain import attention_rollout, load_vit
from superager.models import ViTNet
from superager.regions import REGIONS, region_scores

CKPT = Path(os.environ.get("SUPERAGER_CKPT", "results/checkpoints/vit_seed42.pt"))
PLANES = {"axial": AXIAL, "coronal": CORONAL, "sagittal": SAGITTAL}

st.set_page_config(page_title="SuperAger explainability demo", layout="wide")

cfg = Config()
device = torch.device("cpu")


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    """Build the ViT and load the checkpoint. Returns (model, threshold, is_trained)."""
    if CKPT.exists():
        model, threshold = load_vit(str(CKPT), cfg, device)
        return model, threshold, True
    model = ViTNet(cfg).to(device).eval()
    return model, 0.5, False  # smoke-test path: plumbing runs, numbers are meaningless


@st.cache_data(show_spinner="Preprocessing volume...")
def preprocess(raw: bytes, name: str) -> np.ndarray:
    """Bytes from the uploader -> (3, size, size) float32, planes (axial, coronal, sagittal)."""
    suffix = ".nii.gz" if name.endswith(".gz") else ".nii"
    tmp = Path("/tmp") / f"upload{suffix}"
    tmp.write_bytes(raw)
    volume = np.squeeze(nib.as_closest_canonical(nib.load(str(tmp))).get_fdata())
    normalised = normalise_volume(volume.astype(np.float32), threshold=cfg.mask_threshold)
    return triplanar(normalised, cfg.size_2d)


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

model, threshold, trained = load_model()
if not trained:
    st.warning(
        f"No checkpoint at {CKPT} — running with an untrained head. "
        "The pipeline executes end to end but the outputs are not meaningful."
    )

upload = st.file_uploader("T1-weighted MRI", type=["nii", "gz"])
if upload is None:
    st.info("Waiting for a .nii or .nii.gz file.")
    st.stop()

x_np = preprocess(upload.getvalue(), upload.name)
x = torch.from_numpy(x_np[None]).to(device)  # (1, 3, H, W): planes (axial, coronal, sagittal)

with torch.no_grad():
    prob = torch.softmax(model(x), dim=1)[0, 1].item()

st.metric("P(SuperAger)", f"{prob:.3f}", delta=f"decision threshold {threshold:.2f}")
st.progress(prob)

st.subheader("Attention rollout")
cols = st.columns(len(PLANES))
saliency = {}
for col, (plane, plane_idx) in zip(cols, PLANES.items()):
    with col:
        saliency[plane] = attention_rollout(model, x, plane=plane_idx)
        show_map(x_np[plane_idx], saliency[plane], plane)

st.subheader("Regional analysis (axial)")
scores = region_scores(saliency["axial"])
table = pd.DataFrame(
    [{"Region": r, "I_R": scores.get(r, float("nan")),
      "S_R": scores.get(r, float("nan")) * prob} for r in REGIONS]
).sort_values("I_R", ascending=False)
st.dataframe(table.style.format({"I_R": "{:.3f}", "S_R": "{:.3f}"}),
             hide_index=True, use_container_width=True)

st.caption(
    "The 6x6 grid is a coarse positional proxy, not an atlas registration. "
    "Region labels assume a canonical RAS orientation."
)
