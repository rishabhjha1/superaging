"""Streamlit demo for the SuperAger ViT-B/16 explainability pipeline.

Local:   streamlit run app_streamlit.py
Deploy:  share.streamlit.io -> repo, branch main, main file app_streamlit.py

Only the four imports in the WIRE block below are project-specific. If a name
or signature does not match your package, fix it there and nothing else in this
file needs to change.
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

# ---------------------------------------------------------------- WIRE ----
# 1. build_vit()             -> nn.Module (frozen backbone + your head)
# 2. volume_to_triplanar(v)  -> np.ndarray (3, 224, 224) float32 in [0, 1]
# 3. attention_rollout(m, x) -> np.ndarray (224, 224) float32, one plane
# 4. region_scores(saliency) -> dict {region: float}
from superager.models.vit import build_vit
from superager.data.preprocess import volume_to_triplanar
from superager.explain.rollout import attention_rollout
from superager.regions import region_scores, REGIONS
# --------------------------------------------------------------------------

CKPT = Path(os.environ.get("SUPERAGER_CKPT", "checkpoints/vit_seed42.pt"))
PLANES = ("axial", "coronal", "sagittal")

st.set_page_config(page_title="SuperAger explainability demo", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    """Build the ViT and load the checkpoint. Returns (model, is_trained)."""
    model = build_vit()
    if CKPT.exists():
        state = torch.load(CKPT, map_location="cpu")
        state = state.get("model", state.get("state_dict", state))
        model.load_state_dict(state, strict=False)
        trained = True
    else:
        trained = False  # smoke-test path: plumbing runs, numbers are meaningless
    model.eval()
    return model, trained


@st.cache_data(show_spinner="Preprocessing volume...")
def preprocess(raw: bytes, name: str) -> np.ndarray:
    """Bytes from the uploader -> (3, 224, 224) float32."""
    suffix = ".nii.gz" if name.endswith(".gz") else ".nii"
    tmp = Path("/tmp") / f"upload{suffix}"
    tmp.write_bytes(raw)
    vol = nib.as_closest_canonical(nib.load(str(tmp)))
    return volume_to_triplanar(np.asanyarray(vol.dataobj).astype(np.float32))


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

model, trained = load_model()
if not trained:
    st.warning(
        f"No checkpoint at {CKPT} — running with an untrained head. "
        "The pipeline executes end to end but the outputs are not meaningful."
    )

upload = st.file_uploader("T1-weighted MRI", type=["nii", "gz"])
if upload is None:
    st.info("Waiting for a .nii or .nii.gz file.")
    st.stop()

x = preprocess(upload.getvalue(), upload.name)
tensor = torch.from_numpy(x).unsqueeze(1).repeat(1, 3, 1, 1)  # (3 views, 3, H, W)

with torch.no_grad():
    logits = model(tensor)
    if logits.ndim == 2 and logits.shape[0] == len(PLANES):
        logits = logits.mean(0, keepdim=True)  # mean fusion across views
    prob = torch.softmax(logits, dim=-1)[0, 1].item()

st.metric("P(SuperAger)", f"{prob:.3f}")
st.progress(prob)

st.subheader("Attention rollout")
cols = st.columns(len(PLANES))
saliency = {}
for col, plane, view in zip(cols, PLANES, tensor):
    with col:
        saliency[plane] = attention_rollout(model, view.unsqueeze(0))
        show_map(x[PLANES.index(plane)], saliency[plane], plane)

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
