"""Cohort assembly and 2.5D multi-plane preprocessing (paper Sec. 3.1-3.2).

Preprocessing stages, in order (the order matters -- per-site z-scoring can
reintroduce extreme values, so the second percentile clip must follow it):

    1. reorient to canonical RAS
    2. strict brain mask: 15%-of-max threshold, hole-fill, largest component
    3. robust intensity normalisation by median / IQR inside the mask
    4. optional histogram matching to a reference pooled from n_ref_scans
    5. per-scan [1, 99] percentile clip, rescaled to [0, 1]
    6. per-site z-score, then a SECOND global [1, 99] clip

Output is one (3, size, size) tensor per subject, channels ordered
(axial, coronal, sagittal).
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import ndimage

from .config import SEED, Config

log = logging.getLogger(__name__)

AXIAL, CORONAL, SAGITTAL = 0, 1, 2

ADNI_RE = re.compile(r"(\d{3})_S_(\d{4})")
OASIS_RE = re.compile(r"(OAS\d+[_-]\d+[_-]MR\d+|OAS2[_-]\d{4})", re.IGNORECASE)

POSITIVE = {"1", "superager", "sa", "super", "super_ager"}
NEGATIVE = {"0", "normal", "cn", "control", "typical", "nondemented", "normal_ager"}


# ============================================================ COHORT (Sec. 3.1)
def subject_id(path) -> str:
    """ADNI `003_S_1234` or OASIS `OAS2_0045`, else the filename stem."""
    s = str(path)
    m = ADNI_RE.search(s)
    if m:
        return f"{m.group(1)}_S_{m.group(2)}"
    m = OASIS_RE.search(s)
    return m.group(1).replace("-", "_").upper() if m else Path(path).stem.upper()


def site_of(path) -> str:
    """ADNI encodes acquisition site in the subject ID; OASIS is one pooled site."""
    m = ADNI_RE.search(str(path))
    return f"ADNI_{m.group(1)}" if m else "OASIS"


def cohort_of(path) -> str:
    return "ADNI" if ADNI_RE.search(str(path)) else "OASIS"


def _to_label(value) -> int:
    v = str(value).strip().lower()
    if v in POSITIVE:
        return 1
    if v in NEGATIVE:
        return 0
    try:
        return int(float(v))
    except ValueError:
        return -1


def find_nifti_files(roots: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        p = Path(root)
        if not p.exists():
            log.warning("Image root does not exist: %s", p)
            continue
        for pattern in ("*.nii", "*.nii.gz"):
            files.extend(f for f in p.rglob(pattern) if not f.name.startswith("._"))
    return sorted(files)


def collect_records(labels_csv: str, image_roots: Sequence[str]) -> pd.DataFrame:
    """Join the label CSV to NIfTI paths, keeping ONE scan per subject.

    Returns a frame with columns: path, subject, label, site, cohort, [age].
    Keeping one scan per subject matters because OASIS-2 is longitudinal;
    repeat sessions would place the same brain on both sides of a split.
    """
    csv_path = Path(labels_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Labels CSV not found: {csv_path}")

    lab = pd.read_csv(csv_path)
    lab.columns = [c.strip().lower() for c in lab.columns]

    subj_col = next((c for c in lab.columns if c in
                     ("subject", "subject_id", "subjectid", "id", "participant_id")), None)
    label_col = next((c for c in lab.columns if c in
                      ("label", "diagnosis", "group", "class", "dx")), None)
    age_col = next((c for c in lab.columns if c in ("age", "age_at_scan")), None)
    if subj_col is None or label_col is None:
        raise ValueError(f"Need subject and label columns; found {list(lab.columns)}")

    lab["_y"] = lab[label_col].map(_to_label)
    dropped = int((lab["_y"] < 0).sum())
    if dropped:
        log.warning("Dropped %d rows with unrecognised labels.", dropped)
    lab = lab[lab["_y"] >= 0]

    niftis = find_nifti_files(image_roots)
    if not niftis:
        raise RuntimeError(f"No NIfTI files found under {list(image_roots)}")

    by_subject: dict[str, list[Path]] = {}
    for f in niftis:
        by_subject.setdefault(subject_id(f).upper(), []).append(f)

    rows, unmatched = [], []
    for _, r in lab.iterrows():
        sid = str(r[subj_col]).strip().replace("-", "_").replace(".", "_").upper()
        paths = by_subject.get(sid)
        if paths is None:  # ID formatting drift between CSV and filenames
            hit = next((k for k in by_subject if sid in k or k in sid), None)
            paths = by_subject.get(hit) if hit else None
        if not paths:
            unmatched.append(sid)
            continue
        rec = dict(path=paths[0], subject=subject_id(paths[0]).upper(),
                   label=int(r["_y"]), site=site_of(paths[0]),
                   cohort=cohort_of(paths[0]))
        if age_col and pd.notna(r.get(age_col)):
            rec["age"] = float(r[age_col])
        rows.append(rec)

    if unmatched:
        log.warning("%d labelled subjects had no matching NIfTI (e.g. %s)",
                    len(unmatched), unmatched[:5])
    if not rows:
        raise RuntimeError("No NIfTI files matched any labelled subject.")

    df = (pd.DataFrame(rows)
          .drop_duplicates(subset="subject", keep="first")
          .sort_values("subject")
          .reset_index(drop=True))

    log.info("Cohort: %d scans | %d typical / %d SuperAger | %d sites | %s",
             len(df), int((df.label == 0).sum()), int((df.label == 1).sum()),
             df.site.nunique(), df.cohort.value_counts().to_dict())
    return df


def audit_site_label(df: pd.DataFrame) -> pd.DataFrame:
    """Report sites containing only one class -- there, site *is* the label."""
    table = pd.crosstab(df["site"], df["label"])
    pure = [s for s in table.index if (table.loc[s] > 0).sum() == 1]
    if pure:
        log.warning("%d/%d sites are single-class (e.g. %s). Site-aware "
                    "stratification mitigates but cannot remove this confound.",
                    len(pure), len(table), pure[:5])
    return table


# ============================================================ PREPROCESSING (Sec. 3.2)
def brain_mask(volume: np.ndarray, threshold: float = 0.15) -> np.ndarray:
    """15%-of-max threshold, morphological hole-filling, largest component."""
    if volume.max() <= 0:
        return np.ones_like(volume, dtype=bool)
    mask = ndimage.binary_fill_holes(volume > threshold * float(volume.max()))
    labelled, n = ndimage.label(mask)
    if n > 1:  # keep the brain, drop skull and neck fragments
        sizes = ndimage.sum(mask, labelled, range(1, n + 1))
        mask = labelled == int(np.argmax(sizes)) + 1
    return np.asarray(mask, dtype=bool)


def load_volume(path) -> np.ndarray:
    """Load a NIfTI reoriented to canonical RAS.

    After reorientation the axes are (L->R, P->A, I->S), so vol[:, :, k] is
    axial, vol[:, j, :] is coronal, and vol[i, :, :] is sagittal. Every
    downstream region label depends on this.
    """
    volume = np.squeeze(nib.as_closest_canonical(nib.load(str(path))).get_fdata())
    while volume.ndim > 3:
        volume = volume[..., 0]
    return volume.astype(np.float32)


def resize_2d(slice_2d: np.ndarray, size: int) -> np.ndarray:
    t = torch.from_numpy(np.ascontiguousarray(slice_2d))[None, None].float()
    return F.interpolate(t, (size, size), mode="bilinear", align_corners=False)[0, 0].numpy()


def resize_3d(volume: np.ndarray, size: int) -> np.ndarray:
    t = torch.from_numpy(np.ascontiguousarray(volume))[None, None].float()
    return F.interpolate(t, (size,) * 3, mode="trilinear", align_corners=False)[0, 0].numpy()


def build_histogram_reference(paths: Sequence, size: int, n_ref: int = 20,
                              n_voxels: int = 20000,
                              threshold: float = 0.15) -> np.ndarray | None:
    """Pool robust-normalised intensities from n_ref scans into a sorted reference."""
    rng = np.random.RandomState(SEED)
    pool = []
    for i in rng.choice(len(paths), size=min(n_ref, len(paths)), replace=False):
        vol = resize_3d(load_volume(paths[i]), size)
        values = vol[brain_mask(vol, threshold)]
        if values.size:
            iqr = np.subtract(*np.percentile(values, [75, 25])) or 1.0
            pool.append((values - np.median(values)) / iqr)
    if not pool:
        return None
    pool = np.concatenate(pool)
    return np.sort(rng.choice(pool, size=min(n_voxels, len(pool)), replace=False))


def normalise_volume(volume: np.ndarray, reference: np.ndarray | None = None,
                     threshold: float = 0.15) -> np.ndarray:
    """Stages 2-5: mask, median/IQR, optional histogram match, clip to [0, 1]."""
    volume = volume.astype(np.float32)
    mask = brain_mask(volume, threshold)
    if mask.sum() < 10:
        mask = np.ones_like(volume, dtype=bool)

    values = volume[mask]
    iqr = np.subtract(*np.percentile(values, [75, 25])) or 1.0
    out = (volume - np.median(values)) / iqr

    if reference is not None:  # rank-based histogram matching inside the mask
        inside = out[mask]
        quantiles = inside.argsort().argsort().astype(np.float32) / max(1, inside.size - 1)
        out[mask] = np.interp(quantiles, np.linspace(0, 1, len(reference)),
                              reference).astype(np.float32)

    lo, hi = np.percentile(out[mask], [1, 99])
    hi = hi if hi > lo else lo + 1.0
    return np.clip((out - lo) / (hi - lo), 0, 1).astype(np.float32)


def per_site_normalise(X: np.ndarray, sites: np.ndarray) -> np.ndarray:
    """Stage 6: per-site z-score, then a SECOND global [1, 99] clip.

    The clip has to run after z-scoring: z-scoring pushes a few voxels into the
    tails, and clipping only beforehand leaves scanner-driven outliers in place.
    """
    out = X.copy()
    for site in np.unique(sites):
        idx = np.where(sites == site)[0]
        if len(idx) < 3:  # too few scans for a stable site mean
            continue
        out[idx] = (out[idx] - out[idx].mean()) / (out[idx].std() + 1e-6)
    lo, hi = np.percentile(out, 1), np.percentile(out, 99)
    hi = hi if hi > lo else lo + 1.0
    return np.clip((out - lo) / (hi - lo), 0, 1).astype(np.float32)


def extract_slice(volume: np.ndarray, plane: str, fraction: float = 0.5) -> np.ndarray:
    """One 2D slice at a fractional depth. Canonical RAS assumed."""
    if plane == "axial":
        k = int(round(fraction * (volume.shape[2] - 1)))
        return volume[:, :, k]
    if plane == "coronal":
        k = int(round(fraction * (volume.shape[1] - 1)))
        return volume[:, k, :]
    if plane == "sagittal":
        k = int(round(fraction * (volume.shape[0] - 1)))
        return volume[k, :, :]
    raise ValueError(f"Unknown plane {plane!r}; use axial|coronal|sagittal")


def triplanar(volume: np.ndarray, size: int = 224,
              fractions: dict | None = None) -> np.ndarray:
    """Canonical-RAS mid-slices -> (3, size, size) as (axial, coronal, sagittal)."""
    f = {"axial": 0.5, "coronal": 0.5, "sagittal": 0.5}
    if fractions:
        f.update(fractions)
    return np.stack([resize_2d(extract_slice(volume, p, f[p]), size)
                     for p in ("axial", "coronal", "sagittal")]).astype(np.float32)


def _cache_key(df: pd.DataFrame, cfg: Config) -> str:
    signature = (f"{len(df)}|{cfg.size_2d}|{cfg.harmonize}|{cfg.ref_size}|"
                 f"{cfg.mask_threshold}|" + "|".join(map(str, df.path)))
    return hashlib.md5(signature.encode()).hexdigest()[:12]


def preprocess(df: pd.DataFrame, cfg: Config) -> np.ndarray:
    """Full 2.5D pipeline over a cohort, with an on-disk cache."""
    cache_path = Path(cfg.outdir) / "cache" / f"X_{_cache_key(df, cfg)}.npy"
    if cfg.cache and cache_path.exists():
        log.info("Loading cached preprocessed array: %s", cache_path)
        return np.load(cache_path)

    reference = None
    if cfg.harmonize:
        log.info("Building histogram reference from %d scans ...", cfg.n_ref_scans)
        reference = build_histogram_reference(df.path.tolist(), cfg.ref_size,
                                              cfg.n_ref_scans,
                                              threshold=cfg.mask_threshold)

    log.info("Preprocessing %d scans ...", len(df))
    X = np.stack([
        triplanar(normalise_volume(load_volume(p), reference, cfg.mask_threshold),
                  cfg.size_2d)
        for p in df.path
    ])

    if cfg.harmonize:
        X = per_site_normalise(X, df.site.to_numpy())

    if cfg.cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, X)
        log.info("Cached preprocessed array -> %s", cache_path)
    return X
