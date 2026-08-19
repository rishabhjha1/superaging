"""
SuperAger vs typical-ager classification from T1w structural MRI.

Five models under one 2.5D preprocessing pipeline:
    LogReg | SimpleCNN | ResNet-18 | DenseNet-40 | ViT-B/16 (proposed)

Usage:
    python train.py --labels data/labels.csv --roots data/ADNI data/OASIS
"""

import argparse, math, random, re, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from scipy import ndimage
from scipy.stats import ttest_rel, wilcoxon
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression as SKLogReg
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")
SEED, SIZE = 42, 224
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = DEV.type == "cuda"

# ---- hyperparameters (paper Sec. 3.3-3.4) ----
CFG = dict(
    seeds=[42, 1, 2, 3, 4], n_splits=5, batch=16, warmup=3, patience=8,
    lr_head=1e-4, lr_scratch=8e-4, wd=1e-4, vit_wd=5e-5, label_smooth=0.05,
    epochs_scratch=60, epochs_finetune=30, min_lr_ratio=0.05,
)


def set_seed(s):
    np.random.seed(s); random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ============================================================ DATA
ADNI_RE = re.compile(r"(\d{3})_S_(\d{4})")
OASIS_RE = re.compile(r"(OAS\d+[_-]\d+[_-]MR\d+|OAS2[_-]\d{4})", re.I)
POS = {"1", "superager", "sa", "super", "super_ager"}
NEG = {"0", "normal", "cn", "control", "typical", "nondemented"}


def subject_id(p):
    m = ADNI_RE.search(str(p))
    if m:
        return f"{m.group(1)}_S_{m.group(2)}"
    m = OASIS_RE.search(str(p))
    return m.group(1).replace("-", "_").upper() if m else Path(p).stem


def site_of(p):
    m = ADNI_RE.search(str(p))
    return f"ADNI_{m.group(1)}" if m else "OASIS"


def collect_records(labels_csv, roots):
    """Join the label CSV to NIfTI paths. One scan per subject (no leakage)."""
    lab = pd.read_csv(labels_csv)
    lab.columns = [c.strip().lower() for c in lab.columns]
    sc = next(c for c in lab.columns if c in ("subject", "subject_id", "id"))
    lc = next(c for c in lab.columns if c in ("label", "group", "diagnosis"))

    def to_int(v):
        v = str(v).strip().lower()
        return 1 if v in POS else 0 if v in NEG else -1

    lab["y"] = lab[lc].map(to_int)
    lab = lab[lab.y >= 0]

    files = []
    for r in roots:
        for pat in ("*.nii", "*.nii.gz"):
            files += [f for f in Path(r).rglob(pat) if not f.name.startswith("._")]

    by_sub = {}
    for f in sorted(files):
        by_sub.setdefault(subject_id(f).upper(), []).append(f)

    rows = []
    for _, r in lab.iterrows():
        sid = str(r[sc]).strip().replace("-", "_").upper()
        paths = by_sub.get(sid) or next(
            (v for k, v in by_sub.items() if sid in k or k in sid), None)
        if paths:
            rows.append(dict(path=paths[0], subject=sid, label=int(r.y),
                             site=site_of(paths[0])))

    df = pd.DataFrame(rows).drop_duplicates("subject").reset_index(drop=True)
    if df.empty:
        raise RuntimeError("No NIfTI files matched any labelled subject.")
    print(f"{len(df)} scans | {(df.label == 0).sum()} typical / "
          f"{(df.label == 1).sum()} SuperAger | {df.site.nunique()} sites")

    ct = pd.crosstab(df.site, df.label)
    pure = [s for s in ct.index if (ct.loc[s] > 0).sum() == 1]
    if pure:
        print(f"WARNING: {len(pure)}/{len(ct)} sites are single-class -- "
              f"scanner signature could act as a label.")
    return df


# ============================================================ PREPROCESSING (Sec. 3.2)
def brain_mask(d, thr=0.15):
    """15%-of-max threshold, hole-fill, largest connected component."""
    if d.max() <= 0:
        return np.ones_like(d, bool)
    m = ndimage.binary_fill_holes(d > thr * float(d.max()))
    lab, n = ndimage.label(m)
    if n > 1:
        m = lab == int(np.argmax(ndimage.sum(m, lab, range(1, n + 1)))) + 1
    return m.astype(bool)


def load_volume(path):
    """Load a NIfTI reoriented to canonical RAS: axes are (L->R, P->A, I->S)."""
    v = np.squeeze(nib.as_closest_canonical(nib.load(str(path))).get_fdata())
    while v.ndim > 3:
        v = v[..., 0]
    return v.astype(np.float32)


def normalise(v):
    """Robust median/IQR within the brain mask, then [1,99] clip to [0,1]."""
    m = brain_mask(v)
    if m.sum() < 10:
        m = np.ones_like(v, bool)
    x = v[m]
    iqr = np.subtract(*np.percentile(x, [75, 25])) or 1.0
    out = (v - np.median(x)) / iqr
    lo, hi = np.percentile(out[m], [1, 99])
    return np.clip((out - lo) / (max(hi, lo + 1.0) - lo), 0, 1).astype(np.float32)


def resize2d(s, size=SIZE):
    t = torch.from_numpy(np.ascontiguousarray(s))[None, None]
    return F.interpolate(t, (size, size), mode="bilinear", align_corners=False)[0, 0].numpy()


def triplanar(v, size=SIZE):
    """Canonical-RAS mid-slices -> (3, size, size) as (axial, coronal, sagittal)."""
    return np.stack([
        resize2d(v[:, :, v.shape[2] // 2], size),   # fix S -> axial
        resize2d(v[:, v.shape[1] // 2, :], size),   # fix A -> coronal
        resize2d(v[v.shape[0] // 2, :, :], size),   # fix R -> sagittal
    ]).astype(np.float32)


def per_site_norm(X, sites):
    """Per-site z-score, then a SECOND global clip (z-scoring reintroduces tails)."""
    X = X.copy()
    for s in np.unique(sites):
        i = np.where(sites == s)[0]
        if len(i) >= 3:
            X[i] = (X[i] - X[i].mean()) / (X[i].std() + 1e-6)
    lo, hi = np.percentile(X, 1), np.percentile(X, 99)
    return np.clip((X - lo) / (max(hi, lo + 1.0) - lo), 0, 1).astype(np.float32)


def preprocess(df, cache=None):
    if cache and Path(cache).exists():
        print(f"Loading cached array: {cache}")
        return np.load(cache)
    print(f"Preprocessing {len(df)} scans ...")
    X = np.stack([triplanar(normalise(load_volume(p))) for p in df.path])
    X = per_site_norm(X, df.site.to_numpy())
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, X)
    return X


# ============================================================ SPLITS (Sec. 3.4)
def make_splits(y, groups, sites, n_splits=5, seed=SEED):
    """Subject-grouped, site-aware nested splits: outer test fold + inner K-fold CV.

    The stratification key is (label, site), not label alone -- a site-pure fold
    turns a scanner signature into a free label. Falls back to label-only when
    sites are too small to stratify on, and says so.
    """
    def key(yy, ss):
        # multiplier 1000, not 100: ADNI can exceed 100 sites and a smaller
        # multiplier collides label 0/site 100 with label 1/site 0.
        return yy * 1000 + pd.factorize(ss)[0]

    def ok(k, g):
        return np.unique(k, return_counts=True)[1].min() >= n_splits and \
            len(np.unique(g)) >= n_splits

    def resolve(yy, gg, ss):
        k = key(yy, ss)
        if ok(k, gg):
            return k
        merged = ss.astype(object).copy()
        u, c = np.unique(ss, return_counts=True)
        merged[np.isin(ss, [s for s, n in zip(u, c) if n < n_splits])] = "_SMALL_"
        k = key(yy, merged)
        if ok(k, gg):
            return k
        print("  note: site-aware stratification infeasible; using label only.")
        return yy.copy()

    outer = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
    dev, test = next(iter(outer.split(np.zeros(len(y)), resolve(y, groups, sites), groups)))

    inner = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed + 1)
    ks = resolve(y[dev], groups[dev], sites[dev])
    folds = [(dev[tr], dev[va]) for tr, va in
             inner.split(np.zeros(len(dev)), ks, groups[dev])]

    for a, b in [(test, dev)] + folds:
        if set(groups[a]) & set(groups[b]):
            raise RuntimeError("Subject leakage across partitions.")
    return test, folds


# ============================================================ MODELS (Sec. 3.3)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to_imagenet(x):
    x3 = x.repeat(1, 3, 1, 1)
    return (x3 - MEAN.to(x.device)) / STD.to(x.device)


class LogReg:
    """Linear reference: PCA-50 on the flattened, standardised axial slice."""
    pretrained = False

    def __init__(self):
        self.sc, self.pca = StandardScaler(), None
        self.clf = SKLogReg(C=0.1, class_weight="balanced", max_iter=2000)

    def fit(self, X, y):
        z = self.sc.fit_transform(X[:, 0].reshape(len(X), -1))
        self.pca = PCA(min(50, len(X) - 1), random_state=SEED)
        self.clf.fit(self.pca.fit_transform(z), y)

    def predict_proba(self, X):
        z = self.sc.transform(X[:, 0].reshape(len(X), -1))
        return self.clf.predict_proba(self.pca.transform(z))[:, 1]


class SimpleCNN(nn.Module):
    """Two-block from-scratch CNN, applied per plane, features concatenated."""
    pretrained = False

    def __init__(self):
        super().__init__()
        blk = lambda i, o: nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o),
            nn.ReLU(), nn.MaxPool2d(2))
        self.enc = nn.Sequential(blk(1, 16), blk(16, 32))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(96, 64),
            nn.ReLU(), nn.Dropout(0.40), nn.Linear(64, 2))

    def forward(self, x):
        return self.head(torch.cat([self.enc(x[:, i:i + 1]) for i in range(3)], 1))


class ResNet18Net(nn.Module):
    """Frozen ImageNet ResNet-18; per-plane features mean-pooled."""
    pretrained = True

    def __init__(self):
        super().__init__()
        rn = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        fd, rn.fc = rn.fc.in_features, nn.Identity()
        for p in rn.parameters():
            p.requires_grad = False
        self.backbone = rn
        self.head = nn.Sequential(nn.LayerNorm(fd), nn.Dropout(0.50), nn.Linear(fd, 2))

    def forward(self, x):
        B = x.shape[0]
        f = self.backbone(to_imagenet(x.reshape(B * 3, 1, *x.shape[2:])))
        return self.head(f.reshape(B, 3, -1).mean(1))


class ViTNet(nn.Module):
    """PROPOSED: frozen ViT-B/16 + parameter-free mean fusion + lightweight head.

    Small-data adaptations (Sec. 3.3): frozen backbone (86M -> 0.6M trainable),
    head-only optimisation, mild augmentation, mean fusion, and 4-view TTA.
    """
    pretrained = True

    def __init__(self):
        super().__init__()
        vit = torchvision.models.vit_b_16(
            weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1)
        fd, vit.heads = vit.heads.head.in_features, nn.Identity()
        for p in vit.parameters():
            p.requires_grad = False
        self.backbone = vit
        self.head = nn.Sequential(
            nn.LayerNorm(fd), nn.Linear(fd, 256), nn.GELU(),
            nn.Dropout(0.20), nn.Linear(256, 2))

    def forward(self, x):
        B = x.shape[0]
        f = self.backbone(to_imagenet(x.reshape(B * 3, 1, *x.shape[2:])))
        return self.head(f.reshape(B, 3, -1).mean(1))


class DenseNet40(nn.Module):
    """Compact from-scratch DenseNet (depth 40, k=12, ~1.02M params) at 96x96."""
    pretrained = False

    def __init__(self, k=12, depth=40, drop=0.10):
        super().__init__()
        n = (depth - 4) // 3
        self.stem, ch = nn.Conv2d(3, 16, 3, padding=1, bias=False), 16
        blocks = []
        for b in range(3):
            layers = []
            for _ in range(n):
                layers.append(nn.ModuleDict(dict(
                    bn=nn.BatchNorm2d(ch),
                    conv=nn.Conv2d(ch, k, 3, padding=1, bias=False))))
                ch += k
            blocks.append(nn.ModuleList(layers))
            if b < 2:
                blocks.append(nn.ModuleDict(dict(
                    bn=nn.BatchNorm2d(ch), conv=nn.Conv2d(ch, ch, 1, bias=False))))
        self.stages = nn.ModuleList(blocks)
        self.bn = nn.BatchNorm2d(ch)
        self.head = nn.Sequential(nn.Dropout(0.30), nn.Linear(ch, 2))
        self.drop = drop

    def forward(self, x):
        x = self.stem(F.interpolate(x, (96, 96), mode="bilinear", align_corners=False))
        for stage in self.stages:
            if isinstance(stage, nn.ModuleList):          # dense block
                for lyr in stage:
                    o = lyr["conv"](F.relu(lyr["bn"](x)))
                    if self.drop:
                        o = F.dropout(o, self.drop, self.training)
                    x = torch.cat([x, o], 1)
            else:                                          # transition
                x = F.avg_pool2d(stage["conv"](F.relu(stage["bn"](x))), 2)
        x = F.adaptive_avg_pool2d(F.relu(self.bn(x)), 1).flatten(1)
        return self.head(x)


# ============================================================ TRAINING (Sec. 3.4)
def zoom(x, s):
    th = torch.tensor([[1 / s, 0, 0], [0, 1 / s, 0]], dtype=x.dtype,
                      device=x.device).unsqueeze(0).repeat(x.shape[0], 1, 1)
    return F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False),
                         align_corners=False, padding_mode="border")


def augment(x, strong):
    """The ViT skips affine/gamma: pretrained features dislike geometric distortion."""
    if random.random() < 0.5:
        x = torch.flip(x, [-1])
    if strong and random.random() < 0.4:
        x = zoom(x, 0.95 + 0.10 * random.random())
    if strong and random.random() < 0.25:
        x = x.clamp(0, 1) ** (0.9 + 0.2 * random.random())
    if random.random() < 0.25:
        x = x * (0.95 + 0.10 * torch.rand(1, device=x.device))
    return (x + 0.01 * torch.randn_like(x)).clamp(0, 1)


@torch.no_grad()
def predict(model, X, idx, tta=False):
    model.eval()
    out = []
    for b in range(0, len(idx), 16):
        x = torch.from_numpy(X[idx[b:b + 16]]).float().to(DEV)
        with torch.autocast(DEV.type, enabled=AMP):
            views = [x, torch.flip(x, [-1]), zoom(x, 1.05), zoom(x, 0.95)] if tta else [x]
            p = torch.stack([torch.softmax(model(v), 1)[:, 1] for v in views]).mean(0)
        out.append(p.float().cpu().numpy())
    return np.concatenate(out)


def pick_threshold(y, p):
    """Chosen on pooled VALIDATION predictions, never on test.

    Adopted only if it beats the default 0.5 by more than two points, so a
    marginal noise-driven threshold does not get locked in.
    """
    base = balanced_accuracy_score(y, p >= 0.5)
    cands = [(balanced_accuracy_score(y, p >= t), t) for t in np.linspace(0.2, 0.8, 31)]
    best, t = max(cands)
    return float(t) if best - base > 0.02 else 0.5


def train_fold(cls, X, y, tr, va, epochs):
    model = cls().to(DEV)
    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y[tr])
    crit = nn.CrossEntropyLoss(weight=torch.tensor(cw).float().to(DEV),
                              label_smoothing=CFG["label_smooth"])
    params = [p for p in model.parameters() if p.requires_grad]
    lr = CFG["lr_head"] if model.pretrained else CFG["lr_scratch"]
    wd = CFG["vit_wd"] if isinstance(model, ViTNet) else CFG["wd"]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    scaler = torch.amp.GradScaler(DEV.type, enabled=AMP)
    strong = not isinstance(model, ViTNet)

    best = dict(score=-1, state=None, auc=float("nan"), vprob=None)
    stale = 0
    for ep in range(epochs):
        w, mr = CFG["warmup"], CFG["min_lr_ratio"]
        f = (ep + 1) / w if ep < w else \
            mr + 0.5 * (1 - mr) * (1 + math.cos(math.pi * (ep - w) / max(1, epochs - w)))
        for g in opt.param_groups:
            g["lr"] = lr * f

        model.train()
        for b in range(0, len(tr), CFG["batch"]):
            j = np.random.permutation(tr)[b:b + CFG["batch"]]
            x = augment(torch.from_numpy(X[j]).float().to(DEV), strong)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(DEV.type, enabled=AMP):
                loss = crit(model(x), torch.from_numpy(y[j]).long().to(DEV))
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()

        p = predict(model, X, va)
        auc = roc_auc_score(y[va], p) if len(set(y[va])) > 1 else 0.0
        pred = (p >= 0.5).astype(int)
        score = 0.5 * auc + 0.5 * balanced_accuracy_score(y[va], pred)
        # a degenerate single-class epoch can score well at n=200; exclude it
        if len(set(pred)) > 1 and score > best["score"]:
            best.update(score=score, auc=auc, vprob=p.copy(),
                        state={k: v.detach().cpu().clone()
                               for k, v in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
        if stale >= CFG["patience"]:
            break

    del model; torch.cuda.empty_cache()
    return best


def run_model(name, cls, X, y, test, folds, outdir, seed):
    """Fold-ensembled evaluation: test probs are the mean over inner-fold checkpoints."""
    probs, vt, vp, aucs = [], [], [], []
    epochs = CFG["epochs_finetune"] if cls.pretrained else CFG["epochs_scratch"]

    for i, (tr, va) in enumerate(folds):
        if cls is LogReg:
            m = LogReg(); m.fit(X[tr], y[tr])
            p = m.predict_proba(X[va])
            probs.append(m.predict_proba(X[test]))
            auc = roc_auc_score(y[va], p) if len(set(y[va])) > 1 else float("nan")
            ch = dict(vprob=p, auc=auc, state=None)
        else:
            ch = train_fold(cls, X, y, tr, va, epochs)
            m = cls().to(DEV); m.load_state_dict(ch["state"])
            probs.append(predict(m, X, test, tta=True))
            if cls is ViTNet and i == 0:
                torch.save({"state_dict": ch["state"], "seed": seed},
                           Path(outdir) / f"vit_seed{seed}.pt")
            del m; torch.cuda.empty_cache()
        aucs.append(ch["auc"]); vt.append(y[va]); vp.append(ch["vprob"])
        print(f"    fold {i+1}: val AUC {ch['auc']:.3f}")

    ens = np.mean(probs, 0)
    thr = pick_threshold(np.concatenate(vt), np.concatenate(vp))
    pred = (ens >= thr).astype(int)
    return dict(auc=roc_auc_score(y[test], ens),
                auprc=average_precision_score(y[test], ens),
                bal=balanced_accuracy_score(y[test], pred),
                f1=f1_score(y[test], pred, zero_division=0),
                acc=accuracy_score(y[test], pred))


# ============================================================ SIGNIFICANCE (Sec. 4.1)
def significance(agg, names, ref="ViT", metric="auc", alpha=0.05):
    """Paired tests on per-seed held-out metrics. Every model sees the same seeds,
    so agg[m][metric][i] pairs with agg[ref][metric][i] by seed index.

    NOTE: at n=5 seeds the exact signed-rank test cannot fall below p=0.0625,
    so it is a directional check. The paired t-test is primary.
    """
    a = np.array(agg[ref][metric], float)
    rows = []
    print(f"\nPAIRED SIGNIFICANCE - {ref} vs baselines on {metric} (n={len(a)} seeds)")
    print(f"{'baseline':<12}{'mean d':>9}{'t p':>9}{'Wilcox p':>11}")
    for other in [n for n in names if n != ref]:
        b = np.array(agg[other][metric], float)
        d = a - b
        t_p = float(ttest_rel(a, b).pvalue)
        w_p = 1.0 if np.allclose(d, 0) else float(
            wilcoxon(a, b, alternative="two-sided").pvalue)
        print(f"{other:<12}{d.mean():>+9.3f}{t_p:>9.3f}{w_p:>11.3f}")
        rows.append(dict(baseline=other, mean_diff=d.mean(), t_p=t_p, wilcoxon_p=w_p))

    # Holm-Bonferroni across the baseline comparisons
    p = np.array([r["t_p"] for r in rows])
    run = 0.0
    for rank, i in enumerate(np.argsort(p)):
        run = max(run, min(1.0, p[i] * (len(p) - rank)))
        rows[i]["t_p_holm"] = run
    print("\nHolm-corrected:")
    for r in rows:
        print(f"  {ref} vs {r['baseline']:<11} {r['t_p']:.3f} -> {r['t_p_holm']:.3f} "
              f"({'sig' if r['t_p_holm'] < alpha else 'ns'})")
    return pd.DataFrame(rows)


# ============================================================ MAIN
MODELS = [("LogReg", LogReg), ("SimpleCNN", SimpleCNN), ("ResNet18", ResNet18Net),
          ("DenseNet40", DenseNet40), ("ViT", ViTNet)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--roots", nargs="+", default=["data/ADNI", "data/OASIS"])
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--seeds", type=int, nargs="+", default=CFG["seeds"])
    args = ap.parse_args()

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEV}\n")

    df = collect_records(args.labels, args.roots)
    X = preprocess(df, cache=out / "X.npy")
    y, groups, sites = (df.label.to_numpy(), df.subject.to_numpy(), df.site.to_numpy())

    names = [n for n, _ in MODELS]
    keys = ("auc", "auprc", "bal", "f1", "acc")
    agg = {n: {k: [] for k in keys} for n in names}

    for seed in args.seeds:
        print(f"\n{'#' * 60}\n# SEED {seed}\n{'#' * 60}")
        set_seed(seed)
        test, folds = make_splits(y, groups, sites, CFG["n_splits"], seed)
        print(f"held-out test: {len(test)} scans ({(y[test] == 1).sum()} SuperAger)")
        for name, cls in MODELS:
            print(f"  --- {name} ---")
            r = run_model(name, cls, X, y, test, folds, out, seed)
            print(f"  => AUC {r['auc']:.3f} | AUPRC {r['auprc']:.3f} | "
                  f"BalAcc {r['bal']:.3f}")
            for k in keys:
                agg[name][k].append(r[k])

    print(f"\n{'=' * 72}\nRESULTS - held-out test, mean +/- std over "
          f"{len(args.seeds)} seeds\n{'=' * 72}")
    print(f"{'Model':<12}{'AUC':>16}{'BalAcc':>16}{'AUPRC':>16}{'Acc':>12}")
    rows = []
    for n in sorted(names, key=lambda n: -np.mean(agg[n]["auc"])):
        m = {k: (np.mean(agg[n][k]), np.std(agg[n][k])) for k in keys}
        print(f"{n:<12}" + "".join(f"{m[k][0]:>10.3f} ±{m[k][1]:.3f}"
                                   for k in ("auc", "bal", "auprc"))
              + f"{m['acc'][0]:>12.3f}")
        rows.append(dict(model=n, **{f"{k}_{s}": v for k in keys
                                     for s, v in zip(("mean", "std"), m[k])}))

    pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
    significance(agg, names).to_csv(out / "significance.csv", index=False)
    print(f"\nSaved {out}/results.csv, {out}/significance.csv, {out}/vit_seed*.pt")


if __name__ == "__main__":
    main()
