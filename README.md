# Explainable Deep Learning for Classifying Cognitive Superagers from Structural MRI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)

Official implementation of *Explainable Deep Learning for Classifying Cognitive Superagers from Structural MRI*.

This repository benchmarks five models on T1-weighted scans (ADNI and OASIS) to separate SuperAgers from typical agers, and applies explainability tools (attention rollout, Grad-CAM, LIME) to identify distinguishing brain regions without regional priors.

![Explainability stack](Explainability%20Stack.png)

## 🚀 Getting Started

Follow these steps to replicate the environment, benchmarks, and explainability stack.

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/rishabhjha1/superaging.git
cd superaging
```

### 2️⃣ Set Up Environment

Create and activate the environment using either conda or pip:

```bash
conda env create -f environment.yml
conda activate superager
# OR
pip install -r requirements.txt
```

Run every command below from the repository root — the entry points import the
`superager` package from the working directory.

### 3️⃣ Data Preparation

Imaging data is not distributed directly. You must obtain access to ADNI and OASIS and sign the necessary data use agreements.

Refer to [data/README.md](data/README.md) for label file formatting, subject ID parsing, and the expected directory layout.

### 4️⃣ Train the Models

To reproduce the benchmark for all five models across five seeds (outputs to `results/`):

```bash
python train.py --labels data/labels.csv --roots data/ADNI data/OASIS --outdir results
```

To train a single model:

```bash
python train.py --models ViT --outdir results/vit
```

For a fast smoke test that exercises every stage on a handful of epochs and two seeds:

```bash
python train.py --quick --models ViT --outdir results/smoke
```

`train.py` writes `results_summary.csv` (Table 1), `results_per_seed.csv`, `significance.csv`, `seed_summary.png`, `site_audit.csv`, `run_manifest.json`, and ViT checkpoints under `results/checkpoints/`.

### 5️⃣ Evaluate Checkpoints

`eval.py` reproduces the reported metrics **without retraining**. Each checkpoint stores the seed it was trained under, and `eval.py` rebuilds that exact split before scoring:

```bash
python eval.py --checkpoints 'results/checkpoints/vit_seed*.pt' \
               --labels data/labels.csv \
               --roots data/ADNI data/OASIS \
               --outdir results/eval
```

A single checkpoint:

```bash
python eval.py --checkpoints results/checkpoints/vit_seed42.pt
```

For a genuinely held-out external cohort, pass `--external` to score every subject instead of reconstructing a training split:

```bash
python eval.py --checkpoints results/checkpoints/vit_seed42.pt \
               --roots data/EXTERNAL --external
```

> ⚠️ Without `--external`, a checkpoint evaluated against a *different* cohort reports optimistic numbers, because subjects it trained on will land in the reconstructed test fold. Use `--external` whenever the data is not the cohort the checkpoint was trained on.

Outputs are `eval_metrics.csv` plus per-checkpoint `predictions_*.csv`. Each row is a different seed and therefore a different test fold, so the spread across rows is seed-to-seed variance, not a confidence interval on one estimate.

### 6️⃣ Run Explainability

Generate the explainability stack (attention rollout, Grad-CAM, LIME, grid-based regional analysis) for a specific checkpoint:

```bash
python explain.py --ckpt results/checkpoints/vit_seed42.pt \
                  --labels data/labels.csv \
                  --roots data/ADNI data/OASIS \
                  --outdir results/explain
```

This writes `explainability_stack.png`, `regional_table.csv` (Table 2), `method_agreement.csv`, and `notes.txt`.

**Check orientation before quoting anatomy.** Region labels come from a 6×6 grid over a canonical-RAS axial slice. A flipped or transposed volume swaps frontal and parietal labels while leaving every number in Table 2 unchanged and entirely plausible-looking:

```bash
python explain.py --ckpt results/checkpoints/vit_seed42.pt --check-orientation
```

Inspect the overlay it writes: frontal cells must sit at the **top** of the image and parietal at the bottom.

## 📊 Results

Held-out test performance, mean ± std over 5 seeds. Reproduce with:

```bash
python train.py --labels data/labels.csv --roots data/ADNI data/OASIS --outdir results
```

| Model | AUC | Balanced Acc. | AUPRC | Accuracy |
|---|---|---|---|---|
| LogReg | — | — | — | — |
| LightweightCNN | — | — | — | — |
| ResNet18 | — | — | — | — |
| DenseNet40 | — | — | — | — |
| **ViT** | — | — | — | — |

> 📝 **These cells are placeholders.** The published Table 1 values are not stored anywhere in this repository, and they have deliberately not been guessed. Paste the numbers from the paper — or from the `results_summary.csv` that the command above writes — into this table before publishing. The column order matches `format_table` in [superager/stats.py](superager/stats.py) exactly.

Significance testing against the ViT reference (Wilcoxon signed-rank with Holm–Bonferroni correction) is written to `significance.csv`. At n=5 seeds the signed-rank test cannot fall below p=0.0625, so it is a directional check only.

Regional attention importance (Table 2) and cross-method agreement are produced by `explain.py`; the paper reports r = 0.84 between Grad-CAM and attention on that basis.

### Reading the numbers honestly

Three caveats are built into the code and reported alongside every run:

- **Age-only baseline.** `train.py` fits a logistic model on age alone (Sec. 3.1) and records it in `run_manifest.json`. Imaging models that do not clear it are recovering age, not SuperAger status.
- **Site confounding.** `site_audit.csv` flags sites containing only one class. There, site *is* the label. Site-aware stratification mitigates but cannot remove this.
- **Relative regional scores.** With the default `peak` scaling, I_R values are relative (top region ≈ 1.0), not absolute attention mass. Use `--scaling raw` for unnormalised means.

## 🧠 Pretrained Models

**No trained checkpoints are distributed with this repository.**

Model weights are derived from ADNI and OASIS scans, and both data use agreements restrict redistribution of derived data. Releasing checkpoints trained on them could also expose subject-level information. Train your own with `train.py` after obtaining data access — `results/checkpoints/vit_seed<seed>.pt` is written automatically for the ViT.

What *is* downloaded automatically is the ImageNet pretraining: `ViT_B_16_Weights.IMAGENET1K_V1` and `ResNet18_Weights.IMAGENET1K_V1` are fetched by torchvision on first use. If they are unavailable (offline machines, no cache), both models log a warning and fall back to random initialisation — the run will complete, but results will not match the paper. Check for this line:

```
WARNING  ViT pretrained weights unavailable (...); random init.
```

Each saved checkpoint contains `model_name`, `state_dict`, `seed`, and the tuned decision `threshold`, which is what lets `eval.py` rebuild the matching split and reuse the tuned operating point.

## 🖥️ Interactive Demo

A Streamlit app runs the full pipeline on a single uploaded scan:

```bash
pip install streamlit
SUPERAGER_CKPT=results/checkpoints/vit_seed42.pt streamlit run streamlit_app.py
```

Upload a T1-weighted `.nii` or `.nii.gz` and it reports P(SuperAger), per-plane attention rollout maps, and the regional table. Without a checkpoint it still runs end to end on an untrained head, which is useful for verifying the install but produces meaningless numbers.

The demo calls the same `build_input`, `attention_rollout`, and `region_scores` functions as `explain.py`. One stage is necessarily skipped: the per-site z-score (Sec. 3.2, stage 6) needs a whole cohort and cannot be applied to a single scan.

## 🔬 Method

**Preprocessing:** Canonical RAS reorientation, brain-masking, median/IQR normalization, optional histogram matching, and percentile clipping. Scans are formatted as 3 × 224 × 224 tensors holding the axial, coronal, and sagittal mid-slices.

**Models:** ViT-B/16 adapted via a frozen backbone (86.0M total parameters, 0.20M trainable under the shipped `vit_hidden_dim=256`), parameter-free mean-pooling view fusion, and four-view test-time averaging. Benchmarked against LogReg, a lightweight CNN, frozen ResNet-18, and DenseNet-40.

> ⚠️ The class docstring in [superager/models.py](superager/models.py) describes the head as "~0.6M trainable", which corresponds to `vit_hidden_dim=768`, not the `256` that [superager/config.py](superager/config.py) actually defaults to. The measured count at the shipped default is 0.20M. Reconcile the default against the paper before publishing — if the reported results used a 768-wide head, the default config does not reproduce them.

**Optimization:** AdamW, cosine annealing with warmup, class-balanced cross-entropy with label smoothing, and early stopping.

**Splits:** Subject-grouped and site-stratified 5-fold cross-validation. One scan per subject, so longitudinal OASIS-2 sessions cannot straddle a split.

**Explainability:** Attention rollout with residual augmentation, Grad-CAM on final-block token embeddings, LIME over SLIC superpixels, and a 6×6 grid-based regional analysis swept across five slice depths.

## 📁 Project Structure

```
superaging/
├── train.py              # benchmark all five models across seeds -> Table 1
├── eval.py               # score released checkpoints without retraining
├── explain.py            # attention rollout, Grad-CAM, LIME, regional analysis
├── streamlit_app.py      # interactive single-scan demo
├── superager/
│   ├── config.py         # every hyperparameter, as one serialisable dataclass
│   ├── data.py           # cohort assembly + 2.5D multi-plane preprocessing
│   ├── splits.py         # subject-grouped, site-stratified splits
│   ├── models.py         # LogReg, LightweightCNN, ResNet18, DenseNet40, ViT
│   ├── engine.py         # training loop, TTA inference, metrics, thresholds
│   ├── explain.py        # rollout, Grad-CAM, LIME, grid analysis
│   ├── regions.py        # 6x6 grid -> five coarse regions (Table 2)
│   ├── stats.py          # aggregation, Wilcoxon + Holm-Bonferroni
│   └── plotting.py       # explainability stack figure, orientation check
├── data/README.md        # label format and directory layout
├── environment.yml
└── requirements.txt
```

Configuration is centralised in [superager/config.py](superager/config.py). Every entry point accepts `--config path/to.yaml`, and CLI flags override the YAML.

## 📝 Citation

If you use this code, please cite:

```bibtex
@article{jha_superagers,
  title   = {Explainable Deep Learning for Classifying Cognitive Superagers
             from Structural MRI},
  author  = {Jha, Rishabh},
  journal = {TODO},
  year    = {TODO},
  note    = {Code: https://github.com/rishabhjha1/superaging}
}
```

> 📝 Venue and year are placeholders — fill them in from the published record. The author list has been taken from the repository's commit history and should be corrected to match the paper.

Please also follow the ADNI and OASIS citation requirements for the imaging data.

## 📄 License

[MIT](LICENSE).

Note: ADNI and OASIS datasets are governed by their respective data use agreements, and ImageNet pretrained weights carry their own licenses. See [LICENSE](LICENSE) for details.
