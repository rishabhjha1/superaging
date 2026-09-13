# Explainable Deep Learning for Classifying Cognitive SuperAgers from Structural MRI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)

Official implementation of *Explainable Deep Learning for Classifying Cognitive SuperAgers from Structural MRI* (Jha, Keenan, Gawryluk, and Mbilinyi). <!-- TODO: link the paper here once an arXiv / DOI / venue URL is available --> This repository benchmarks five models on T1-weighted scans (ADNI and OASIS) to separate SuperAgers from typical agers, and applies four explainability methods (attention rollout, Grad-CAM, LIME, and grid-based regional analysis) to identify distinguishing brain regions without regional priors.

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

### 3️⃣ Data Preparation

Imaging data is not distributed directly. You must obtain access to ADNI and OASIS and sign the necessary data use agreements. See [`data/README.md`](data/README.md) for the expected directory layout, the `labels.csv` schema, and how subject IDs and sites are parsed.

### 4️⃣ Train the Models

To reproduce the benchmark for all five models across five seeds (outputs to `results/`, including `results/checkpoints/`):

```bash
python train.py --labels data/labels.csv --roots data/ADNI data/OASIS --outdir results
```

To train a single model:

```bash
python train.py --models ViT --outdir results/vit
```

Pass `--quick` for a fast smoke test (fewer epochs, seeds, and LIME samples) to verify the pipeline runs before committing to a full run.

### 5️⃣ Evaluate a Checkpoint

To reproduce Table 1 from a checkpoint without retraining — `eval.py` reconstructs the exact held-out split the checkpoint was trained under from the seed stored inside it:

```bash
python eval.py --checkpoints "results/checkpoints/vit_seed*.pt" \
               --labels data/labels.csv \
               --roots data/ADNI data/OASIS
```

Pass `--external` to score every subject instead of reconstructing the training split — the correct mode for a genuinely held-out external cohort.

### 6️⃣ Run Explainability

Generate the explainability stack (attention rollout, Grad-CAM, LIME, grid-based regional analysis; paper Table 2 and Fig. 2) for a specific checkpoint:

```bash
python explain.py --ckpt results/checkpoints/vit_seed42.pt \
                  --labels data/labels.csv \
                  --roots data/ADNI data/OASIS \
                  --outdir results/explain
```

Run `python explain.py --ckpt <checkpoint> --check-orientation ...` first on a new cohort — it writes a labelled grid overlay so you can confirm frontal/parietal are not swapped before trusting any region name.

### 7️⃣ (Optional) Interactive Demo

A Streamlit app runs the trained ViT and explainability stack on a single uploaded scan:

```bash
export SUPERAGER_CKPT=results/checkpoints/vit_seed42.pt   # optional; defaults to this path
streamlit run streamlit_app.py
```

Without a checkpoint at `SUPERAGER_CKPT`, the app still runs end to end on a randomly initialised head so the plumbing can be smoke-tested — the predictions are just not meaningful in that case.

## Method Summary

**Preprocessing:** Canonical RAS reorientation, strict brain-masking, robust median/IQR normalization, optional histogram matching, per-site z-scoring, and percentile clipping. Each subject is reduced to three orthogonal mid-plane slices (axial, coronal, sagittal), formatted as a 3 × 224 × 224 tensor.

**Models:** A classical logistic regression (PCA + linear), a from-scratch Lightweight CNN, a compact from-scratch DenseNet-40, a frozen pretrained ResNet-18, and the proposed model — a pretrained ViT-B/16 adapted to the small-sample regime via a frozen backbone, a lightweight trainable head, parameter-free mean fusion across the three planes, and four-view test-time averaging.

**Optimization:** AdamW, cosine annealing with warmup, class-balanced cross-entropy with label smoothing, and early stopping on a combined AUC / balanced-accuracy criterion.

**Splits:** Subject-grouped, site-stratified 5-fold cross-validation, repeated over 5 seeds, so no subject or acquisition site leaks across a fold.

**Explainability:** Attention rollout, Grad-CAM, and LIME applied to the trained ViT's axial view, plus a 6×6 grid-based regional analysis mapping cells to five coarse lobar regions (frontal, parietal, temporal, central, subcortical).

## Results

Held-out test performance, mean ± std over 5 seeds (paper Table 1):

| Model           | Test AUC        | Test Bal. Acc.  | Test AUPRC      | Test Acc.       |
|-----------------|------------------|------------------|------------------|------------------|
| LogReg          | 0.649 ± 0.067    | 0.621 ± 0.070    | 0.478 ± 0.095    | 0.683 ± 0.077    |
| ResNet-18       | 0.727 ± 0.125    | 0.670 ± 0.080    | 0.441 ± 0.158    | 0.753 ± 0.056    |
| DenseNet-40     | 0.806 ± 0.107    | 0.747 ± 0.078    | 0.487 ± 0.221    | 0.784 ± 0.061    |
| Lightweight CNN | 0.810 ± 0.101    | 0.730 ± 0.080    | 0.501 ± 0.221    | 0.829 ± 0.060    |
| **ViT (ours)**  | **0.841 ± 0.059**| **0.759 ± 0.058**| **0.550 ± 0.158**| **0.835 ± 0.040**|

Reproduce this table with the training command in step 4 above (`results/results_summary.csv`), or re-evaluate your own trained checkpoints with the command in step 5 (`results/eval/eval_metrics.csv`) — see [Pretrained Models](#pretrained-models) below for why no checkpoints ship with this repo. Regional attention importance and SuperAger scores (paper Table 2) are written by the explainability command in step 6 (`results/explain/regional_table.csv`).

The ViT's advantage is not a statistically separable margin over every baseline at n = 5 seeds — see `results/significance.csv` and paper Sec. 4.1 for the paired significance tests and their caveats.

## Pretrained Models

No pretrained checkpoints are bundled with this repository: the models here are trained on ADNI and OASIS scans, which we cannot redistribute derived artifacts of without checking each cohort's data use agreement, and we have not yet done so. Training a full benchmark from scratch (step 4) takes on the order of minutes to tens of minutes per model on a single GPU at this sample size (n = 200).

Checkpoints you train yourself are written to `<outdir>/checkpoints/<model>_seed<seed>.pt` (e.g. `results/checkpoints/vit_seed42.pt`) and are self-describing — `eval.py` and `explain.py` read the model name, seed, and decision threshold directly from the checkpoint.

## Project Structure

```
superaging/
├── train.py                # Entry point: benchmark all 5 models across seeds (Table 1)
├── eval.py                 # Entry point: reproduce Table 1 from a trained checkpoint
├── explain.py              # Entry point: explainability stack for one checkpoint (Table 2, Fig. 2)
├── streamlit_app.py        # Interactive single-scan demo
├── superager/              # Library code
│   ├── config.py           # All hyperparameters (Config dataclass)
│   ├── data.py             # Cohort assembly + 2.5D preprocessing pipeline
│   ├── models.py           # LogReg, Lightweight CNN, ResNet-18, DenseNet-40, ViT-B/16
│   ├── engine.py           # Training loop, metrics, age-only sanity baseline
│   ├── splits.py           # Subject-grouped, site-stratified cross-validation
│   ├── explain.py          # Attention rollout, Grad-CAM, LIME, grid analysis
│   ├── regions.py          # Grid-to-region mapping and regional aggregation
│   ├── stats.py            # Paired significance tests
│   └── plotting.py         # Figures (seed summary, explainability stack)
├── data/README.md          # Expected data layout and labels.csv schema
├── environment.yml         # Conda environment spec
└── requirements.txt        # Pip dependency spec
```

## Citation

If you use this code, please cite:

```bibtex
@misc{jha2026superager,
  title  = {Explainable Deep Learning for Classifying Cognitive {SuperAgers} from Structural {MRI}},
  author = {Jha, Rishabh and Keenan, Haley E. and Gawryluk, Jodie R. and Mbilinyi, Ashery},
  year   = {2026},
  note   = {Preprint; update with the published venue/DOI once available}
}
```

## Contributing

Issues and pull requests are welcome. Please open an issue to discuss significant changes before submitting a PR.

## 📄 License

MIT (see [LICENSE](LICENSE)). Note: ADNI and OASIS datasets are governed by their own data use agreements and are not covered by this license.
