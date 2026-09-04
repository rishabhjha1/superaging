# Explainable Deep Learning for Classifying Cognitive Superagers from Structural MRI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)

Official implementation of *Explainable Deep Learning for Classifying Cognitive Superagers from Structural MRI*. This repository benchmarks five models on T1-weighted scans (ADNI and OASIS) to separate SuperAgers from typical agers, and applies explainability tools (attention rollout, Grad-CAM, LIME) to identify distinguishing brain regions without regional priors.

## 🚀 Getting Started

Follow these steps to replicate the environment, benchmarks, and explainability stack.

### 1️⃣ Clone the Repository
```bash
git clone [https://github.com/rishabhjha1/superaging.git](https://github.com/rishabhjha1/superaging.git)
cd superaging
```
2️⃣ Set Up Environment
Create and activate the environment using either pip or conda:

Bash
conda env create -f environment.yml
conda activate superager
# OR
pip install -r requirements.txt
3️⃣ Data Preparation
Imaging data is not distributed directly. You must obtain access to ADNI and OASIS.

Ensure you have signed the necessary data use agreements.

Refer to data/README.md for label file formatting and subject ID parsing.

4️⃣ Train the Models
To reproduce the benchmark for all five models across five seeds (outputs to results/):

Bash
python train.py --labels data/labels.csv --roots data/ADNI data/OASIS --outdir results
To train a single model (e.g., ViT):

Bash
python train.py --models ViT --outdir results/vit
5️⃣ Run Explainability
Generate the explainability stack (attention rollout, Grad-CAM, LIME) for a specific checkpoint:

Bash
python explain.py --ckpt checkpoints/vit_seed42.pt \
                  --labels data/labels.csv \
                  --roots data/ADNI data/OASIS \
                  --outdir results/explain
Preprocessing: Canonical RAS reorientation, brain-masking, median/IQR normalization, and percentile clipping. Scans are formatted as 3 × 224 × 224 tensors.

Models: ViT is adapted via frozen backbones, mean-pooling view fusion, and test-time averaging.

Optimization: AdamW optimizer, cosine annealing, class-balanced cross-entropy, and early stopping.

Splits: Subject-grouped and site-stratified 5-fold cross-validation.

📄 License & Acknowledgements
License: MIT (See LICENSE). Note: ADNI and OASIS datasets are governed by their respective data use agreements.
