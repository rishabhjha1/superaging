# Explainable Deep Learning for Classifying Cognitive Superagers from Structural MRI


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)

This repository is the official implementation of *Explainable Deep Learning for
Classifying Cognitive Superagers from Structural MRI*.

SuperAgers are older adults whose episodic memory matches that of people decades
younger. We ask two questions. First, can SuperAgers be separated from typical
agers when only a few hundred subjects are available? We benchmark five models of
increasing capacity on 200 T1-weighted scans from ADNI and OASIS. Second, and
more consequentially, which brain regions distinguish them? We apply attention
rollout, Grad-CAM, LIME, and grid-based regional analysis to the best model and
find convergence on parietal, temporal, and frontal cortex — regions already
implicated in cognitive preservation, recovered by a model given no regional
priors.

![Explainability stack](explainability stack.png)

## Requirements

To install requirements:

```bash
pip install -r requirements.txt
```

Or with conda:

```bash
conda env create -f environment.yml
conda activate superager
```



**Data.** ADNI and OASIS both require a signed data use agreement, so no imaging
data is distributed here. See [`data/README.md`](data/README.md) for how to
request access, what the label file should contain, and how subject IDs are
parsed. To verify the pipeline runs without any agreement:



## Training

To train the models in the paper, run:

```bash
python train.py --labels data/labels.csv --roots data/ADNI data/OASIS --outdir results
```

This runs all five models across five seeds under one 2.5D
preprocessing pipeline, and writes `results/results_summary.csv`,
`results_per_seed.csv`, `significance.csv`, `seed_summary.png`,
`run_manifest.json`, and one ViT checkpoint per seed in `results/checkpoints/`.

Train a single model:

```bash
python train.py --models ViT --outdir results/vit
```



<details>
<summary><b>Training procedure and hyperparameters</b></summary>

**Preprocessing (Sec. 3.2).** Volumes are reoriented to canonical RAS, brain-masked
(15%-of-max threshold → hole-fill → largest connected component), normalised by
median/IQR inside the mask, optionally histogram-matched to a reference pooled
from 20 scans, clipped to the [1, 99] percentile range, z-scored per acquisition
site, then clipped again. The second clip must follow the z-scoring, which
otherwise pushes voxels into the tails. Each subject becomes a 3 × 224 × 224
tensor of axial, coronal, and sagittal mid-slices.

**Models (Sec. 3.3).**

 |

The ViT is adapted to ~200 subjects five ways: the backbone is frozen (86M → 0.6M
trainable); only the head trains, at lr 1e-4 with weight decay 5e-5, so nothing
drifts from the ImageNet initialisation under the natural-image-to-MRI shift;
aggressive geometric augmentation is dropped; the three planes are fused by mean
pooling, which adds no parameters and treats views as exchangeable evidence; and
inference averages four test-time views.

**Optimisation (Sec. 3.4).** AdamW; cosine annealing with a 3-epoch warmup
(min-LR ratio 0.05); class-balanced cross-entropy with label smoothing ε = 0.05;
early stopping (patience 8) on 0.5·AUC + 0.5·balanced accuracy. From-scratch
models use lr 8e-4 and up to 60 epochs because randomly initialised weights need
larger updates; pretrained heads use lr 1e-4 and up to 30. Batch size 16.

**Splits (Sec. 3.4).** Grouped by subject so no subject spans a train/eval
boundary, and stratified on `(label, site)` so no fold becomes site-pure —
otherwise a scanner signature acts as a free label. One outer test fold, inner
5-fold CV; test predictions are the mean over the inner-fold checkpoints. The
decision threshold is chosen on pooled validation predictions and adopted only if
it beats 0.5 by more than two points.

</details>

## Evaluation

To evaluate released checkpoints without retraining:

```bash
bash scripts/download_pretrained.sh checkpoints
python eval.py --checkpoints 'checkpoints/vit_seed*.pt' \
               --labels data/labels.csv --roots data/ADNI data/OASIS
```

`eval.py` reconstructs the exact split each checkpoint was trained under (the
seed is stored in the checkpoint), runs the fold ensemble, and writes
`eval_metrics.csv` plus per-subject predictions.



## Explainability


```bash

python explain.py --ckpt checkpoints/vit_seed42.pt \
                  --labels data/labels.csv --roots data/ADNI data/OASIS \
                  --outdir results/explain
```



Reproduce with:

```bash
python explain.py --ckpt checkpoints/vit_seed42.pt \
                  --labels data/labels.csv --roots data/ADNI data
```

Grad-CAM importance is positively associated with attention across the five
regions (r = 0.84); LIME independently marks parietal and temporal superpixels as
the strongest positive contributors (local fidelity 0.78).



## Repository layout

```

train.py              benchmark all five models          -> Table 1
eval.py               evaluate checkpoints, no retraining
explain.py            explainability stack               -> Fig. 2, Table 2


```



## Acknowledgements

Data used in this project were obtained from the Alzheimer's Disease
Neuroimaging Initiative (ADNI) and the Open Access Series of Imaging Studies
(OASIS). ADNI investigators contributed to the design and implementation of ADNI
and provided data but did not participate in the analysis or writing of this
report. A complete listing is available at
<https://adni.loni.usc.edu/wp-content/uploads/how_to_apply/ADNI_Acknowledgement_List.pdf>.

## License

MIT — see [LICENSE](LICENSE). This covers the software only; the ADNI and OASIS
datasets remain under their own data use agreements.
