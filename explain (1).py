#!/usr/bin/env python
"""Generate the explainability stack for a trained ViT (paper Fig. 2, Table 2).

Examples
--------
Check slice orientation — run this ONCE per dataset, before anything else:

    python explain.py --ckpt checkpoints/vit_seed42.pt --check-orientation

Produce the figure and regional table:

    python explain.py --ckpt checkpoints/vit_seed42.pt \
                      --labels data/labels.csv --roots data/ADNI data/OASIS

Report unscaled attention weights instead of peak-scaled ones:

    python explain.py --ckpt checkpoints/vit_seed42.pt --scaling raw

WARNING — read before quoting anatomy
-------------------------------------
The 6x6 grid assumes a canonical-RAS axial slice with frontal at the top. A
flipped or transposed volume swaps the frontal and parietal labels while leaving
every number in the table unchanged and plausible-looking. --check-orientation
is the only way to see this.

Importance is peak-scaled by default: the top region sits at ~1.0 by
construction and the rest are fractions of it. Report values as *relative*
attention. The grid is a coarse lobar proxy, not an atlas-registered
parcellation.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from superager.config import load_config
from superager.data import (
    build_histogram_reference,
    collect_records,
    extract_slice,
    load_volume,
    resize_2d,
)
from superager.engine import get_device, set_seed
from superager.explain import (
    AXIAL,
    attention_rollout,
    build_input,
    build_regional_table,
    cell_predictions,
    grad_cam,
    load_vit,
    method_agreement,
    regional_probabilities,
    run_lime,
)
from superager.plotting import explainability_stack, orientation_check

log = logging.getLogger("explain")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", "--checkpoint", dest="ckpt", required=True)
    p.add_argument("--config", help="YAML config; CLI flags override it")
    p.add_argument("--labels", dest="labels_csv")
    p.add_argument("--roots", dest="image_roots", nargs="+")
    p.add_argument("--outdir", default="results/explain")
    p.add_argument("--n-subjects", type=int, default=30,
                   help="Subjects pooled into the regional table")
    p.add_argument("--exemplar", type=int, default=0,
                   help="Index of the subject shown in the figure panels")
    p.add_argument("--scaling", choices=["peak", "raw", "sum"], default=None,
                   help="Regional scaling; 'peak' (default) reports RELATIVE attention")
    p.add_argument("--check-orientation", action="store_true",
                   help="Write the labelled grid overlay and exit")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")

    overrides = {k: v for k, v in vars(args).items()
                 if k in ("labels_csv", "image_roots")}
    if args.scaling:
        overrides["region_scaling"] = args.scaling
    cfg = load_config(args.config, **overrides)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    device = get_device()

    df = collect_records(cfg.labels_csv, cfg.image_roots)

    # ---- orientation check: cheap, and the only guard against silent mislabelling ----
    if args.check_orientation:
        volume = load_volume(df.path.iloc[args.exemplar])
        slice_2d = resize_2d(extract_slice(volume, "axial", 0.5), cfg.size_2d)
        path = orientation_check(slice_2d, volume.shape, outdir / "orientation_check.png")
        print(f"\nWrote {path}")
        print("Inspect it before quoting anatomy: frontal cells must sit at the "
              "TOP of the image and parietal at the bottom. If they are swapped, "
              "every region name in Table 2 is wrong even though the numbers "
              "will look entirely reasonable.")
        return

    model, threshold = load_vit(args.ckpt, cfg, device)
    reference = (build_histogram_reference(df.path.tolist(), cfg.ref_size,
                                           cfg.n_ref_scans, threshold=cfg.mask_threshold)
                 if cfg.harmonize else None)

    row = df.iloc[args.exemplar]
    log.info("Exemplar: %s (%s, label=%d)", row.subject, row.cohort, row.label)
    x_np = build_input(row.path, cfg, 0.5, reference)
    x = torch.from_numpy(x_np[None]).float().to(device)

    log.info("Attention rollout ...")
    rollout = attention_rollout(model, x)

    log.info("Grad-CAM ...")
    cam, probs = grad_cam(model, x)

    log.info("LIME (%d perturbations, %d superpixels) ...",
             cfg.lime_samples, cfg.lime_segments)
    lime = run_lime(model, x_np, cfg, device)

    log.info("Grid-based regional analysis ...")
    cells = cell_predictions(model, x_np, cfg, device)
    region_probs = regional_probabilities(cells)

    log.info("Regional table over %d subjects x %d depths ...",
             min(args.n_subjects, len(df)), len(cfg.slice_fractions))
    table = build_regional_table(model, df.path.tolist()[:args.n_subjects],
                                 cfg, device, reference)
    table.to_csv(outdir / "regional_table.csv", index=False)
    print("\nTable 2 — regional attention importance "
          f"({cfg.region_scaling} scaling)\n{table.to_string(index=False)}")

    agreement = method_agreement(rollout, cam, lime["saliency"], cfg.region_scaling)
    agreement.to_csv(outdir / "method_agreement.csv", index=False)
    print(f"\nCross-method agreement\n{agreement.to_string(index=False)}")

    figure = explainability_stack(
        dict(slice=x_np[AXIAL], cell_probs=cells, region_probs=region_probs,
             rollout=rollout, gradcam=cam, probs=probs, lime=lime),
        outdir / "explainability_stack.png", cfg.grid_size)

    with open(outdir / "notes.txt", "w") as f:
        f.write(
            f"checkpoint       : {args.ckpt}\n"
            f"decision threshold: {threshold:.3f}\n"
            f"exemplar         : {row.subject} ({row.cohort}, label={row.label})\n"
            f"depth sweep      : {cfg.slice_fractions}\n"
            f"subjects pooled  : {min(args.n_subjects, len(df))}\n"
            f"I_R scaling      : {cfg.region_scaling}\n"
            f"LIME fidelity    : {lime['fidelity']:.3f} "
            f"({lime['n_pos']} pro-SuperAger / {lime['n_neg']} pro-Normal)\n\n"
            "CAVEATS\n"
            "  - The grid is a coarse lobar proxy, not an atlas parcellation.\n"
            "  - Peak-scaled I_R is RELATIVE attention (top region ~1.0).\n"
            "  - n=5 regions: agreement coefficients show rank agreement only.\n"
            "  - One trained model: this is agreement among METHODS, not evidence\n"
            "    that the same regions surface on every training run.\n")

    print(f"\nWrote {figure}, regional_table.csv, method_agreement.csv, notes.txt")


if __name__ == "__main__":
    main()
