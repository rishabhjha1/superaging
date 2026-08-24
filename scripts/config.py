"""Configuration for the SuperAger pipeline.

Every hyperparameter reported in the paper lives here, so a run is described by
a single serialisable object. Values default to those in Sec. 3.3-3.4.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

SEED = 42


@dataclass
class Config:
    """All settings for training and evaluation."""

    # ---- paths ----
    labels_csv: str = "data/labels.csv"
    image_roots: list[str] = field(default_factory=lambda: ["data/ADNI", "data/OASIS"])
    outdir: str = "results"

    # ---- preprocessing (Sec. 3.2) ----
    size_2d: int = 224
    mask_threshold: float = 0.15      # brain mask: 15% of volume maximum
    harmonize: bool = True            # histogram reference + per-site z-scoring
    ref_size: int = 32                # downsample edge for the histogram reference
    n_ref_scans: int = 20
    cache: bool = True

    # ---- splits (Sec. 3.4) ----
    n_splits: int = 5
    seeds: list[int] = field(default_factory=lambda: [42, 1, 2, 3, 4])

    # ---- optimisation (Sec. 3.4) ----
    batch_size: int = 16
    warmup: int = 3
    patience: int = 8
    lr_head: float = 1e-4             # pretrained-backbone heads
    lr_scratch: float = 8e-4          # random init needs larger updates
    weight_decay: float = 1e-4
    vit_weight_decay: float = 5e-5
    label_smoothing: float = 0.05
    min_lr_ratio: float = 0.05
    epochs_scratch: int = 60
    epochs_finetune: int = 30
    threshold_margin: float = 0.02    # tuned threshold must beat 0.5 by >2 points

    # ---- model-specific ----
    cnn_channels: list[int] = field(default_factory=lambda: [16, 32])
    cnn_dropout: float = 0.40
    resnet_dropout: float = 0.50
    vit_hidden_dim: int = 256
    vit_dropout: float = 0.20
    vit_tta: bool = True
    densenet_depth: int = 40
    densenet_growth: int = 12
    densenet_dropout: float = 0.10
    densenet_input_size: int = 96     # the one documented exception to 224x224
    logreg_pca_dim: int = 50
    logreg_c: float = 0.1

    # ---- explainability (Sec. 3.5) ----
    lime_samples: int = 1000
    lime_segments: int = 50
    lime_compactness: float = 10.0
    grid_size: int = 6
    slice_fractions: list[float] = field(
        default_factory=lambda: [0.40, 0.45, 0.50, 0.55, 0.60])
    region_scaling: str = "peak"      # see regions.py -- "peak" means RELATIVE

    # ---- runtime ----
    seed: int = SEED
    num_workers: int = 4
    quick: bool = False               # short smoke-test run

    def __post_init__(self) -> None:
        if self.quick:
            self.epochs_scratch, self.epochs_finetune = 4, 3
            self.warmup, self.patience = 1, 2
            self.seeds = self.seeds[:2]
            self.lime_samples = 100
        if self.region_scaling not in ("peak", "raw", "sum"):
            raise ValueError(f"region_scaling must be peak|raw|sum, "
                             f"got {self.region_scaling!r}")

    # ---- (de)serialisation ----
    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
        return cls(**data)

    def merge_cli(self, overrides: dict[str, Any]) -> Config:
        """Return a copy with non-None CLI overrides applied."""
        data = asdict(self)
        data.update({k: v for k, v in overrides.items() if v is not None})
        return Config(**data)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, sort_keys=True)

    @property
    def checkpoint_dir(self) -> Path:
        return Path(self.outdir) / "checkpoints"


def load_config(yaml_path: str | None, **overrides: Any) -> Config:
    """Load a YAML config (or defaults) and apply CLI overrides."""
    base = Config.from_yaml(yaml_path) if yaml_path else Config()
    return base.merge_cli(overrides)
