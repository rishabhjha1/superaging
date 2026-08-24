"""Model architectures (paper Sec. 3.3).

Five models of increasing representational capacity on identical
(3, 224, 224) inputs, with one documented exception: DenseNet-40 trains on
96 x 96, following the original CIFAR-scale design and to limit overfitting on
high-resolution slices at this sample size.

    Model         Params   Trainable   Pretrained
    LogReg          --        --         no
    SimpleCNN     ~0.02M    all          no
    ResNet-18     ~11.2M    ~0.001M      ImageNet
    DenseNet-40   ~1.02M    all          no
    ViT-B/16      ~86M      ~0.6M        ImageNet   <- proposed
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression as SKLogReg
from sklearn.preprocessing import StandardScaler

from .config import SEED, Config

log = logging.getLogger(__name__)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to_imagenet(x: torch.Tensor) -> torch.Tensor:
    """Grayscale (B, 1, H, W) -> ImageNet-normalised 3-channel."""
    x3 = x.repeat(1, 3, 1, 1)
    return (x3 - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)


def _init_linear(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)


# ============================================================ BASELINES
class LogReg:
    """Linear reference: PCA on the flattened, standardised axial slice.

    Establishes what is recoverable without any spatial modelling.
    """

    name = "LogReg"
    pretrained_backbone = False
    is_torch = False

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.scaler = StandardScaler()
        self.pca: PCA | None = None
        self.clf = SKLogReg(C=cfg.logreg_c, class_weight="balanced", max_iter=2000)

    def fit(self, X: np.ndarray, y: np.ndarray) -> LogReg:
        flat = X[:, 0].reshape(len(X), -1)          # channel 0 = axial
        z = self.scaler.fit_transform(flat)
        n = max(2, min(self.cfg.logreg_pca_dim, z.shape[0] - 1, z.shape[1]))
        self.pca = PCA(n_components=n, random_state=SEED)
        self.clf.fit(self.pca.fit_transform(z), y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        flat = X[:, 0].reshape(len(X), -1)
        z = self.pca.transform(self.scaler.transform(flat))
        return self.clf.predict_proba(z)[:, 1]


class SimpleCNN(nn.Module):
    """From-scratch two-block CNN applied per plane, features concatenated."""

    name = "SimpleCNN"
    pretrained_backbone = False
    is_torch = True

    def __init__(self, cfg: Config):
        super().__init__()
        c1, c2 = cfg.cnn_channels

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(), nn.MaxPool2d(2))

        self.encoder = nn.Sequential(block(1, c1), block(c1, c2))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(c2 * 3, 64), nn.ReLU(),
            nn.Dropout(cfg.cnn_dropout), nn.Linear(64, 2))
        _init_linear(self.head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [self.encoder(x[:, i:i + 1]) for i in range(3)]
        return self.head(torch.cat(feats, dim=1))


class ResNet18Net(nn.Module):
    """Frozen ImageNet ResNet-18; per-plane features mean-pooled.

    Isolates what generic pretrained features contribute without adaptation.
    """

    name = "ResNet18"
    pretrained_backbone = True
    is_torch = True

    def __init__(self, cfg: Config):
        super().__init__()
        try:
            weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
            backbone = torchvision.models.resnet18(weights=weights)
        except Exception as exc:  # offline / no cached weights
            log.warning("ResNet-18 pretrained weights unavailable (%s); random init.", exc)
            backbone = torchvision.models.resnet18(weights=None)

        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        for p in backbone.parameters():
            p.requires_grad = False

        self.backbone = backbone
        self.head = nn.Sequential(nn.LayerNorm(feat_dim),
                                  nn.Dropout(cfg.resnet_dropout),
                                  nn.Linear(feat_dim, 2))
        _init_linear(self.head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        f = self.backbone(to_imagenet(x.reshape(B * 3, 1, *x.shape[2:])))
        return self.head(f.reshape(B, 3, -1).mean(dim=1))


class DenseNet40(nn.Module):
    """Compact from-scratch DenseNet: depth 40, k=12, theta=1.0, non-bottleneck.

    Runs at 96 x 96, roughly one fifth the pixel count of the other models,
    yet stays competitive -- a purpose-sized architecture is a strong
    from-scratch baseline at n = 200.
    """

    name = "DenseNet40"
    pretrained_backbone = False
    is_torch = True

    def __init__(self, cfg: Config):
        super().__init__()
        self.input_size = cfg.densenet_input_size
        self.drop = cfg.densenet_dropout
        k = cfg.densenet_growth
        n_layers = (cfg.densenet_depth - 4) // 3

        channels = 16
        self.stem = nn.Conv2d(3, channels, 3, padding=1, bias=False)
        stages: list[nn.Module] = []
        for block in range(3):
            layers = []
            for _ in range(n_layers):
                layers.append(nn.ModuleDict(dict(
                    bn=nn.BatchNorm2d(channels),
                    conv=nn.Conv2d(channels, k, 3, padding=1, bias=False))))
                channels += k
            stages.append(nn.ModuleList(layers))
            if block < 2:  # transition (theta = 1.0, so channels are preserved)
                stages.append(nn.ModuleDict(dict(
                    bn=nn.BatchNorm2d(channels),
                    conv=nn.Conv2d(channels, channels, 1, bias=False))))
        self.stages = nn.ModuleList(stages)
        self.bn_final = nn.BatchNorm2d(channels)
        self.head = nn.Sequential(nn.Dropout(0.30), nn.Linear(channels, 2))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_size:
            x = F.interpolate(x, (self.input_size,) * 2, mode="bilinear",
                              align_corners=False)
        x = self.stem(x)
        for stage in self.stages:
            if isinstance(stage, nn.ModuleList):            # dense block
                for layer in stage:
                    out = layer["conv"](F.relu(layer["bn"](x)))
                    if self.drop > 0:
                        out = F.dropout(out, self.drop, self.training)
                    x = torch.cat([x, out], dim=1)
            else:                                           # transition
                x = F.avg_pool2d(stage["conv"](F.relu(stage["bn"](x))), 2)
        x = F.adaptive_avg_pool2d(F.relu(self.bn_final(x)), 1).flatten(1)
        return self.head(x)


# ============================================================ PROPOSED MODEL
class ViTNet(nn.Module):
    """ViT-B/16 adapted to a few hundred subjects (Sec. 3.3).

    Five adaptations, each scaling capacity down to the data:
      (i)   frozen backbone -- 86M parameters drop to ~0.6M trainable
      (ii)  head-only optimisation at lr 1e-4, weight decay 5e-5, so no drift
            from the ImageNet initialisation under the natural-image-to-MRI shift
      (iii) mild augmentation only (see engine.augment)
      (iv)  parameter-free mean fusion across the three planes
      (v)   four-view test-time augmentation
    """

    name = "ViT"
    pretrained_backbone = True
    is_torch = True

    def __init__(self, cfg: Config):
        super().__init__()
        if cfg.size_2d != 224:
            raise ValueError(f"ViT-B/16 requires size_2d=224, got {cfg.size_2d}")
        try:
            weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
            backbone = torchvision.models.vit_b_16(weights=weights)
        except Exception as exc:
            log.warning("ViT pretrained weights unavailable (%s); random init.", exc)
            backbone = torchvision.models.vit_b_16(weights=None)

        feat_dim = backbone.heads.head.in_features
        backbone.heads = nn.Identity()
        for p in backbone.parameters():
            p.requires_grad = False

        self.backbone = backbone
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, cfg.vit_hidden_dim), nn.GELU(),
            nn.Dropout(cfg.vit_dropout),
            nn.Linear(cfg.vit_hidden_dim, 2))
        _init_linear(self.head)

    def plane_features(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) -> (B, 3, 768) per-plane CLS embeddings."""
        B = x.shape[0]
        f = self.backbone(to_imagenet(x.reshape(B * 3, 1, *x.shape[2:])))
        return f.reshape(B, 3, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.plane_features(x).mean(dim=1))


# ============================================================ REGISTRY
MODELS: dict[str, type] = {
    "LogReg": LogReg,
    "SimpleCNN": SimpleCNN,
    "ResNet18": ResNet18Net,
    "DenseNet40": DenseNet40,
    "ViT": ViTNet,
}

MODEL_ORDER = ["LogReg", "SimpleCNN", "ResNet18", "DenseNet40", "ViT"]


def build_model(name: str, cfg: Config):
    if name not in MODELS:
        raise KeyError(f"Unknown model {name!r}; available: {list(MODELS)}")
    return MODELS[name](cfg)


def count_parameters(model: nn.Module) -> tuple[float, float]:
    """(total, trainable) parameter counts in millions."""
    total = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    return total, trainable


def epochs_for(model_name: str, cfg: Config) -> int:
    """From-scratch models train longer than pretrained heads."""
    cls = MODELS[model_name]
    return cfg.epochs_finetune if cls.pretrained_backbone else cfg.epochs_scratch
