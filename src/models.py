"""Pabrik model CNN: arsitektur transfer learning + baseline CNN dari nol.

Setiap entry di ``ARCHITECTURES`` mengembalikan model dengan ``classifier``
berupa ``nn.Sequential`` yang bisa dibekukan/dibuka secara seragam, sehingga
skrip training hanya perlu tahu nama arsitektur.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torchvision import models

from .config import NUM_CLASSES

IMAGENET_WEIGHTS: Dict[str, Optional[str]] = {
    "resnet18": "ResNet18_Weights",
    "resnet34": "ResNet34_Weights",
    "resnet50": "ResNet50_Weights",
    "efficientnet_b0": "EfficientNet_B0_Weights",
    "mobilenet_v3_large": "MobileNet_V3_Large_Weights",
    "densenet121": "DenseNet121_Weights",
    "convnext_tiny": "ConvNeXt_Tiny_Weights",
    "vgg16_bn": "VGG16_BN_Weights",
}


def _replace_classifier(model: nn.Module, name: str, dropout: float) -> nn.Module:
    """Ganti kepala klasifikasi ImageNet dengan kepala 10 kelas."""
    if name.startswith("resnet"):
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, NUM_CLASSES))
    elif name.startswith("efficientnet") or name.startswith("mobilenet"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)
    elif name.startswith("densenet"):
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, NUM_CLASSES))
    elif name.startswith("convnext"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)
    elif name.startswith("vgg"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)
    else:  # pragma: no cover
        raise ValueError(f"Tidak tahu cara mengganti classifier {name!r}")
    return model


def build_model(
    name: str,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    dropout: float = 0.3,
) -> nn.Module:
    """Bangun model berdasarkan nama arsitektur."""
    name = name.lower()

    if name == "custom_cnn":
        return CustomCNN(num_classes=num_classes, dropout=dropout)

    if name not in IMAGENET_WEIGHTS:
        raise ValueError(
            f"Arsitektur {name!r} tidak dikenal. Pilihan: "
            f"{sorted(list(IMAGENET_WEIGHTS) + ['custom_cnn'])}"
        )

    weights = None
    if pretrained:
        weights = getattr(models, IMAGENET_WEIGHTS[name]).DEFAULT

    factory = getattr(models, name)
    model = factory(weights=weights)
    if pretrained:
        for param in model.parameters():
            param.requires_grad = False
    return _replace_classifier(model, name, dropout)


def get_param_groups(model: nn.Module, *, head_lr: float, backbone_lr: float) -> List[Dict]:
    """Pisahkan parameter kepala dan backbone untuk learning rate berbeda."""
    head_params, backbone_params = [], []
    head_prefixes = ("fc.", "classifier.")
    for param_name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param_name.startswith(head_prefixes):
            head_params.append(param)
        else:
            backbone_params.append(param)
    groups: List[Dict] = [{"params": head_params, "lr": head_lr}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": backbone_lr})
    return groups


def unfreeze_backbone(model: nn.Module, last_blocks: Optional[int] = None) -> None:
    """Buka parameter backbone.

    last_blocks=None -> seluruh backbone dilatih ulang (fine-tuning penuh).
    last_blocks=N    -> hanya N blok terakhir yang dilatih (hemat & anti overfit).
    """
    for param in model.parameters():
        param.requires_grad = True
    if last_blocks is None:
        return

    named = list(model.named_parameters())
    block_names: List[str] = []
    for param_name, _ in named:
        parts = param_name.split(".")
        if len(parts) > 1:
            block_names.append(".".join(parts[:2]))
    unique_blocks = list(dict.fromkeys(block_names))
    keep = set(unique_blocks[-last_blocks:])
    for param_name, param in named:
        prefix = ".".join(param_name.split(".")[:2])
        if not param_name.startswith(("fc.", "classifier.")) and prefix not in keep:
            param.requires_grad = False


class CustomCNN(nn.Module):
    """CNN 4-blok dari nol sebagai baseline pembanding (tanpa pretrained)."""

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.4) -> None:
        super().__init__()
        self.features = nn.Sequential(
            self._block(3, 32),
            self._block(32, 64),
            self._block(64, 128),
            self._block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        self._init_weights()

    @staticmethod
    def _block(cin: int, cout: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))


def build_feature_extractor(name: str, pretrained: bool = True) -> Tuple[nn.Module, int]:
    """Kembalikan model dengan kepala klasifikasi diganti Identity.

    Output forward menjadi representasi penultimate (dimensi fitur), dipakai
    untuk linear probe / analisis representasi.
    """
    model = build_model(name, pretrained=pretrained)
    if name.startswith("resnet"):
        dim = model.fc[-1].in_features
        model.fc = nn.Identity()
    elif name.startswith(("efficientnet", "mobilenet", "convnext", "vgg")):
        dim = model.classifier[-1].in_features
        model.classifier[-1] = nn.Identity()
    elif name.startswith("densenet"):
        dim = model.classifier[-1].in_features
        model.classifier = nn.Identity()
    else:  # pragma: no cover
        raise ValueError(f"Tidak mendukung feature extractor untuk {name!r}")
    return model, int(dim)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Hitung jumlah parameter model."""
    return sum(
        p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only)
    )


def model_summary(name: str, num_classes: int = NUM_CLASSES) -> Dict[str, object]:
    """Ringkasan ringan untuk pemilihan model tanpa harus melatih."""
    model = build_model(name, num_classes=num_classes, pretrained=False)
    total = count_parameters(model, trainable_only=False)
    trainable = count_parameters(model, trainable_only=True)
    return {"name": name, "params_total": total, "params_trainable": trainable}
