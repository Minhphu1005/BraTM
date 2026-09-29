"""
models/__init__.py
===================
Model registry — add new models here and reference them by name in config.
"""

from .unet2d import UNet2D
from .attention_unet2d import AttentionUNet2D

__all__ = ["UNet2D", "AttentionUNet2D", "build_model"]

# Registry: config name → class
MODEL_REGISTRY = {
    "UNet2D":          UNet2D,
    "AttentionUNet2D": AttentionUNet2D,
}


def build_model(cfg: dict):
    """
    Instantiate model from config dict.

    Expected cfg keys:
      name (str): Model name from MODEL_REGISTRY
      in_channels (int)
      num_classes (int)
      unet / attention_unet: sub-dict with model-specific kwargs
    """
    name = cfg["name"]
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )

    cls = MODEL_REGISTRY[name]
    in_ch  = cfg.get("in_channels", 4)
    n_cls  = cfg.get("num_classes", 4)

    if name == "UNet2D":
        kwargs = cfg.get("unet", {})
        model = cls(
            in_channels=in_ch,
            num_classes=n_cls,
            features=kwargs.get("features", [32, 64, 128, 256, 512]),
            bilinear=kwargs.get("bilinear", True),
            dropout=kwargs.get("dropout", 0.2),
        )
    elif name == "AttentionUNet2D":
        kwargs = cfg.get("attention_unet", {})
        model = cls(
            in_channels=in_ch,
            num_classes=n_cls,
            features=kwargs.get("features", [64, 128, 256, 512]),
            dropout=kwargs.get("dropout", 0.1),
        )
    else:
        model = cls(in_channels=in_ch, num_classes=n_cls)

    n_params = model.count_parameters()
    print(f"  Model: {name} | Parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    return model
