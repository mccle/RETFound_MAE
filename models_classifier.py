import torch
import torch.nn as nn
import models_vit

from timm.models.layers import trunc_normal_
from util.pos_embed import interpolate_pos_embed
from pathlib import Path
from typing import Literal


def ViTClassifier(
    num_classes: int = 1,
    hidden_features: int = 512,
    dropout: float = 0.2,
    activation: Literal["GELU", "ReLU"] = "GELU",
    vit_arch: str = "vit_large_patch16",
    vit_kwargs: dict = {
        "img_size": (512, 1024),
        "num_classes": 1000,
        "drop_path_rate": 0.1,
        "global_pool": True
    },
    vit_weights: Path | str | None = '/autofs/space/crater_001/datasets/private/mee_parkinsons/models/RETFound_cfp_weights.pth',
    freeze_feature_extraction: bool = False
) -> nn.Module:
    model = getattr(models_vit, vit_arch)(**vit_kwargs)

    if vit_weights:
        checkpoint = torch.load(vit_weights, map_location="cpu")

        print(f"Loading ViT weights from {vit_weights}")
        checkpoint_model = checkpoint["model"]
        state_dict = model.state_dict()
        for k in ["head.weight", "head.bias"]:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        interpolate_pos_embed(model, checkpoint_model, vit_kwargs["img_size"])

        msg = model.load_state_dict(checkpoint_model, strict=False)

        if vit_kwargs.get("global_pool", False):
            assert set(msg.missing_keys) == {"head.weight", "head.bias", "fc_norm.weight", "fc_norm.bias"}
        else:
            assert set(msg.missing_keys) == {"head.weight", "head.bias"}

        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)

    classifier_head = models_vit.Mlp(
        in_features=model.head.in_features,
        hidden_features=hidden_features,
        out_features=num_classes,
        act_layer=getattr(nn, activation),
        drop=dropout
    )

    model.head = classifier_head

    if freeze_feature_extraction:
        for param in model.parameters():
            param.requires_grad = False

        for param in model.head.parameters():
            param.requires_grad = True


    return model
